"""Undo of the last adjudication batch, rebuilt by audit replay.

An undo never reverse-edits the current working JSON. Instead it rebuilds
the merge order's state from scratch:

  1. the **immutable initial conflict snapshot** (``source_conflicts``) and
     a fresh ``merge3(base, left, right)`` give the original working
     document and the original open conflicts;
  2. the audit chain is replayed in sequence order. Each adjudication
     event (``resolutions`` / ``rule-resolutions`` / ``reuse-resolutions`` /
     ``draft-commit``) reapplies exactly the pointers it originally closed;
     events already compensated by a previous ``resolution-undo`` (their
     seq appears in some undo event's ``detail.undone_seq``) are skipped;
  3. undoing one more batch means marking the most recent *not yet
     compensated* adjudication as compensated and rebuilding again. The
     pointers that batch closed are therefore reopened with their original
     base/left/right candidates and the document returns to the state a
     merge that had never seen that batch would hold.

Everything here is pure and in-memory; the caller persists the rebuilt
document/conflicts and appends the compensating undo event in one
transaction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from . import audit
from .merge import merge3, parse_pointer, set_at, delete_at
from .patch import canonical_json

# Action recorded by a reuse event; mirrors reuse.ACTION_SET / ACTION_DELETE.
_ACTION_SET = "set"
_ACTION_DELETE = "delete"


class ReplayInconsistent(Exception):
    """The stored audit chain cannot rebuild the merge order's state.

    Indicates tampered or corrupt audit data: the undo is rejected and no
    state is written.
    """


@dataclass(frozen=True)
class ReplayedEvent:
    """One audit event in the form the replay driver needs."""

    seq: int
    method: str
    pointers: tuple[str, ...]
    detail: dict


def load_events(conn, merge_id: str) -> list[ReplayedEvent]:
    """Read a merge order's events in sequence order."""
    rows = conn.execute(
        "SELECT seq, method, pointers, detail FROM merge_audit_events "
        "WHERE merge_id = ? ORDER BY seq ASC",
        (merge_id,),
    ).fetchall()
    events = []
    for row in rows:
        events.append(
            ReplayedEvent(
                seq=row["seq"],
                method=row["method"],
                pointers=tuple(json.loads(row["pointers"])),
                detail=json.loads(row["detail"]),
            )
        )
    return events


def compensated_seqs(events: list[ReplayedEvent]) -> set[int]:
    """Seq numbers of adjudication events already compensated by an undo."""
    seqs: set[int] = set()
    for event in events:
        if event.method == audit.METHOD_RESOLUTION_UNDO:
            undone = event.detail.get("undone_seq")
            if isinstance(undone, int):
                seqs.add(undone)
    return seqs


def latest_live_adjudication(
    events: list[ReplayedEvent], compensated: set[int]
) -> ReplayedEvent | None:
    """Most recent adjudication not yet compensated (undo events skipped).

    Walks newest-first; an undo event can itself never be undone — a later
    undo simply skips past it to the next live verdict.
    """
    for event in reversed(events):
        if event.method == audit.METHOD_RESOLUTION_UNDO:
            continue
        if event.seq in compensated:
            continue
        return event
    return None


def initial_state(row) -> tuple[Any, list[dict]]:
    """The original merge3 document and the immutable initial conflict list.

    ``row`` is the merges table row. The persisted snapshot is the
    authoritative initial conflict set; merge3 is recomputed from base/left/
    right to recover the initial placeholder document and is cross-checked
    against the snapshot so a corrupt snapshot cannot silently diverge.
    """
    base = json.loads(row["base_doc"])
    left = json.loads(row["left_doc"])
    right = json.loads(row["right_doc"])
    document, computed = merge3(base, left, right)
    if row["source_conflicts"] is not None:
        initial_conflicts = json.loads(row["source_conflicts"])
        computed_view = {c["pointer"]: canonical_json(c) for c in computed}
        stored_view = {c["pointer"]: canonical_json(c) for c in initial_conflicts}
        if computed_view != stored_view:
            # The immutable snapshot no longer agrees with a fresh merge:
            # rebuilding would be meaningless, so refuse rather than guess.
            raise ReplayInconsistent(
                "initial conflict snapshot does not match base/left/right"
            )
    else:
        # Rows created before snapshots existed: the recomputed conflicts
        # are the only initial set available.
        initial_conflicts = computed
    return document, initial_conflicts


def _event_items(event: ReplayedEvent) -> list[dict]:
    """Translate one stored adjudication event back into replay steps.

    Each step is either a normal ``_apply_batch``-style item
    (choice/value/value_provided) or a reuse deletion marker
    ``{"pointer": ..., "reuse_delete": True}``, which the driver resolves
    against the missing side of the conflict at replay time.
    """
    detail = event.detail
    items = detail.get("items", [])
    steps: list[dict] = []

    if event.method == audit.METHOD_RESOLUTIONS:
        for entry in items:
            if "choice" in entry:
                steps.append(
                    {
                        "pointer": entry["pointer"],
                        "choice": entry["choice"],
                        "value": None,
                        "value_provided": False,
                    }
                )
            else:
                steps.append(
                    {
                        "pointer": entry["pointer"],
                        "choice": None,
                        "value": entry["value"],
                        "value_provided": True,
                    }
                )
    elif event.method == audit.METHOD_RULE_RESOLUTIONS:
        for entry in items:
            steps.append(
                {
                    "pointer": entry["pointer"],
                    "choice": entry["side"],
                    "value": None,
                    "value_provided": False,
                }
            )
    elif event.method == audit.METHOD_DRAFT_COMMIT:
        for entry in items:
            if "choice" in entry:
                steps.append(
                    {
                        "pointer": entry["pointer"],
                        "choice": entry["choice"],
                        "value": None,
                        "value_provided": False,
                    }
                )
            else:
                steps.append(
                    {
                        "pointer": entry["pointer"],
                        "choice": None,
                        "value": entry["value"],
                        "value_provided": True,
                    }
                )
    elif event.method == audit.METHOD_REUSE_RESOLUTIONS:
        for entry in items:
            if entry.get("action") == _ACTION_DELETE:
                # The source result no longer held this path: the source
                # honoured a deletion. Matched conflicts have identical
                # ``missing`` semantics, so delete by choosing the missing
                # side of *this* conflict.
                steps.append({"pointer": entry["pointer"], "reuse_delete": True})
            else:
                steps.append(
                    {
                        "pointer": entry["pointer"],
                        "choice": None,
                        "value": entry["value"],
                        "value_provided": True,
                    }
                )
    else:  # pragma: no cover - unknown methods cannot enter via CHECK
        raise ReplayInconsistent(f"unknown adjudication method {event.method!r}")

    if {step["pointer"] for step in steps} != set(event.pointers):
        raise ReplayInconsistent(
            f"event {event.seq} detail does not cover its recorded pointers"
        )
    return steps


def _apply_step(document, conflicts, step: dict):
    """Apply one replay step to the in-memory document/conflict set."""
    pointer = step["pointer"]
    conflict = conflicts.pop(pointer, None)
    if conflict is None:
        raise ReplayInconsistent(
            f"event closes {pointer!r} but that conflict is not open in replay"
        )
    tokens = parse_pointer(pointer)
    if step.get("reuse_delete"):
        missing = conflict.get("missing") or []
        side = next(
            (name for name in ("left", "right", "base") if name in missing), None
        )
        if side is None or not tokens:
            raise ReplayInconsistent(
                f"reuse deletion of {pointer!r} cannot be replayed: "
                "no missing side"
            )
        document = delete_at(document, tokens)
    elif step.get("choice") is not None:
        if step["choice"] in conflict["missing"]:
            document = delete_at(document, tokens)
        else:
            document = set_at(document, tokens, conflict[step["choice"]])
    elif step.get("value_provided"):
        document = set_at(document, tokens, step["value"])
    else:  # pragma: no cover - _event_items never emits such a step
        raise ReplayInconsistent(f"event step for {pointer!r} has no decision")
    return document


def rebuild_state(
    row, events: list[ReplayedEvent], skip_seqs: set[int]
) -> tuple[Any, list[dict]]:
    """Rebuild (document, open_conflicts) via replay from the initial state.

    Adjudication events whose seq is in ``skip_seqs`` (the previously
    compensated set, optionally plus the batch being undone now) are not
    replayed; undo events are never replayed.
    """
    document, initial_conflicts = initial_state(row)
    # Per-pointer lookup; conflicts are consumed in application order below.
    conflicts = {c["pointer"]: dict(c) for c in initial_conflicts}
    opened = {c["pointer"] for c in initial_conflicts}
    for event in events:
        if event.method == audit.METHOD_RESOLUTION_UNDO:
            continue
        if event.seq in skip_seqs:
            continue
        steps = _event_items(event)
        for step in steps:
            document = _apply_step(document, conflicts, step)
            opened.discard(step["pointer"])
    remaining = [c for c in initial_conflicts if c["pointer"] in opened]
    return document, remaining
