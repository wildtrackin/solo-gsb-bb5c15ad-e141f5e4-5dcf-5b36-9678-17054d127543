"""End-to-end tests for resolution undo (POST /merges/{id}/resolution-undo).

An undone batch must vanish as if it had never happened: the working
document and the open conflicts are rebuilt from the immutable initial
conflict snapshot by replaying the audit chain (with the compensated
events skipped), not by reverse-editing the current JSON. Each successful
undo bumps the revision by one and appends a chained ``resolution-undo``
event naming the undone seq and the restored pointers; later undos skip
already-compensated batches. Stale revisions, resolved merges and
"nothing left to undo" return 409 and leave neither state change nor
event.
"""

import json as _json
import os
import tempfile

os.environ["DATABASE_PATH"] = os.path.join(tempfile.mkdtemp(), "test_undo.db")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import db  # noqa: E402
from app.audit import GENESIS_PREV_HASH  # noqa: E402
from app.main import app  # noqa: E402
from app.patch import canonical_json  # noqa: E402

import hashlib  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def make_merge(client, base, left, right, schema=None):
    payload = {"base": base, "left": left, "right": right}
    if schema is not None:
        payload["result_schema"] = schema
    resp = client.post("/merges", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()


def resolve(client, merge_id, revision, resolutions):
    resp = client.post(
        f"/merges/{merge_id}/resolutions",
        json={"revision": revision, "resolutions": resolutions},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def undo(client, merge_id, revision, expect=200):
    resp = client.post(
        f"/merges/{merge_id}/resolution-undo", json={"revision": revision}
    )
    assert resp.status_code == expect, resp.text
    return resp.json()


def audit(client, merge_id):
    resp = client.get(f"/merges/{merge_id}/audit")
    assert resp.status_code == 200, resp.text
    return resp.json()


def recompute_hash(merge_id, event):
    payload = {
        "merge_id": merge_id,
        "seq": event["seq"],
        "method": event["method"],
        "revision_before": event["revision_before"],
        "revision_after": event["revision_after"],
        "pointers": event["pointers"],
        "detail": event["detail"],
        "prev_hash": event["prev_hash"],
        "created_at": event["created_at"],
    }
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def three_conflict_merge(client):
    """/a, /b, /c conflicts: base 1, left 2, right 3."""
    return make_merge(
        client,
        base={"a": 1, "b": 1, "c": 1},
        left={"a": 2, "b": 2, "c": 2},
        right={"a": 3, "b": 3, "c": 3},
    )


PLACEHOLDER_KEY = "__merge_conflict__"


# ---------------------------------------------------------------------------
# Basic undo semantics
# ---------------------------------------------------------------------------


def test_undo_reopens_last_batch_and_restores_placeholder(client):
    merge = three_conflict_merge(client)
    mid = merge["merge_id"]

    resolve(client, mid, 1, [{"pointer": "/a", "choice": "left"}])
    resolve(client, mid, 2, [{"pointer": "/b", "value": 20}])
    body = undo(client, mid, 3)

    assert body["status"] == "pending"
    assert body["revision"] == 4
    # The /b batch is gone: the placeholder and the original candidates
    # come back, while the untouched /a adjudication stays in effect.
    assert body["document"]["b"] == {PLACEHOLDER_KEY: "/b"}
    assert body["document"]["a"] == 2
    assert body["document"]["c"] == {PLACEHOLDER_KEY: "/c"}
    pointers = {c["pointer"]: c for c in body["conflicts"]}
    assert set(pointers) == {"/b", "/c"}
    assert pointers["/b"]["base"] == 1
    assert pointers["/b"]["left"] == 2
    assert pointers["/b"]["right"] == 3
    assert pointers["/b"]["missing"] == []
    assert body["result"] is None
    # Response names the compensated adjudication.
    assert body["undone_seq"] == 2
    assert body["undone_method"] == "resolutions"
    assert body["restored_pointers"] == ["/b"]


def test_undo_returns_to_exactly_the_initial_state_for_a_single_batch(client):
    # Four conflicts so resolving three in one batch still leaves the merge
    # pending: the batch that produces the immutable result can never undo.
    merge = make_merge(
        client,
        base={"a": 1, "b": 1, "c": 1, "d": 1},
        left={"a": 2, "b": 2, "c": 2, "d": 2},
        right={"a": 3, "b": 3, "c": 3, "d": 3},
    )
    mid = merge["merge_id"]
    resolve(
        client,
        mid,
        1,
        [
            {"pointer": "/a", "choice": "left"},
            {"pointer": "/b", "value": None},
            {"pointer": "/c", "choice": "right"},
        ],
    )
    body = undo(client, mid, 2)
    assert body["status"] == "pending"
    assert body["revision"] == 3
    assert body["document"] == merge["document"]
    assert body["conflicts"] == merge["conflicts"]
    assert body["conflict_count"] == 4
    assert body["restored_pointers"] == ["/a", "/b", "/c"]


def test_two_undos_then_new_resolution_and_chain_still_coherent(client):
    merge = three_conflict_merge(client)
    mid = merge["merge_id"]
    resolve(client, mid, 1, [{"pointer": "/a", "choice": "left"}])
    resolve(client, mid, 2, [{"pointer": "/b", "value": 20}])

    # Undo the /b batch (compensates event 2).
    body = undo(client, mid, 3)
    assert body["revision"] == 4
    assert body["undone_seq"] == 2

    # A fresh adjudication on the reopened /b moves the merge on normally.
    body = resolve(client, mid, 4, [{"pointer": "/b", "choice": "right"}])
    assert body["document"]["b"] == 3

    # Undo again: it must skip event 2 (already compensated) and the undo
    # event at seq 3, and compensate the newest *live* verdict: event 4
    # (the new /b choice).
    body = undo(client, mid, 5)
    assert body["undone_seq"] == 4
    assert body["restored_pointers"] == ["/b"]
    assert body["document"]["b"] == {PLACEHOLDER_KEY: "/b"}
    assert body["document"]["a"] == 2
    assert {c["pointer"] for c in body["conflicts"]} == {"/b", "/c"}
    assert body["revision"] == 6

    # Undo yet again: skips the two compensated batches and both undo
    # events, reaching event 1 (the /a choice).
    body = undo(client, mid, 6)
    assert body["undone_seq"] == 1
    assert body["restored_pointers"] == ["/a"]
    assert body["document"] == merge["document"]
    assert {c["pointer"] for c in body["conflicts"]} == {"/a", "/b", "/c"}
    assert body["revision"] == 7

    trail = audit(client, mid)
    assert trail["chain_valid"] is True
    assert trail["first_invalid_seq"] is None


# ---------------------------------------------------------------------------
# Undo across adjudication methods (all replay via the initial snapshot)
# ---------------------------------------------------------------------------


def test_undo_rule_batch_rebuilds_from_snapshot(client):
    merge = three_conflict_merge(client)
    mid = merge["merge_id"]
    resolve(client, mid, 1, [{"pointer": "/a", "choice": "left"}])
    resp = client.post(
        f"/merges/{mid}/rule-resolutions",
        json={"revision": 2, "rules": [{"right": "/b"}]},
    )
    assert resp.status_code == 200
    body = undo(client, mid, 3)
    assert body["undone_method"] == "rule-resolutions"
    assert body["restored_pointers"] == ["/b"]
    assert body["document"]["a"] == 2
    assert body["document"]["b"] == {PLACEHOLDER_KEY: "/b"}
    assert body["document"]["c"] == {PLACEHOLDER_KEY: "/c"}
    assert {c["pointer"] for c in body["conflicts"]} == {"/b", "/c"}


def test_undo_reuse_batch_restores_candidates(client):
    def source_merge():
        return make_merge(
            client,
            base={"a": 1, "b": 1, "c": 1, "d": 1},
            left={"a": 2, "b": 2, "c": 2, "d": 2},
            right={"a": 3, "b": 3, "c": 3, "d": 3},
        )

    source = source_merge()
    sid = source["merge_id"]
    # Source resolves only three conflicts, then closes the fourth
    # separately so it becomes resolved.
    resolve(
        client,
        sid,
        1,
        [
            {"pointer": "/a", "choice": "left"},
            {"pointer": "/b", "value": "B"},
            {"pointer": "/c", "choice": "right"},
        ],
    )
    resolve(client, sid, 2, [{"pointer": "/d", "choice": "left"}])

    # The target has an extra conflict (/e) the source never knew, so a
    # full reuse still leaves it pending and therefore undoable.
    target = make_merge(
        client,
        base={"a": 1, "b": 1, "c": 1, "d": 1, "e": 1},
        left={"a": 2, "b": 2, "c": 2, "d": 2, "e": 2},
        right={"a": 3, "b": 3, "c": 3, "d": 3, "e": 3},
    )
    mid = target["merge_id"]
    resp = client.post(
        f"/merges/{mid}/reuse-resolutions",
        json={"revision": 1, "source_merge_id": sid},
    )
    assert resp.status_code == 200
    applied = resp.json()
    assert applied["status"] == "pending"
    assert applied["applied_count"] == 4
    body = undo(client, mid, 2)
    assert body["undone_method"] == "reuse-resolutions"
    assert body["restored_pointers"] == ["/a", "/b", "/c", "/d"]
    assert body["document"] == target["document"]
    assert body["conflicts"] == target["conflicts"]


def test_undo_draft_commit_batch(client):
    merge = three_conflict_merge(client)
    mid = merge["merge_id"]
    did = client.post(f"/merges/{mid}/drafts").json()["draft_id"]
    client.post(
        f"/merges/{mid}/drafts/{did}/resolutions",
        json={
            "resolutions": [
                {"pointer": "/a", "choice": "left"},
                {"pointer": "/b", "value": 9},
            ]
        },
    )
    resp = client.post(f"/merges/{mid}/drafts/{did}/commit")
    assert resp.status_code == 200
    body = undo(client, mid, 2)
    assert body["undone_method"] == "draft-commit"
    assert body["restored_pointers"] == ["/a", "/b"]
    assert body["document"] == merge["document"]
    assert {c["pointer"] for c in body["conflicts"]} == {"/a", "/b", "/c"}
    # The draft stays committed/invalidated; only the merge is rebuilt.
    draft = client.get(f"/merges/{mid}/drafts/{did}").json()
    assert draft["status"] == "committed"


def test_undo_restores_deleted_key_and_missing_side(client):
    # left deletes /x, right modifies it: choosing left honours the delete.
    merge = make_merge(
        client,
        base={"x": 1, "y": 1},
        left={"y": 2},
        right={"x": 3, "y": 3},
    )
    mid = merge["merge_id"]
    resolve(client, mid, 1, [{"pointer": "/x", "choice": "left"}])
    body = undo(client, mid, 2)
    # The deleted key is back, carrying the conflict placeholder.
    assert body["document"]["x"] == {PLACEHOLDER_KEY: "/x"}
    conflict = next(c for c in body["conflicts"] if c["pointer"] == "/x")
    assert conflict["missing"] == ["left"]
    assert {c["pointer"] for c in body["conflicts"]} == {"/x", "/y"}


# ---------------------------------------------------------------------------
# Error paths: 409 / 404, atomicity, no events on failure
# ---------------------------------------------------------------------------


def test_undo_with_no_adjudication_is_409_and_writes_nothing(client):
    merge = three_conflict_merge(client)
    mid = merge["merge_id"]
    body = undo(client, mid, 1, expect=409)
    assert "undo" in body["detail"]
    assert audit(client, mid)["events"] == []
    view = client.get(f"/merges/{mid}").json()
    assert view["revision"] == 1
    assert view["conflicts"] == merge["conflicts"]


def test_undo_after_everything_already_undone_is_409(client):
    merge = three_conflict_merge(client)
    mid = merge["merge_id"]
    resolve(client, mid, 1, [{"pointer": "/a", "choice": "left"}])
    undo(client, mid, 2)
    # The only live adjudication is already compensated: nothing to undo.
    undo(client, mid, 3, expect=409)
    view = client.get(f"/merges/{mid}").json()
    assert view["revision"] == 3  # the failed undo did not bump revision
    # Only the original event and its undo exist; the failure added none.
    trail = audit(client, mid)
    assert [e["method"] for e in trail["events"]] == [
        "resolutions",
        "resolution-undo",
    ]


def test_stale_revision_returns_409_with_current_revision(client):
    merge = three_conflict_merge(client)
    mid = merge["merge_id"]
    resolve(client, mid, 1, [{"pointer": "/a", "choice": "left"}])
    resp = client.post(
        f"/merges/{mid}/resolution-undo", json={"revision": 1}
    )
    assert resp.status_code == 409
    assert resp.json()["current_revision"] == 2
    assert audit(client, mid)["event_count"] == 1


def test_undo_on_resolved_merge_is_409_and_result_stays_immutable(client):
    merge = three_conflict_merge(client)
    mid = merge["merge_id"]
    resolve(
        client,
        mid,
        1,
        [
            {"pointer": "/a", "choice": "left"},
            {"pointer": "/b", "choice": "left"},
            {"pointer": "/c", "choice": "left"},
        ],
    )
    view = client.get(f"/merges/{mid}").json()
    assert view["status"] == "resolved"
    undo(client, mid, 2, expect=409)
    after = client.get(f"/merges/{mid}").json()
    assert after["status"] == "resolved"
    assert after["result"] == view["result"]
    assert after["revision"] == 2
    # No undo event was appended to the resolved merge.
    assert all(
        e["method"] != "resolution-undo" for e in audit(client, mid)["events"]
    )


def test_undo_unknown_merge_returns_404(client):
    assert (
        client.post(
            "/merges/no-such-merge/resolution-undo", json={"revision": 1}
        ).status_code
        == 404
    )


# ---------------------------------------------------------------------------
# Undo then the normal schema gate still applies to the re-closing batch
# ---------------------------------------------------------------------------


def test_schema_gate_runs_again_after_undo_reopens_last_conflict(client):
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer", "minimum": 10}},
    }
    merge = make_merge(
        client, base={"a": 1}, left={"a": 2}, right={"a": 3}, schema=schema
    )
    mid = merge["merge_id"]
    resolve(client, mid, 1, [{"pointer": "/a", "value": 11}])
    assert client.get(f"/merges/{mid}").json()["status"] == "resolved"

    # Resolved merges cannot be undone at all (result immutable).
    undo(client, mid, 2, expect=409)

    # A pending case instead: resolve partially, undo, then a bad final
    # choice must still be stopped by the gate.
    merge2 = make_merge(
        client,
        base={"a": 1, "b": 1},
        left={"a": 2, "b": 2},
        right={"a": 3, "b": 3},
        schema={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer", "minimum": 10},
            },
        },
    )
    m2 = merge2["merge_id"]
    resolve(client, m2, 1, [{"pointer": "/a", "choice": "left"}])
    undo(client, m2, 2)  # reopen /a: revision 3, both conflicts open again
    resolve(client, m2, 3, [{"pointer": "/a", "choice": "left"}])  # rev 4
    # Final batch /b fails the gate: 422, state untouched.
    resp = client.post(
        f"/merges/{m2}/resolutions",
        json={"revision": 4, "resolutions": [{"pointer": "/b", "value": 1}]},
    )
    assert resp.status_code == 422
    view = client.get(f"/merges/{m2}").json()
    assert view["status"] == "pending"
    assert view["revision"] == 4
    # The valid value passes and closes the merge.
    view = resolve(client, m2, 4, [{"pointer": "/b", "value": 10}])
    assert view["status"] == "resolved"


# ---------------------------------------------------------------------------
# Audit event for the undo: hash chain, seq skipping, payload
# ---------------------------------------------------------------------------


def test_undo_appends_chained_event_and_chain_verifies(client):
    merge = three_conflict_merge(client)
    mid = merge["merge_id"]
    resolve(client, mid, 1, [{"pointer": "/a", "choice": "left"}])
    resolve(client, mid, 2, [{"pointer": "/b", "value": 20}])
    undo(client, mid, 3)

    trail = audit(client, mid)
    assert trail["chain_valid"] is True
    assert trail["first_invalid_seq"] is None
    events = trail["events"]
    assert [e["seq"] for e in events] == [1, 2, 3]
    assert [e["method"] for e in events] == [
        "resolutions",
        "resolutions",
        "resolution-undo",
    ]
    undo_event = events[2]
    assert undo_event["revision_before"] == 3
    assert undo_event["revision_after"] == 4
    assert undo_event["pointers"] == ["/b"]
    assert undo_event["detail"] == {
        "undone_seq": 2,
        "undone_method": "resolutions",
        "restored_pointers": ["/b"],
    }
    assert undo_event["prev_hash"] == events[1]["hash"]
    assert undo_event["hash"] == recompute_hash(mid, undo_event)
    assert events[0]["prev_hash"] == GENESIS_PREV_HASH


def test_failed_undo_leaves_no_event(client):
    merge = three_conflict_merge(client)
    mid = merge["merge_id"]
    # Stale revision before any adjudication, and nothing-to-undo at the
    # current revision: both 409, neither records an event.
    client.post(f"/merges/{mid}/resolution-undo", json={"revision": 5})
    client.post(f"/merges/{mid}/resolution-undo", json={"revision": 1})
    assert audit(client, mid)["events"] == []


def test_undo_replay_detects_tampered_event_content(client):
    merge = three_conflict_merge(client)
    mid = merge["merge_id"]
    resolve(client, mid, 1, [{"pointer": "/a", "choice": "left"}])
    resolve(client, mid, 2, [{"pointer": "/b", "value": 20}])
    # Tamper event 1's detail out-of-band: replay can no longer reproduce
    # the stored document, so the undo is refused with no state change.
    event = db.query_all(
        "SELECT detail FROM merge_audit_events WHERE merge_id = ? AND seq = 1",
        (mid,),
    )[0]
    tampered = _json.loads(event["detail"])
    tampered["items"][0]["choice"] = "right"
    db.execute(
        "UPDATE merge_audit_events SET detail = ? WHERE merge_id = ? AND seq = 1",
        (_json.dumps(tampered), mid),
    )
    resp = client.post(
        f"/merges/{mid}/resolution-undo", json={"revision": 3}
    )
    assert resp.status_code == 409
    view = client.get(f"/merges/{mid}").json()
    assert view["revision"] == 3
    assert view["document"]["b"] == 20
    # No event was appended by the refused undo.
    assert audit(client, mid)["event_count"] == 2


def test_audit_trail_marks_tampered_undo_chain_invalid(client):
    merge = three_conflict_merge(client)
    mid = merge["merge_id"]
    resolve(client, mid, 1, [{"pointer": "/a", "choice": "left"}])
    undo(client, mid, 2)
    # Tamper the undo event's stored pointers; verification flags it.
    db.execute(
        "UPDATE merge_audit_events SET pointers = ? WHERE merge_id = ? AND seq = 2",
        (_json.dumps(["/tampered"]), mid),
    )
    trail = audit(client, mid)
    assert trail["chain_valid"] is False
    assert trail["first_invalid_seq"] == 2


# ---------------------------------------------------------------------------
# Rebuilding really starts from the immutable initial snapshot, not the
# current JSON: reopen a conflict whose candidate document content the
# client could not reconstruct by reverse-editing (nested + deletion).
# ---------------------------------------------------------------------------


def test_undo_of_batch_then_undo_of_earlier_batch_full_rebuild(client):
    merge = make_merge(
        client,
        base={"db": {"host": "a", "port": 1}, "keep": 1, "extra": 1},
        left={"db": {"host": "b", "port": 1}, "keep": 2, "extra": 2},
        right={"db": {"host": "c", "port": 2}, "keep": 3, "extra": 3},
    )
    mid = merge["merge_id"]
    # Conflicts at /db/host, /keep and /extra (/db/port auto-merged to 2).
    assert {c["pointer"] for c in merge["conflicts"]} == {
        "/db/host",
        "/keep",
        "/extra",
    }
    resolve(client, mid, 1, [{"pointer": "/db/host", "choice": "left"}])
    resolve(client, mid, 2, [{"pointer": "/keep", "value": 99}])

    first = undo(client, mid, 3)
    assert {c["pointer"] for c in first["conflicts"]} == {"/keep", "/extra"}
    assert first["document"]["db"] == {"host": "b", "port": 2}
    second = undo(client, mid, 4)
    # Full rebuild: /db/host placeholder returns, /db/port keeps the
    # automatic merge result, /keep placeholder returns.
    assert second["document"] == merge["document"]
    assert second["conflicts"] == merge["conflicts"]
    assert second["document"]["db"]["port"] == 2
    assert second["revision"] == 5


def test_preexisting_db_with_old_check_constraint_gets_migrated(client):
    # Simulate an old-volume database: the audit table with the original
    # four-method CHECK. init_schema's rebuild must let undo events land.
    path = os.path.join(tempfile.mkdtemp(), "legacy.db")
    from app import db as db_module

    old_path = db_module.DATABASE_PATH
    db_module._conn = None
    db_module.DATABASE_PATH = path
    try:
        conn = db_module.connect()
        conn.executescript(
            """
            CREATE TABLE merges (
                id TEXT PRIMARY KEY, idempotency_key TEXT UNIQUE,
                status TEXT NOT NULL CHECK (status IN ('pending', 'resolved')),
                revision INTEGER NOT NULL, base_doc TEXT NOT NULL,
                left_doc TEXT NOT NULL, right_doc TEXT NOT NULL,
                document TEXT NOT NULL, conflicts TEXT NOT NULL,
                source_conflicts TEXT, result TEXT, result_schema TEXT,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE merge_drafts (
                id TEXT PRIMARY KEY, merge_id TEXT NOT NULL,
                status TEXT NOT NULL, base_revision INTEGER NOT NULL,
                document TEXT NOT NULL, conflicts TEXT NOT NULL,
                resolutions TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE merge_audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, merge_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                method TEXT NOT NULL CHECK (method IN (
                    'resolutions', 'rule-resolutions', 'reuse-resolutions',
                    'draft-commit')),
                revision_before INTEGER NOT NULL, revision_after INTEGER NOT NULL,
                pointers TEXT NOT NULL, detail TEXT NOT NULL,
                prev_hash TEXT NOT NULL, hash TEXT NOT NULL,
                created_at TEXT NOT NULL, UNIQUE (merge_id, seq)
            );
            """
        )
        conn.commit()
        db_module.init_schema()

        with TestClient(app) as c:
            merge = make_merge(
                c,
                base={"a": 1, "b": 1},
                left={"a": 2, "b": 2},
                right={"a": 3, "b": 3},
            )
            mid = merge["merge_id"]
            resolve(c, mid, 1, [{"pointer": "/a", "choice": "left"}])
            body = undo(c, mid, 2)
            assert body["restored_pointers"] == ["/a"]
            trail = audit(c, mid)
            assert trail["chain_valid"] is True
            assert trail["events"][1]["method"] == "resolution-undo"
    finally:
        # Restore the module's connection for any later in-process work.
        db_module._conn = None
        db_module.DATABASE_PATH = old_path
        db_module.init_schema()
