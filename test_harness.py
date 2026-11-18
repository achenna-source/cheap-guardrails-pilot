"""Local test of the harness with a gold-replay mock (no LLM)."""
import sys, os, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
import jsonschema

bad = collections.Counter()
for tid in range(len(H.TASKS_TEST)):
    for arm, mode, corrupt in (("A0", "native", False), ("A1", "native", True), ("A2", "json", False)):
        r = H.run_episode(H.GoldReplayClient(tid, mode, corrupt), tid, arm, H.MockUserClient())
        assert r["stop_reason"] in ("user_stop", "transfer"), r["stop_reason"]
        if r["reward"] != 1.0:
            bad[(arm, tid)] += 1
            print("FAIL", arm, tid, r["r_actions"], r["outputs"], r["stop_reason"])
        if arm == "A1":
            assert r["stats"]["retries"] == 1 and r["stats"]["invalid_calls"] == 1, r["stats"]
# the A2 grammar schema must accept every gold action
U = H.action_union_schema()
n = 0
for t in H.TASKS_TEST:
    for a in t.actions:
        jsonschema.validate({"name": a.name, "arguments": a.kwargs}, U); n += 1
jsonschema.validate({"name": "respond", "arguments": {"content": "hi"}}, U)
try:
    jsonschema.validate({"name": "cancel_pending_order", "arguments": {"order_id": "#W1"}}, U)
    print("SCHEMA TOO LOOSE")
except jsonschema.ValidationError:
    pass
print("gold actions accepted by A2 schema:", n)
print("episodes failing:", sum(bad.values()), "of", 3 * len(H.TASKS_TEST))
print("union schema size (chars):", len(json.dumps(U)))
