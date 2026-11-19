"""Checks added after the internal review (exploratory, primary set unless stated).

Usage: python -I review_checks.py <repo_dir>
"""
import collections, json, os, sys

REPO = os.path.abspath(sys.argv[1])
sys.path.insert(0, REPO)
import harness as H   # noqa: E402

FROZEN = json.load(open(os.path.join(REPO, "frozen_tasks.json")))
PRIMARY = set(FROZEN["primary_40"])


def rows(model):
    return [json.loads(l) for l in open(os.path.join(REPO, "results", f"results_qwen2.5_{model}.jsonl"), encoding="utf-8")]


for model in ("3b", "7b"):
    R = [r for r in rows(model) if "harness_error" not in r]
    for arm in ("A0", "A1", "A2"):
        A = [r for r in R if r["arm"] == arm and r["task_id"] in PRIMARY]
        n = len(A)
        llm = sum(r["stats"]["llm_seconds"] for r in A) / n
        user = sum(r["stats"].get("user_seconds", 0) for r in A) / n
        ep = sum(r["stats"]["episode_seconds"] for r in A) / n
        calls = sum(r["stats"]["llm_calls"] for r in A)
        gen = sum(r["stats"]["gen_tokens"] for r in A)
        tool_calls = invalid = 0
        for r in A:
            for t in r["trace"]:
                if t["kind"] == "tool" and "guardrail" not in t:
                    tool_calls += 1
                    args = t["args"] if isinstance(t["args"], dict) else {}
                    if H.validate_call(t["name"], args) is not None:
                        invalid += 1
        print(f"{model} {arm} n={n} agent_llm_s={llm:.0f} user_s={user:.0f} episode_s={ep:.0f} "
              f"model_turns={calls} gen_tokens_per_turn={gen / calls:.0f} executed_calls={tool_calls} schema_invalid_executed={invalid}")
    # noise floor with tool-call sequences only
    by = collections.defaultdict(dict)
    for r in R:
        by[r["arm"]][r["task_id"]] = r
    for subset in ("primary_40", "extension_75", "all"):
        ids = PRIMARY if subset == "primary_40" else (set(FROZEN["extension_75"]) if subset == "extension_75" else set(range(115)))
        quiet = [t for t in sorted(set(by["A0"]) & set(by["A1"]) & ids) if by["A1"][t]["stats"]["invalid_calls"] == 0]
        same_r = sum(by["A0"][t]["reward"] == by["A1"][t]["reward"] for t in quiet)
        seq = lambda r: [(a["name"], json.dumps(a["kwargs"], sort_keys=True)) for a in r["executed"] if a["name"] != "respond"]
        same_tools = sum(seq(by["A0"][t]) == seq(by["A1"][t]) for t in quiet)
        print(f"{model} noise {subset}: quiet={len(quiet)} changed={len(quiet) - same_r} same_tool_sequence={same_tools}")
