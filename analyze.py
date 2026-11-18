"""Offline analysis of the pilot traces (rules fixed before the full run).

Inputs : results_<model>.jsonl files (one episode per line).
Outputs: printed tables + analysis.json.
Usage  : python -I analyze.py <results_dir> [--set primary_40|extension_75|all]
"""
import collections, glob, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
WRITE = {"cancel_pending_order", "exchange_delivered_order_items", "modify_pending_order_address",
         "modify_pending_order_items", "modify_pending_order_payment", "modify_user_address",
         "return_delivered_order_items"}


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def key(a):
    return json.dumps({"name": a["name"], "kwargs": _canon(a["kwargs"])}, sort_keys=True)


def _canon(kwargs):
    """Order-insensitive form of a write's arguments (list order does not change the state)."""
    kw = dict(kwargs) if isinstance(kwargs, dict) else {}
    if isinstance(kw.get("item_ids"), list) and isinstance(kw.get("new_item_ids"), list) \
            and len(kw["item_ids"]) == len(kw["new_item_ids"]):
        pairs = sorted(zip(map(str, kw["item_ids"]), map(str, kw["new_item_ids"])))
        kw["item_ids"] = [a for a, _ in pairs]
        kw["new_item_ids"] = [b for _, b in pairs]
    elif isinstance(kw.get("item_ids"), list):
        kw["item_ids"] = sorted(map(str, kw["item_ids"]))
    return kw


def classify(r):
    """Primary failure class of a failed episode (first matching rule wins)."""
    if "harness_error" in r:
        return "E: harness error"
    gold = collections.Counter(key(a) for a in r["gold_actions"] if a["name"] in WRITE)
    gold_tools = {a["name"] for a in r["gold_actions"] if a["name"] in WRITE}
    ok_writes = collections.Counter()
    wrong_arg = wrong_tool = 0
    for t in r["trace"]:
        if t.get("kind") == "tool" and t.get("write") and not str(t.get("observation", "")).startswith("Error"):
            k = key({"name": t["name"], "kwargs": t["args"]})
            if gold[k] - ok_writes[k] > 0:
                ok_writes[k] += 1
            elif t["name"] in gold_tools:
                wrong_arg += 1
            else:
                wrong_tool += 1
    if wrong_arg or wrong_tool:
        return "W1: wrong arguments in a write" if wrong_arg else "W2: unrequested write"
    if sum((gold - ok_writes).values()) > 0:
        attempted = any(t.get("kind") == "tool" and t.get("write") for t in r["trace"])
        return "M1: write attempted, rejected by environment" if attempted else "M2: required write never attempted"
    if not r["r_outputs"]:
        return "O: required information not given"
    return "X: other state mismatch"


def gate_detail(r):
    """What a confirmation step in front of write calls would see in this episode.

    Returns (proposed, wrong_rejected_by_shop, wrong_executed). A proposal is
    'wrong' when it matches no unused gold write; 'wrong_executed' are the wrong
    proposals the shop accepted, i.e. the ones only a human confirmer could stop.
    """
    if "harness_error" in r:
        return 0, 0, 0
    gold = collections.Counter(key(a) for a in r["gold_actions"] if a["name"] in WRITE)
    used = collections.Counter()
    proposed = wrong_rej = wrong_exe = 0
    for t in r.get("trace", []):
        if t.get("kind") == "tool" and t.get("write") and "guardrail" not in t:
            proposed += 1
            k = key({"name": t["name"], "kwargs": t["args"] if isinstance(t["args"], dict) else {}})
            if gold[k] - used[k] > 0:
                used[k] += 1
            elif str(t.get("observation", "")).startswith("Error"):
                wrong_rej += 1
            else:
                wrong_exe += 1
    return proposed, wrong_rej, wrong_exe


def gate_stats(r):
    p, wr, we = gate_detail(r)
    return p, wr + we


def main():
    rdir = sys.argv[1]
    which = sys.argv[sys.argv.index("--set") + 1] if "--set" in sys.argv else "primary_40"
    frozen = json.load(open(os.path.join(HERE, "frozen_tasks.json")))
    ids = set(range(115)) if which == "all" else set(frozen[which])
    out = {}
    for p in sorted(glob.glob(os.path.join(rdir, "results_*.jsonl"))):
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        rows = [r for r in rows if r["task_id"] in ids]
        if not rows:
            continue
        model = rows[0]["model"]
        by = collections.defaultdict(dict)
        for r in rows:
            by[r["arm"]][r["task_id"]] = r
        print(f"\n=== {model}  set={which}")
        res = {}
        for arm in sorted(by):
            R = list(by[arm].values())
            n = len(R)
            k = int(sum(r["reward"] for r in R))
            lo, hi = wilson(k, n)
            st = [r.get("stats", {}) for r in R]
            fails = [r for r in R if r["reward"] < 1]
            cls = collections.Counter(classify(r) for r in fails)
            g = [gate_stats(r) for r in R]
            g_fail = [gate_stats(r) for r in fails]
            gd = [gate_detail(r) for r in R]
            env_msgs = collections.Counter()
            for r in R:
                for t in r.get("trace", []):
                    o = str(t.get("observation", ""))
                    if t.get("kind") == "tool" and (o.startswith("Error") or o.startswith("Unknown")):
                        env_msgs["not found" if "not found" in o else ("state/policy" if ("non-" in o or "cannot" in o) else "other")] += 1
            fail_gate_visible = sum(1 for (pp, w) in g_fail if w > 0)
            res[arm] = {
                "n": n, "success": k, "wilson95": [round(lo, 3), round(hi, 3)],
                "malformed_text": sum(s.get("malformed_text", 0) for s in st),
                "episodes_with_malformed": sum(1 for s in st if s.get("malformed_text", 0) > 0),
                "invalid_calls": sum(s.get("invalid_calls", 0) for s in st),
                "rejected": sum(s.get("rejected", 0) for s in st),
                "env_errors": sum(s.get("env_errors", 0) for s in st),
                "harness_errors": sum(1 for r in R if "harness_error" in r),
                "stop_reasons": dict(collections.Counter(r.get("stop_reason") for r in R)),
                "fail_classes": dict(sorted(cls.items())),
                "gate_writes_per_episode": round(sum(pp for pp, _ in g) / n, 2),
                "gate_wrong_writes": sum(w for _, w in g),
                "gate_total_writes": sum(pp for pp, _ in g),
                "gate_wrong_rejected_by_shop": sum(x[1] for x in gd),
                "gate_wrong_executed": sum(x[2] for x in gd),
                "episodes_with_wrong_executed_write": sum(1 for x in gd if x[2] > 0),
                "env_error_types": dict(env_msgs),
                "transfers": sum(1 for r in R if r.get("stop_reason") == "transfer"),
                "failed_with_wrong_write": fail_gate_visible,
                "failed_without_wrong_write": len(fails) - fail_gate_visible,
                "mean_llm_calls": round(sum(s.get("llm_calls", 0) for s in st) / n, 1),
                "mean_gen_tokens": round(sum(s.get("gen_tokens", 0) for s in st) / n),
                "mean_prompt_tokens": round(sum(s.get("prompt_tokens", 0) for s in st) / n),
                "mean_seconds": round(sum(s.get("episode_seconds", 0) for s in st) / n, 1),
            }
            print(arm, json.dumps(res[arm]))
        for a, b in (("A0", "A1"), ("A0", "A2"), ("A1", "A2")):
            if a in by and b in by:
                common = sorted(set(by[a]) & set(by[b]))
                b_ = sum(1 for t in common if by[a][t]["reward"] < 1 <= by[b][t]["reward"])
                c_ = sum(1 for t in common if by[a][t]["reward"] >= 1 > by[b][t]["reward"])
                res[f"{a}_vs_{b}"] = {"n_pairs": len(common), f"only_{b}": b_, f"only_{a}": c_,
                                      "mcnemar_exact_p": round(mcnemar_exact(b_, c_), 3)}
                print(f"{a} vs {b}:", res[f"{a}_vs_{b}"])
        # Noise floor: where the A1 check never fired, A1 ran exactly A0's protocol,
        # so any difference between the two is run-to-run variation of batched decoding.
        if "A0" in by and "A1" in by:
            quiet = [t for t in sorted(set(by["A0"]) & set(by["A1"]))
                     if by["A1"][t].get("stats", {}).get("invalid_calls", 0) == 0
                     and "harness_error" not in by["A0"][t] and "harness_error" not in by["A1"][t]]
            same_reward = sum(1 for t in quiet if by["A0"][t]["reward"] == by["A1"][t]["reward"])
            same_actions = sum(1 for t in quiet if by["A0"][t]["executed"] == by["A1"][t]["executed"])
            res["noise_floor_A0_A1"] = {"tasks_where_A1_never_fired": len(quiet),
                                        "same_reward": same_reward, "same_action_sequence": same_actions}
            print("noise floor:", res["noise_floor_A0_A1"])
        out[model] = res
    json.dump(out, open(os.path.join(rdir, f"analysis_{which}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
