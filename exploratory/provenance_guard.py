"""Offline counterfactual (exploratory): a provenance guard that blocks any tool call
whose identifier arguments never appeared earlier in the customer's messages or in
tool outputs. Counts what it would have blocked in the recorded traces; it cannot
model how the agent would have continued after a block.

Usage: python -I provenance_guard.py <repo_dir> <out_json>
"""
import collections, json, os, sys

sys.argv, REPO, OUT = sys.argv[:1] + [sys.argv[1], os.devnull], sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extra_analysis as X   # noqa: E402

ID_KEYS = ("order_id", "user_id", "product_id", "payment_method_id", "email", "zip")
LIST_KEYS = ("item_ids", "new_item_ids")


def ids_of(args):
    vals = [str(args[k]) for k in ID_KEYS if k in args and args[k] not in (None, "")]
    for k in LIST_KEYS:
        if isinstance(args.get(k), list):
            vals += [str(v) for v in args[k]]
    return vals


def grounded(v, ctx):
    core = v.lstrip("#")
    return bool(core) and core in ctx


def main():
    out = {}
    for model in ("3b", "7b"):
        for subset in ("primary_40", "all"):
            for arm in ("A0", "A1", "A2"):
                c = collections.Counter()
                for r in X.load(model):
                    if r["arm"] != arm or (subset == "primary_40" and r["task_id"] not in X.PRIMARY):
                        continue
                    ev, _ = X.replay(r)
                    gold = collections.Counter(X.key(a) for a in r["gold_actions"] if a["name"] in X.WRITE)
                    used = collections.Counter()
                    for e in ev:
                        vals = ids_of(e["args"])
                        if not vals:
                            continue
                        ctx = e["cust"] + "\n" + e["tools"]
                        ok = all(grounded(v, ctx) for v in vals)
                        err = e["obs"].startswith("Error")
                        c["calls_with_ids"] += 1
                        c["blocked"] += not ok
                        c["blocked_not_found"] += (not ok) and err and "not found" in e["obs"]
                        c["not_found_total"] += err and "not found" in e["obs"]
                        c["blocked_would_succeed"] += (not ok) and not err
                        c["succeeded_total"] += not err
                        if e["name"] in X.WRITE:
                            k = X.key({"name": e["name"], "kwargs": e["args"]})
                            if gold[k] - used[k] > 0:
                                used[k] += 1
                                c["correct_writes"] += 1
                                c["correct_writes_blocked"] += not ok
                            elif not err:
                                c["wrong_executed_writes"] += 1
                                c["wrong_executed_blocked"] += not ok
                out[f"{model}|{subset}|{arm}"] = dict(c)
                print(model, subset, arm, dict(c))
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
