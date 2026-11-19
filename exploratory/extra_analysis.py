"""Exploratory analyses of the article (not part of the frozen protocol).

Replays every executed tool call of every episode on a fresh copy of the shop
database (the shop is deterministic, so this recovers the full observations that
the traces store truncated), then reports:
  1. provenance of the identifiers behind "not found" errors;
  2. identifiers stated by the simulated customer that its instruction does not contain;
  3. a per-episode funnel (customer identified, one correct write, database correct, success);
  4. repeated failing calls, turn composition, overlap of solved tasks across arms,
     success by number of required writes.

Usage: python -I extra_analysis.py <repo_dir> <out_json>
"""
import collections, json, os, re, sys

REPO = os.path.abspath(sys.argv[1])
sys.path.insert(0, REPO)
import harness as H                      # noqa: E402
from analyze import WRITE, gate_detail, key, wilson   # noqa: E402

OUT = sys.argv[2]
FROZEN = json.load(open(os.path.join(REPO, "frozen_tasks.json")))
PRIMARY = set(FROZEN["primary_40"])

DATA = H.fresh_data()
ORDERS, USERS, PRODUCTS = DATA["orders"], DATA["users"], DATA["products"]
ITEM_IDS = {i for p in PRODUCTS.values() for i in p["variants"]}
PAY_IDS = {pm for u in USERS.values() for pm in u["payment_methods"]}
EMAILS = {u["email"] for u in USERS.values()}
REAL = set(ORDERS) | set(USERS) | set(PRODUCTS) | ITEM_IDS | PAY_IDS | EMAILS

RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[A-Za-z]+)+")
RE_ZIP = re.compile(r"(?<!\d)\d{5}(?!\d)")
RE_ORDER = re.compile(r"#?W\d{7}")
# placeholder-looking values: runs of one digit, counting sequences, generic names
RE_PLACEHOLDER = re.compile(r"(\d)\1{3,}|1234|2345|9876|john|jane|doe|your|xxx|customer@|something@|N_new|LWN|DLN", re.I)


def load(model):
    rows = [json.loads(l) for l in open(os.path.join(REPO, "results", f"results_qwen2.5_{model}.jsonl"), encoding="utf-8")]
    return [r for r in rows if "harness_error" not in r]


def replay(r):
    """Executed tool calls with full observations and the context seen before each call."""
    env = H.RetailEnv(r["task_id"])
    users = [d["content"] for d in r["dialogue"] if d["role"] == "user"]
    cust, tools, agent_txt, ui = [users[0]], [], [], 0
    events, mismatch = [], 0
    for t in r["trace"]:
        if t["kind"] == "tool":
            if "guardrail" in t:
                continue
            args = t["args"] if isinstance(t["args"], dict) else {}
            full = env.call(str(t["name"]), args)
            if full[:1500] != str(t.get("observation", ""))[:1500]:
                mismatch += 1
            events.append({"name": str(t["name"]), "args": args, "obs": full,
                           "cust": "\n".join(cust), "tools": "\n".join(tools), "agent": "\n".join(agent_txt)})
            tools.append(full)
        else:
            agent_txt.append(t.get("content") or "")
            ui += 1
            if ui < len(users):
                cust.append(users[ui])
    return events, mismatch


def offending(e):
    """(kind, value) behind a 'not found' error, or None."""
    n, a, o = e["name"], e["args"], e["obs"]
    if not o.startswith("Error") or "not found" not in o:
        return None
    if "order not found" in o:
        return "order", str(a.get("order_id"))
    if "user not found" in o:
        if n == "find_user_id_by_email":
            return "email", str(a.get("email"))
        if n == "find_user_id_by_name_zip":
            return "namezip", (str(a.get("first_name", "")), str(a.get("last_name", "")), str(a.get("zip", "")))
        return "user", str(a.get("user_id"))
    if "product not found" in o:
        return "product", str(a.get("product_id"))
    if "payment method not found" in o:
        return "payment", str(a.get("payment_method_id"))
    m = re.search(r"new item (\S+) not found", o)
    if m:
        return "item", m.group(1)
    if "some item not found" in o:
        od = ORDERS.get(str(a.get("order_id")), {})
        have = {it["item_id"] for it in od.get("items", [])}
        bad = [str(i) for i in (a.get("item_ids") or []) if str(i) not in have]
        return "item", bad[0] if bad else ""
    m = re.search(r"Error: (\S+) not found", o)
    if m:
        return "item", m.group(1)
    return "other", ""


def provenance(kind, val, cust, tools, instr, agent=""):
    """format slip > real ID in the wrong place > from customer (faithful / not in instruction /
    echo of an identifier the agent had written first) > from tool output > unsupported.
    'unsupported' = absent from every earlier customer message and tool output (the agent's own
    earlier messages are not counted as a source)."""
    if kind == "namezip":
        first, last, z = val
        if z and z in cust and first.lower() in cust.lower():
            if z in instr:
                return "customer_faithful"
            return "customer_echo" if z in agent else "customer_unfaithful"
        if z and z in tools:
            return "tool_output"
        return "unsupported"
    s = val
    if kind == "order" and s and not s.startswith("#") and "#" + s in ORDERS:
        return "format"
    if s in REAL:
        return "real_wrong_place"
    if s and s in cust:
        if s in instr:
            return "customer_faithful"
        return "customer_echo" if s in agent else "customer_unfaithful"
    if s and s in tools:
        return "tool_output"
    return "unsupported"


def customer_slips(r, instr):
    """Identifiers stated by the simulated customer that are neither in its instruction nor said earlier by the agent."""
    slips, agent_so_far = [], ""
    for d in r["dialogue"]:
        if d["role"] == "agent":
            agent_so_far += "\n" + d["content"]
            continue
        txt = d["content"]
        for tok in RE_EMAIL.findall(txt) + RE_ZIP.findall(txt) + RE_ORDER.findall(txt):
            core = tok.lstrip("#")
            if core not in instr and core not in agent_so_far:
                slips.append(tok)
    return slips


def target_user(task):
    for a in task.actions:
        if a.name.startswith("find_user_id"):
            return H.TOOLS_MAP[a.name].invoke(data=H.fresh_data(), **a.kwargs)
    for a in task.actions:
        if "user_id" in a.kwargs:
            return a.kwargs["user_id"]
    for a in task.actions:
        if a.kwargs.get("order_id") in ORDERS:
            return ORDERS[a.kwargs["order_id"]]["user_id"]
    return None


def passive_tasks():
    """Tasks that an agent making no successful write passes: gold replay leaves the
    database unchanged (no write, or gold writes the shop itself rejects) and no output is required."""
    base = H.data_hash(H.fresh_data())
    return sorted(t for t in range(len(H.TASKS_TEST))
                  if H.gold_hash(t) == base and not H.TASKS_TEST[t].outputs)


SLIP_EXAMPLES = []
PASSIVE = []


def main():
    PASSIVE.extend(passive_tasks())
    out = {"passive_tasks": PASSIVE}
    for model in ("3b", "7b"):
        rows = load(model)
        res = {}
        for subset in ("primary_40", "all"):
            R = [r for r in rows if subset == "all" or r["task_id"] in PRIMARY]
            per_arm = {}
            for arm in ("A0", "A1", "A2"):
                A = [r for r in R if r["arm"] == arm]
                prov = collections.Counter()
                prov_by_kind = collections.Counter()
                nf = 0
                mism = 0
                slips_ep = 0
                slip_tokens = 0
                funnel = collections.Counter()
                rep_fail = calls = writes = msgs = 0
                cap_eps = cap_eps_loop = 0
                for r in A:
                    task = H.TASKS_TEST[r["task_id"]]
                    instr = task.instruction
                    ev, mm = replay(r)
                    mism += mm
                    seen_fail = set()
                    loops_here = 0
                    for e in ev:
                        calls += 1
                        writes += e["name"] in WRITE
                        sig = json.dumps([e["name"], e["args"]], sort_keys=True, default=str)
                        if e["obs"].startswith("Error"):
                            if sig in seen_fail:
                                rep_fail += 1
                                loops_here += 1
                            seen_fail.add(sig)
                        off = offending(e)
                        if off is None:
                            continue
                        nf += 1
                        p = provenance(off[0], off[1], e["cust"], e["tools"], instr, e["agent"])
                        prov[p] += 1
                        prov_by_kind[f"{off[0]}:{p}"] += 1
                        if p in ("unsupported", "customer_echo"):
                            v = "|".join(off[1]) if off[0] == "namezip" else off[1]
                            prov["agent_introduced"] += 1
                            prov["agent_introduced_placeholder"] += bool(RE_PLACEHOLDER.search(v))
                            if p == "unsupported":
                                prov["unsupported_placeholder"] += bool(RE_PLACEHOLDER.search(v))
                    msgs += sum(1 for t in r["trace"] if t["kind"] == "text")
                    if r["stop_reason"] == "max_steps":
                        cap_eps += 1
                        cap_eps_loop += loops_here >= 3
                    sl = customer_slips(r, instr)
                    slips_ep += bool(sl)
                    slip_tokens += len(sl)
                    if sl and len(SLIP_EXAMPLES) < 12 and arm == "A0":
                        SLIP_EXAMPLES.append((model, arm, r["task_id"], sl[:3], instr[:160]))
                    if r["task_id"] not in PASSIVE:
                        funnel["state_changing_tasks"] += 1
                        funnel["success_state_changing"] += r["reward"] >= 1
                    # funnel (tasks with at least one required write)
                    gold_w = [a for a in r["gold_actions"] if a["name"] in WRITE]
                    if gold_w and r["task_id"] not in PASSIVE:
                        funnel["tasks_with_writes"] += 1
                        tu = target_user(task)
                        ident = any(e["name"].startswith("find_user_id") and e["obs"] == tu for e in ev) or \
                            any(e["name"] == "get_user_details" and not e["obs"].startswith("Error")
                                and e["args"].get("user_id") == tu for e in ev) or \
                            any(e["name"] == "get_order_details" and not e["obs"].startswith("Error")
                                and ORDERS.get(str(e["args"].get("order_id")), {}).get("user_id") == tu for e in ev)
                        p_, wr_, we_ = gate_detail(r)
                        correct = p_ - wr_ - we_
                        funnel["identified"] += ident
                        funnel["identified_and_one_correct_write"] += ident and correct > 0
                        funnel["database_correct"] += bool(r["r_actions"])
                        funnel["success"] += r["reward"] >= 1
                n = len(A)
                per_arm[arm] = {
                    "episodes": n, "replay_mismatches": mism,
                    "not_found_errors": nf, "provenance": dict(prov),
                    "provenance_by_kind": dict(sorted(prov_by_kind.items())),
                    "episodes_with_customer_slip": slips_ep, "customer_slip_tokens": slip_tokens,
                    "funnel": dict(funnel),
                    "tool_calls_per_episode": round(calls / n, 1), "write_calls_per_episode": round(writes / n, 2),
                    "messages_per_episode": round(msgs / n, 1),
                    "repeated_failing_calls": rep_fail, "share_repeated_failing": round(rep_fail / max(calls, 1), 3),
                    "cap_episodes": cap_eps, "cap_episodes_with_3plus_repeats": cap_eps_loop,
                }
            # overlap of solved tasks and success by number of required writes
            solved = {arm: {r["task_id"] for r in R if r["arm"] == arm and r["reward"] >= 1} for arm in ("A0", "A1", "A2")}
            ids = {r["task_id"] for r in R}
            common = set.intersection(*[{r["task_id"] for r in R if r["arm"] == a} for a in ("A0", "A1", "A2")])
            by_w = collections.defaultdict(lambda: [0, 0])
            for r in R:
                nw = sum(1 for a in r["gold_actions"] if a["name"] in WRITE)
                b = "0" if nw == 0 else ("1" if nw == 1 else ("2" if nw == 2 else "3+"))
                by_w[(b, r["arm"])][0] += r["reward"] >= 1
                by_w[(b, r["arm"])][1] += 1
            per_arm["overlap"] = {
                "tasks": len(ids), "tasks_run_by_all_arms": len(common),
                "solved_by_any": len(set().union(*[solved[a] & common for a in solved])),
                "solved_by_all": len(set.intersection(*[solved[a] & common for a in solved])),
                "solved_sets": {a: sorted(solved[a]) for a in solved},
            }
            per_arm["success_by_required_writes"] = {f"{b}|{a}": v for (b, a), v in sorted(by_w.items())}
            res[subset] = per_arm
        out[f"qwen2.5:{model}"] = res
    # task-level counts of required writes on the primary set
    nw = collections.Counter()
    for t in PRIMARY:
        nw[sum(1 for a in H.TASKS_TEST[t].actions if a.name in WRITE)] += 1
    out["primary_required_writes_hist"] = dict(sorted(nw.items()))
    json.dump(out, open(OUT, "w"), indent=1)
    for m in ("qwen2.5:3b", "qwen2.5:7b"):
        for subset in ("primary_40", "all"):
            print(f"\n=== {m} {subset}")
            for arm in ("A0", "A1", "A2"):
                d = out[m][subset][arm]
                print(arm, json.dumps({k: v for k, v in d.items() if k != "provenance_by_kind"}))
            print("overlap", json.dumps({k: v for k, v in out[m][subset]["overlap"].items() if k != "solved_sets"}))
            print("by writes", json.dumps(out[m][subset]["success_by_required_writes"]))
    print("primary required writes", out["primary_required_writes_hist"])
    print("passive tasks", PASSIVE)
    for x in SLIP_EXAMPLES:
        print("SLIP", x)
    out["slip_examples"] = SLIP_EXAMPLES
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
