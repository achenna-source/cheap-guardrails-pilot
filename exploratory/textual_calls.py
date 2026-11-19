"""Exploratory check, added after review: tool calls written as text inside agent messages.

The frozen metric `malformed_text` only flagged messages that START with a JSON call
or contain <tool_call>; a call wrapped in a ```json fence or preceded by prose was
missed. Such a message is sent to the customer and nothing is executed. This script
counts agent messages that contain a call to a known tool written as JSON text, and
turns where the model emitted more than one native tool call (only the first is executed).

Usage: python -I textual_calls.py <repo_dir> <out_json>
"""
import collections, json, os, re, sys

REPO, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.abspath(REPO))
from analyze import WRITE   # noqa: E402,F401

TOOLS = sorted(f[:-3] for f in os.listdir(os.path.join(REPO, "tbr", "tools"))
               if f.endswith(".py") and not f.startswith("__"))
CALL = re.compile(r'"name"\s*:\s*"(' + "|".join(TOOLS) + r')"')
FROZEN = json.load(open(os.path.join(REPO, "frozen_tasks.json")))
PRIMARY = set(FROZEN["primary_40"])

out = {}
for model in ("3b", "7b"):
    rows = [json.loads(l) for l in open(os.path.join(REPO, "results", f"results_qwen2.5_{model}.jsonl"), encoding="utf-8")]
    for subset in ("primary_40", "all"):
        for arm in ("A0", "A1", "A2"):
            c = collections.Counter()
            for r in rows:
                if "harness_error" in r or r["arm"] != arm or (subset == "primary_40" and r["task_id"] not in PRIMARY):
                    continue
                hit_ep = False
                for t in r["trace"]:
                    if t["kind"] == "text":
                        c["messages"] += 1
                        txt = t.get("content") or ""
                        n = len(CALL.findall(txt)) + txt.count("<tool_call>")
                        if n:
                            c["messages_with_text_calls"] += 1
                            c["text_calls"] += n
                            hit_ep = True
                    raw = t.get("raw_tool_calls")
                    if isinstance(raw, list) and len(raw) > 1:
                        c["turns_with_extra_native_calls"] += 1
                        c["dropped_native_calls"] += len(raw) - 1
                c["episodes_with_text_calls"] += hit_ep
                c["episodes"] += 1
            out[f"{model}|{subset}|{arm}"] = dict(c)
            print(model, subset, arm, dict(c))
json.dump(out, open(OUT, "w"), indent=1)
