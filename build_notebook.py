"""Build the self-contained Kaggle notebooks (smoke and full) for the pilot.

The notebook installs Ollama, starts one server per GPU, pulls the models,
clones tau-bench at a pinned commit, vendors the retail subset, writes the
harness and runs the frozen task list. Outputs: /kaggle/working/*.jsonl + logs.
Usage: python -I build_notebook.py   (writes kernels/smoke and kernels/full)
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
TAU_COMMIT = "59a200c6d575d595120f1cb70fea53cef0632f6b"
HARNESS = open(os.path.join(HERE, "harness.py"), encoding="utf-8").read()
VENDOR = open(os.path.join(HERE, "vendor_tau.py"), encoding="utf-8").read()
FROZEN = json.load(open(os.path.join(HERE, "frozen_tasks.json")))


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


CELL_SETUP = r'''# 1. Ollama + tau-bench (pinned) + vendored retail subset + harness
import os, subprocess, sys, time, json, shutil
W = "/kaggle/working"; os.chdir(W)
def sh(cmd, check=True):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
    if check and r.returncode != 0:
        raise RuntimeError(cmd)
    return r
sh("nvidia-smi -L")
sh("apt-get -qq update && apt-get -qq install -y zstd pciutils lshw > /dev/null", check=False)
if shutil.which("ollama") is None:
    sh("curl -fsSL https://ollama.com/install.sh | sh", check=False)
if shutil.which("ollama") is None:
    raise RuntimeError("ollama install failed")
sh("ollama --version", check=False)
sh("pip -q install jsonschema==4.23.0", check=False)
if not os.path.isdir("tau-bench"):
    sh("git clone -q https://github.com/sierra-research/tau-bench.git")
sh("cd tau-bench && git checkout -q TAU_COMMIT && git log -1 --format=%H")
open("vendor_tau.py", "w").write(VENDOR_SRC)
open("harness.py", "w").write(HARNESS_SRC)
sh(f"{sys.executable} -I vendor_tau.py tau-bench .")
'''.replace("TAU_COMMIT", TAU_COMMIT)

CELL_SERVERS = r'''# 2. Two Ollama servers: GPU 0 serves the agent model, GPU 1 the user simulator.
# NUM_PARALLEL=3 on each, so three episodes run at the same time.
gpus = [l for l in subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True).stdout.splitlines() if l.startswith("GPU")]
if len(gpus) < 2:
    raise RuntimeError("needs GPU T4 x2: agent and user simulator on separate GPUs")
AGENT_PORT, USER_PORT = 11434, 11435
procs = []
for i, port in enumerate((AGENT_PORT, USER_PORT)):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), OLLAMA_HOST=f"127.0.0.1:{port}",
               OLLAMA_NUM_PARALLEL="3", OLLAMA_MAX_LOADED_MODELS="1", OLLAMA_KEEP_ALIVE="60m")
    procs.append(subprocess.Popen(["ollama", "serve"], env=env,
                 stdout=open(f"{W}/ollama_{port}.log", "w"), stderr=subprocess.STDOUT))
time.sleep(8)
for m in MODELS + [USER_MODEL]:
    sh(f"OLLAMA_HOST=127.0.0.1:{AGENT_PORT} ollama pull {m}")
sh(f"OLLAMA_HOST=127.0.0.1:{AGENT_PORT} ollama list")
import urllib.request
for m in MODELS + [USER_MODEL]:
    req = urllib.request.Request(f"http://127.0.0.1:{AGENT_PORT}/api/show", data=json.dumps({"model": m}).encode(),
                                 headers={"Content-Type": "application/json"})
    info = json.loads(urllib.request.urlopen(req).read())
    print(m, info.get("details"), flush=True)
'''

CELL_RUN = r'''# 3. Campaign (resumable; one JSONL per model)
sys.path.insert(0, W)
import harness as H, threading, queue, traceback
from concurrent.futures import ThreadPoolExecutor
t_campaign = time.time()
def out_path(m): return f"{W}/results_{m.replace(':', '_')}.jsonl"
def done_keys(m):
    p = out_path(m); s = set()
    if os.path.exists(p):
        for line in open(p):
            r = json.loads(line); s.add((r["task_id"], r["arm"]))
    return s
lock = threading.Lock()
def work(m, tid, arm):
    if time.time() - t_campaign > BUDGET_S:
        return "skipped_budget"
    client = H.OllamaClient(f"http://127.0.0.1:{AGENT_PORT}", m, num_ctx=24576)
    user_client = H.OllamaClient(f"http://127.0.0.1:{USER_PORT}", USER_MODEL, num_ctx=8192)
    try:
        r = H.run_episode(client, tid, arm, user_client)
    except Exception as e:
        r = {"task_id": tid, "arm": arm, "model": m, "reward": 0.0, "harness_error": repr(e),
             "tb": traceback.format_exc()[-2000:]}
    with lock:
        open(out_path(m), "a").write(json.dumps(r) + "\n")
    st = r.get("stats", {})
    print(f"{m} task {tid:3d} {arm} reward={r['reward']} stop={r.get('stop_reason')} "
          f"t={st.get('episode_seconds', 0):.0f}s calls={st.get('llm_calls')} user={st.get('user_calls')}", flush=True)
    return r["reward"]
for m, tids in PLAN:
    have = done_keys(m)
    jobs = [(m, t, a) for t in tids for a in ARMS if (t, a) not in have]
    print(f"== {m}: {len(jobs)} episodes to run", flush=True)
    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(lambda j: work(*j), jobs))
print("campaign wall time (s):", round(time.time() - t_campaign))
'''

CELL_SUMMARY = r'''# 4. Quick summary (the real analysis runs offline)
import collections, glob
for p in sorted(glob.glob(f"{W}/results_*.jsonl")):
    rows = [json.loads(l) for l in open(p)]
    agg = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        agg[r["arm"]][0] += r["reward"]; agg[r["arm"]][1] += 1
    print(os.path.basename(p), {a: f"{int(s)}/{n}" for a, (s, n) in sorted(agg.items())},
          "errors:", sum(1 for r in rows if "harness_error" in r))
for p in procs: p.terminate()
'''


def build(mode):
    if mode == "smoke":
        # smoke tasks are taken from the extension set, so no primary task is seen twice
        cfg = ('MODELS = ["qwen2.5:7b"]\nUSER_MODEL = "qwen2.5:14b"\nARMS = ["A0", "A1", "A2"]\n'
               'PLAN = [("qwen2.5:7b", [1, 3])]\nBUDGET_S = 3 * 3600\n')
    else:
        p, e = FROZEN["primary_40"], FROZEN["extension_75"]
        cfg = ('MODELS = ["qwen2.5:7b", "qwen2.5:3b"]\nUSER_MODEL = "qwen2.5:14b"\nARMS = ["A0", "A1", "A2"]\n'
               f'PRIMARY = {p}\nEXTENSION = {e}\n'
               'PLAN = [("qwen2.5:7b", PRIMARY), ("qwen2.5:3b", PRIMARY),\n'
               '        ("qwen2.5:7b", EXTENSION), ("qwen2.5:3b", EXTENSION)]\n'
               'BUDGET_S = int(10.5 * 3600)  # stop starting new episodes after 10.5 h\n')
    setup = CELL_SETUP.replace("VENDOR_SRC", repr(VENDOR)).replace("HARNESS_SRC", repr(HARNESS))
    nb = {"cells": [md(f"# Cheap-guardrails pilot ({mode}): guardrails for small local tool-using agents on tau-bench retail\n"
                       "Self-contained. Needs **GPU T4 x2** and **Internet on**."),
                    code("# 0. Configuration\n" + cfg), code(setup), code(CELL_SERVERS), code(CELL_RUN), code(CELL_SUMMARY)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}, "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}
    d = os.path.join(HERE, "kernels", mode)
    os.makedirs(d, exist_ok=True)
    fname = f"guardrails-pilot-{mode}.ipynb"
    json.dump(nb, open(os.path.join(d, fname), "w", encoding="utf-8"), indent=1)
    user = os.environ.get("KAGGLE_USERNAME", "your-kaggle-username")
    meta = {"id": f"{user}/guardrails-pilot-{mode}", "title": f"guardrails-pilot-{mode}",
            "code_file": fname, "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [], "competition_sources": [], "kernel_sources": []}
    json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=2)
    print("built", d)


if __name__ == "__main__":
    build("smoke")
    build("full")
