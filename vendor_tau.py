"""Build a minimal, dependency-light copy of the tau-bench retail domain.

tau-bench (Sierra, MIT licence) imports litellm at package import time. The pilot
only needs the retail tools, data, policy (wiki) and test tasks, so we copy those
files into a local package `tbr` and rewrite the two internal imports.
Usage: python -I vendor_tau.py <path-to-tau-bench-repo> <output-dir>
"""
import os, re, shutil, sys

TBTYPES = '''# Minimal stand-ins for the upstream Action / Task types (same fields).
class Action:
    def __init__(self, name, kwargs):
        self.name = name
        self.kwargs = kwargs
    def to_dict(self):
        return {"name": self.name, "kwargs": self.kwargs}


class Task:
    def __init__(self, user_id, actions, instruction, outputs, annotator=None):
        self.user_id = user_id
        self.actions = actions
        self.instruction = instruction
        self.outputs = outputs
        self.annotator = annotator
'''


def build(repo, out):
    src = os.path.join(repo, "tau_bench")
    pkg = os.path.join(out, "tbr")
    if os.path.isdir(pkg):
        shutil.rmtree(pkg)
    os.makedirs(os.path.join(pkg, "tools"))
    os.makedirs(os.path.join(pkg, "data"))
    shutil.copy(os.path.join(repo, "LICENSE"), os.path.join(pkg, "LICENSE_tau_bench"))
    shutil.copy(os.path.join(src, "envs", "tool.py"), os.path.join(pkg, "tool.py"))
    # types: the tasks only need Action and Task; plain classes avoid pydantic
    open(os.path.join(pkg, "tbtypes.py"), "w", encoding="utf-8").write(TBTYPES)
    shutil.copy(os.path.join(src, "envs", "retail", "wiki.md"), os.path.join(pkg, "wiki.md"))
    for f in ("orders.json", "products.json", "users.json", "__init__.py"):
        shutil.copy(os.path.join(src, "envs", "retail", "data", f), os.path.join(pkg, "data", f))
    tools_dir = os.path.join(src, "envs", "retail", "tools")
    for f in os.listdir(tools_dir):
        if f.endswith(".py"):
            shutil.copy(os.path.join(tools_dir, f), os.path.join(pkg, "tools", f))
    shutil.copy(os.path.join(src, "envs", "retail", "tasks_test.py"), os.path.join(pkg, "tasks_test.py"))
    open(os.path.join(pkg, "__init__.py"), "w").write("# vendored subset of tau-bench retail (MIT, Sierra)\n")
    n = 0
    for root, _, files in os.walk(pkg):
        for f in files:
            if not f.endswith(".py"):
                continue
            p = os.path.join(root, f)
            s = open(p, encoding="utf-8").read()
            s2 = s.replace("from tau_bench.envs.tool import Tool", "from tbr.tool import Tool")
            s2 = s2.replace("from tau_bench.types import", "from tbr.tbtypes import")
            if "tau_bench" in s2:
                raise SystemExit(f"unexpected tau_bench import left in {p}")
            if s2 != s:
                n += 1
                open(p, "w", encoding="utf-8").write(s2)
    return n

if __name__ == "__main__":
    n = build(sys.argv[1], sys.argv[2])
    print("vendored tbr, rewrote", n, "files")
