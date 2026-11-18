"""Pilot harness: small local LLM agents on tau-bench retail, three guardrail arms.

Arms
  A0  free decoding, native tool-call template (Ollama `tools`), no checks.
  A1  A0 + post-hoc JSON-schema validation of every tool call, one corrective
      retry; a call that fails twice is rejected (never executed).
  A2  grammar-constrained decoding: Ollama `format` = JSON schema of the action
      union (one branch per tool + a `respond` branch), so every output is a
      schema-valid action by construction.

The confirmation gate is not a run arm. It is analysed offline from the traces:
every executed write call is what a human confirmer would be shown.

User side (Amendment 1, 2026-11-18, before the full run): tau-bench's own LLM
user simulator (same system prompt, verbatim) played by an open-weight model at
temperature 0. A scripted user was used in the 9-episode smoke test and dropped,
because many tau-bench instructions encode conditional customer behaviour that a
fixed reply cannot follow. The episode stops when the simulated user emits
###STOP###, after `transfer_to_human_agents`, or after MAX_STEPS agent calls.

Reward: identical rule to tau-bench `Env.calculate_reward` (hash of the final
database equals the hash obtained by replaying the gold actions on a fresh copy,
and every required output string appears in an agent message).
"""
import copy, hashlib, json, re, time, urllib.request

from tbr.data import load_data
from tbr.tasks_test import TASKS_TEST
from tbr.tools import ALL_TOOLS
from tbr.tbtypes import Action

WIKI = open(__import__("os").path.join(__import__("os").path.dirname(
    __import__("tbr").__file__), "wiki.md"), encoding="utf-8").read()

WRITE_TOOLS = {
    "cancel_pending_order", "exchange_delivered_order_items",
    "modify_pending_order_address", "modify_pending_order_items",
    "modify_pending_order_payment", "modify_user_address",
    "return_delivered_order_items",
}
TERMINATE_TOOLS = {"transfer_to_human_agents"}
MAX_STEPS = 30

RESPOND_DOC = {
    "type": "function",
    "function": {
        "name": "respond",
        "description": "Send a message to the customer. Use it to ask for confirmation, to give information, or to close the conversation.",
        "parameters": {
            "type": "object",
            "properties": {"content": {"type": "string", "description": "The message to the customer."}},
            "required": ["content"],
        },
    },
}
USER_HELLO = "Hi! How can I help you today?"


def user_system_prompt(instruction):
    """Verbatim copy of tau-bench LLMUserSimulationEnv.build_system_prompt."""
    instruction_display = ("\n\nInstruction: " + instruction + "\n") if instruction is not None else ""
    return f"""You are a user interacting with an agent.{instruction_display}
Rules:
- Just generate one line at a time to simulate the user's message.
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
- If the instruction goal is satisified, generate '###STOP###' as a standalone message without anything else to end the conversation.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in the instruction."""


class LLMUser:
    """tau-bench LLM user simulator (roles flipped: the agent speaks as 'user')."""
    def __init__(self, client):
        self.client = client
        self.calls = 0
        self.seconds = 0.0

    def _gen(self):
        res = self.client.chat(self.messages, None, None)
        self.calls += 1
        self.seconds += res["_wall"]
        content = (res.get("message", {}).get("content") or "").strip()
        self.messages.append({"role": "assistant", "content": content})
        return content

    def reset(self, instruction):
        self.messages = [{"role": "system", "content": user_system_prompt(instruction)},
                         {"role": "user", "content": USER_HELLO}]
        return self._gen()

    def step(self, agent_text):
        self.messages.append({"role": "user", "content": agent_text})
        return self._gen()


A2_ADDENDUM = (
    "\n\n# Output format\nEvery output you produce is one JSON object. To call a tool, "
    "output {\"name\": <tool name>, \"arguments\": {...}}. To write to the customer, "
    "output {\"name\": \"respond\", \"arguments\": {\"content\": <your message>}}."
)

TOOLS_MAP = {t.get_info()["function"]["name"]: t for t in ALL_TOOLS}
TOOLS_INFO = [t.get_info() for t in ALL_TOOLS]


# ---------------------------------------------------------------- environment
def _to_hashable(item):
    if isinstance(item, dict):
        return tuple((k, _to_hashable(v)) for k, v in sorted(item.items()))
    if isinstance(item, list):
        return tuple(_to_hashable(e) for e in item)
    if isinstance(item, set):
        return tuple(sorted(_to_hashable(e) for e in item))
    return item


def data_hash(data):
    return hashlib.sha256(str(_to_hashable(data)).encode("utf-8")).hexdigest()


_BASE_DATA = None
def fresh_data():
    global _BASE_DATA
    if _BASE_DATA is None:
        _BASE_DATA = load_data()
    return copy.deepcopy(_BASE_DATA)


_GOLD_HASH = {}
def gold_hash(task_id):
    if task_id not in _GOLD_HASH:
        data = fresh_data()
        for a in TASKS_TEST[task_id].actions:
            if a.name in TOOLS_MAP and a.name not in TERMINATE_TOOLS:
                try:
                    TOOLS_MAP[a.name].invoke(data=data, **a.kwargs)
                except Exception:
                    pass
        _GOLD_HASH[task_id] = data_hash(data)
    return _GOLD_HASH[task_id]


class RetailEnv:
    def __init__(self, task_id):
        self.task_id = task_id
        self.task = TASKS_TEST[task_id]
        self.data = fresh_data()
        self.actions = []          # executed actions, tau-bench style
        self.done = False

    def call(self, name, kwargs):
        self.actions.append(Action(name=name, kwargs=kwargs))
        if name in TOOLS_MAP:
            try:
                obs = TOOLS_MAP[name].invoke(data=self.data, **kwargs)
            except Exception as e:
                obs = f"Error: {e}"
            if name in TERMINATE_TOOLS:
                self.done = True
        else:
            obs = f"Unknown action {name}"
        return obs

    def respond(self, content):
        self.actions.append(Action(name="respond", kwargs={"content": content}))

    def reward(self):
        r_actions = data_hash(self.data) == gold_hash(self.task_id)
        outputs = {}
        for out in self.task.outputs:
            outputs[out] = any(
                a.name == "respond" and out.lower() in a.kwargs["content"].lower().replace(",", "")
                for a in self.actions)
        r_outputs = all(outputs.values()) if outputs else True
        return {"reward": float(r_actions and r_outputs), "r_actions": r_actions,
                "r_outputs": r_outputs, "outputs": outputs}


# ---------------------------------------------------------------- schemas
def _strict(schema):
    """Copy of a JSON schema with additionalProperties=false on every object."""
    s = copy.deepcopy(schema)
    def walk(x):
        if isinstance(x, dict):
            if x.get("type") == "object":
                x.setdefault("additionalProperties", False)
                x.setdefault("properties", {})
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(s)
    return s


def action_union_schema():
    branches = []
    for info in TOOLS_INFO + [RESPOND_DOC]:
        f = info["function"]
        branches.append({
            "type": "object",
            "properties": {"name": {"type": "string", "enum": [f["name"]]},
                           "arguments": _strict(f["parameters"])},
            "required": ["name", "arguments"],
            "additionalProperties": False,
        })
    return {"anyOf": branches}


PARAM_SCHEMAS = {i["function"]["name"]: _strict(i["function"]["parameters"]) for i in TOOLS_INFO}


def validate_call(name, args):
    """Post-hoc check used by arm A1. Returns None if valid, else an error string."""
    import jsonschema
    if name not in PARAM_SCHEMAS:
        return f"unknown tool '{name}'"
    if not isinstance(args, dict):
        return "arguments must be a JSON object"
    try:
        jsonschema.validate(args, PARAM_SCHEMAS[name])
    except jsonschema.ValidationError as e:
        return f"{e.message} (at {'/'.join(str(p) for p in e.path) or 'arguments'})"
    return None


# ---------------------------------------------------------------- LLM client
class OllamaClient:
    def __init__(self, url, model, num_ctx=16384, seed=7, timeout=600):
        self.url, self.model, self.num_ctx, self.seed, self.timeout = url, model, num_ctx, seed, timeout

    def chat(self, messages, tools, fmt=None):
        body = {"model": self.model, "messages": messages, "stream": False,
                "options": {"temperature": 0, "seed": self.seed, "num_ctx": self.num_ctx,
                            "num_predict": 1024},
                "keep_alive": "60m"}
        if tools:
            body["tools"] = tools
        if fmt is not None:
            body["format"] = fmt
        req = urllib.request.Request(self.url + "/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            res = json.loads(r.read())
        res["_wall"] = time.time() - t0
        return res


# ---------------------------------------------------------------- parsing
TOOLCALL_HINT = re.compile(r"<tool_call>|^\s*\{\s*\"name\"\s*:", re.S)


def parse_native(msg):
    """A0/A1: first native tool call, else text. Flags text that looks like a failed call."""
    tcs = msg.get("tool_calls") or []
    if tcs:
        f = tcs[0].get("function", {})
        args = f.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                pass
        return {"kind": "tool", "name": f.get("name"), "args": args, "malformed_text": False}
    content = msg.get("content") or ""
    return {"kind": "text", "content": content, "malformed_text": bool(TOOLCALL_HINT.search(content))}


def parse_constrained(msg):
    """A2: the content is a JSON action (Ollama may also surface it as a tool call)."""
    tcs = msg.get("tool_calls") or []
    if tcs:
        p = parse_native(msg)
        if p["name"] == "respond" and isinstance(p["args"], dict):
            return {"kind": "text", "content": str(p["args"].get("content", "")), "malformed_text": False}
        return p
    content = msg.get("content") or ""
    try:
        obj = json.loads(content)
    except Exception:
        return {"kind": "text", "content": content, "malformed_text": True}
    name, args = obj.get("name"), obj.get("arguments", {})
    if name == "respond":
        return {"kind": "text", "content": str(args.get("content", "")), "malformed_text": False}
    return {"kind": "tool", "name": name, "args": args, "malformed_text": False}


# ---------------------------------------------------------------- episode
def run_episode(client, task_id, arm, user_client):
    env = RetailEnv(task_id)
    task = env.task
    system = WIKI + (A2_ADDENDUM if arm == "A2" else "")
    tools = TOOLS_INFO + ([RESPOND_DOC] if arm == "A2" else [])
    fmt = action_union_schema() if arm == "A2" else None
    user = LLMUser(user_client)
    first = user.reset(task.instruction)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": first}]
    trace, stats = [], {"llm_calls": 0, "prompt_tokens": 0, "gen_tokens": 0, "llm_seconds": 0.0,
                        "malformed_text": 0, "invalid_calls": 0, "retries": 0, "rejected": 0,
                        "env_errors": 0, "user_turns": 0}
    stop_reason, pending_retry = "max_steps", None
    dialogue = [{"role": "user", "content": first}]
    t_start = time.time()
    for step in range(MAX_STEPS):
        res = client.chat(messages, tools, fmt)
        stats["llm_calls"] += 1
        stats["prompt_tokens"] += res.get("prompt_eval_count", 0) or 0
        stats["gen_tokens"] += res.get("eval_count", 0) or 0
        stats["llm_seconds"] += res["_wall"]
        msg = res.get("message", {})
        p = parse_constrained(msg) if arm == "A2" else parse_native(msg)
        rec = {"step": step, "raw_content": (msg.get("content") or "")[:2000],
               "raw_tool_calls": msg.get("tool_calls"), **{k: v for k, v in p.items()}}

        if p["kind"] == "tool":
            name, args = p["name"], p["args"]
            if arm == "A1":
                err = validate_call(name, args)
                if err is not None:
                    stats["invalid_calls"] += 1
                    rec["guardrail_error"] = err
                    messages.append({"role": "assistant", "content": "",
                                     "tool_calls": [{"function": {"name": str(name), "arguments": args if isinstance(args, dict) else {}}}]})
                    if pending_retry is None:
                        stats["retries"] += 1
                        pending_retry = name
                        note = (f"Guardrail check failed: {err}. The call was NOT executed. "
                                "Re-issue a corrected call.")
                        rec["guardrail"] = "retry"
                    else:
                        stats["rejected"] += 1
                        pending_retry = None
                        note = f"Guardrail check failed again: {err}. The call was rejected and NOT executed."
                        rec["guardrail"] = "rejected"
                    messages.append({"role": "tool", "content": note, "tool_name": str(name)})
                    trace.append(rec)
                    continue
                pending_retry = None
            if not isinstance(args, dict):
                args = {}
            obs = env.call(str(name), args)
            if obs.startswith("Error") or obs.startswith("Unknown action"):
                stats["env_errors"] += 1
            rec["observation"] = obs[:1500]
            rec["write"] = name in WRITE_TOOLS
            messages.append({"role": "assistant", "content": "",
                             "tool_calls": [{"function": {"name": str(name), "arguments": args}}]})
            messages.append({"role": "tool", "content": obs, "tool_name": str(name)})
            trace.append(rec)
            if env.done:
                stop_reason = "transfer"
                break
        else:
            if p["malformed_text"]:
                stats["malformed_text"] += 1
            env.respond(p["content"])
            messages.append({"role": "assistant", "content": p["content"]})
            trace.append(rec)
            reply = user.step(p["content"])
            stats["user_turns"] += 1
            dialogue += [{"role": "agent", "content": p["content"]}, {"role": "user", "content": reply}]
            if "###STOP###" in reply:
                stop_reason = "user_stop"
                break
            messages.append({"role": "user", "content": reply})
    stats["episode_seconds"] = time.time() - t_start
    stats["user_calls"], stats["user_seconds"] = user.calls, user.seconds
    rw = env.reward()
    executed = [a.to_dict() for a in env.actions]
    return {"task_id": task_id, "arm": arm, "model": client.model, "user_model": user_client.model,
            **rw, "stop_reason": stop_reason, "dialogue": dialogue,
            "stats": stats, "executed": executed, "trace": trace,
            "gold_actions": [a.to_dict() for a in task.actions], "gold_outputs": task.outputs}


# ---------------------------------------------------------------- mock client (local tests)
class GoldReplayClient:
    """Replays the gold actions of one task, then answers with the required outputs.

    Used only to test the harness locally without any LLM. `mode` selects the
    output channel the mock imitates: 'native' (tool_calls) or 'json' (A2 content).
    `corrupt` injects one schema-invalid call first (tests the A1 path).
    """
    def __init__(self, task_id, mode="native", corrupt=False):
        self.model = "gold-replay"
        acts = [a for a in TASKS_TEST[task_id].actions]
        self.queue = []
        if corrupt:
            self.queue.append(("tool", "get_order_details", {"order": "#W0000000"}))
        for a in acts:
            self.queue.append(("tool", a.name, a.kwargs))
        outs = TASKS_TEST[task_id].outputs
        self.queue.append(("text", "Done. " + " ".join(outs), None))
        self.queue.append(("text", "Goodbye.", None))
        self.mode = mode

    def chat(self, messages, tools, fmt=None):
        kind, name, args = self.queue.pop(0) if self.queue else ("text", "Goodbye.", None)
        if self.mode == "json":
            if kind == "tool":
                content = json.dumps({"name": name, "arguments": args})
            else:
                content = json.dumps({"name": "respond", "arguments": {"content": name}})
            msg = {"role": "assistant", "content": content}
        elif kind == "tool":
            msg = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}
        else:
            msg = {"role": "assistant", "content": name}
        return {"message": msg, "prompt_eval_count": 0, "eval_count": 0, "_wall": 0.0}


class MockUserClient:
    """Local test stand-in for the user simulator: two turns, then ###STOP###."""
    model = "mock-user"
    def __init__(self):
        self.n = 0
    def chat(self, messages, tools, fmt=None):
        self.n += 1
        content = "I need help with my order." if self.n == 1 else ("Yes." if self.n < 3 else "###STOP###")
        return {"message": {"role": "assistant", "content": content}, "_wall": 0.0}
