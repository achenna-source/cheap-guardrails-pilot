# Pilot protocol (frozen 2026-11-18, before the full run)

## Question
On third-party customer-service tasks with state-verified success, what do three
cheap guardrails change for small local LLM agents, and which failures remain?
Descriptive pilot. No confirmatory hypothesis; paired comparisons are reported
with exact McNemar p-values as descriptive statistics only.

## Tasks
tau-bench retail, test split (115 tasks), commit 59a200c6d575d595120f1cb70fea53cef0632f6b.
Primary set: 40 task ids drawn with `random.Random(20261118).sample(range(115), 40)`,
stored in `frozen_tasks.json` (sha256 of the JSON string
1eecb8bc2b1577e35b97e35e1264946dca2f360fa2291e19f251547a5833e7d6).
Extension set: the other 75 tasks, run only if the GPU budget allows, reported
separately. Five extension tasks need no write and no output (any passive agent
passes them): 10, 25, 50, 57, 65. No primary task has this property.

## Agent, arms, decoding
Qwen2.5-Instruct 7B and 3B, Ollama Q4_K_M, temperature 0, seed 7, num_ctx 24576,
num_predict 1024, at most 30 agent calls per episode. System prompt = tau-bench
retail policy (wiki.md). Arms A0 (free decoding, native tool template), A1 (post-hoc
JSON-schema validation with additionalProperties=false, one corrective retry,
then rejection), A2 (Ollama `format` = JSON schema of the action union, plus a
`respond` action and a short output-format note in the system prompt).

## Outcome and analysis rules
Success = tau-bench rule (final database hash equals gold replay hash, and every
required output string appears in an agent message). Wilson 95% intervals.
Failure classes, first rule that applies: E harness error; W1 executed write with
wrong arguments (tool present in gold writes); W2 executed write whose tool is not
among the gold writes; M1 a required write missing after at least one write
attempt; M2 a required write never attempted; O database correct but required
output missing; X other. Confirmation-step load: write calls that reached
execution (A1 rejections excluded) per episode, and the share not matching a
gold write. Tokens and wall-clock seconds per episode.

## Amendment 1 (2026-11-18, after a 9-episode smoke test, before the full run)
The smoke test (Qwen2.5-7B, tasks 0, 2, 5, all arms) used a scripted user that
sent the whole instruction first and then always answered with a confirmation.
It ran cleanly but exposed a validity problem: tau-bench instructions encode
conditional customer behaviour (for example, changing one's mind after a
confirmation request) that a fixed reply cannot reproduce. The scripted user is
therefore replaced by tau-bench's own LLM user simulator (system prompt copied
verbatim) played by Qwen2.5-14B-Instruct Q4_K_M at temperature 0 on a separate
GPU. The episode now ends on ###STOP### from the user, on transfer to a human, or
after 30 agent calls. The smoke-test episodes are discarded and not reported.
No other element of the protocol changed. A second smoke test uses extension
tasks 1 and 3 only.
