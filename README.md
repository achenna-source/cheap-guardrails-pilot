# Cheap-guardrails pilot: harness, frozen protocol and traces

Code, frozen protocol, full traces and exploratory analyses of the pilot study reported in
*Well-Formed but Wrong: Residual Failures of Small Local LLM Agents under Lightweight
Guardrails* (Abdelbassette Chenna, University of Batna 2, 2026; see
[Exploratory analyses](#exploratory-analyses)).
The harness is also the measured baseline of a final-year engineering project on
reliable business agents.

## Question

Schema validation and grammar-constrained decoding remove malformed tool calls by
construction, and human confirmation can block wrong database writes. The pilot asks
what failures remain when small local models act as customer-service agents.

## Setup

| Item | Choice |
|------|--------|
| Tasks | tau-bench retail (Sierra, MIT licence), test split, commit `59a200c6d575d595120f1cb70fea53cef0632f6b`; 40 tasks drawn with seed 20261118 and frozen before any run (`frozen_tasks.json`) |
| Success | tau-bench rule: final database hash equals the gold replay, and every required output appears in an agent message (no LLM judge) |
| Agent | Qwen2.5-Instruct 3B and 7B, Q4_K_M, Ollama 0.40.16, temperature 0, at most 30 model turns |
| Customer | tau-bench LLM user simulator, prompt copied verbatim, played by Qwen2.5-14B Q4_K_M at temperature 0 |
| Hardware | Kaggle, 2 x NVIDIA T4 (agent on one GPU, customer on the other), three episodes in parallel |

| Arm | Guardrail |
|-----|-----------|
| A0 | none: native tool-call template, free decoding |
| A1 | post-hoc JSON-schema validation of every tool call, one corrective retry, then rejection |
| A2 | grammar-constrained decoding: Ollama `format` = JSON schema of all actions (adds a format note and a `respond` action to the prompt) |

The confirmation step is analysed offline: every write call that reaches execution is
what a human confirmer would see. `PREREGISTRATION.md` gives the frozen protocol and
its single amendment (the scripted customer of the first smoke test was replaced by the
tau-bench user simulator before the full run).

## Main results (40 frozen tasks, one run per arm)

| Model | Arm | Success | Writes proposed | Wrong writes executed | Failures with no wrong write proposal | s per episode |
|-------|-----|---------|-----------------|-----------------------|---------------------------------------|---------------|
| 3B | A0 | 2/40 | 53 | 3 | 25/38 | 220 |
| 3B | A1 | 2/40 | 56 | 1 | 19/38 | 178 |
| 3B | A2 | 2/40 | 56 | 3 | 14/38 | 53 |
| 7B | A0 | 5/40 | 76 | 18 | 13/35 | 299 |
| 7B | A1 | 5/40 | 107 | 12 | 11/35 | 281 |
| 7B | A2 | 4/40 | 100 | 7 | 11/36 | 127 |

- No arm differs from another (exact McNemar p = 1.0). On the 37 tasks where the A1
  check never fired, A1 ran exactly A0's protocol, yet 6 outcomes changed for the 7B
  model: batched serving and the simulated customer are not deterministic at temperature 0.
- Every native tool call parsed. Under A0 and A1 the 7B model sometimes wrote a call as
  JSON text inside its message (2 and 6 of the 40 episodes); such a call goes to the
  customer and is never executed, and only A2 removes it. 81 to 99% of the shop's error
  messages are "not found", i.e. identifiers absent from the database.
- Seven of the 115 test tasks can be passed by an agent that changes nothing: their gold
  actions leave the database unchanged (for tasks 12 and 106 of the frozen 40, because the
  shop rejects the gold write) and they need no output. Both 3B successes, in every arm,
  are tasks 12 and 106; on the 38 other frozen tasks the 3B model never succeeded and the
  7B model solved 3 per arm, a different set each time.
- The 7B model also ran the 75 other test tasks (12, 7 and 11 successes for A0, A1, A2);
  the 3B run covered 48 to 49 of them before the GPU budget ran out (7, 6, 6). Five of
  these 75 tasks are among the seven above.

Full numbers: `results/analysis_{primary_40,extension_75,all}.{json,txt}`.

## Files

| Path | Role |
|------|------|
| `harness.py` | environment replica, reward, the three arms, tau-bench user simulator, Ollama client, local mocks (identical to the code run on Kaggle) |
| `vendor_tau.py` | copies the retail tools, data, policy and test tasks from a tau-bench checkout into `tbr/` (avoids the litellm import) |
| `test_harness.py` | local test without any LLM: replaying the gold actions must score 1 on all 115 tasks in all 3 arms |
| `build_notebook.py` | builds the self-contained Kaggle notebooks |
| `notebooks/guardrails-pilot-full.ipynb` | the exact notebook of the full campaign |
| `analyze.py` | success with Wilson intervals, exact McNemar, failure classes, confirmation-step counts, noise floor |
| `results/results_qwen2.5_{3b,7b}.jsonl` | one line per episode: reward, stop reason, statistics, full trace, dialogue, gold actions |
| `results/smoke_discarded/` | the two smoke tests, kept for transparency and excluded from every analysis |

## Reproduce

```bash
git clone https://github.com/sierra-research/tau-bench
git -C tau-bench checkout 59a200c6d575d595120f1cb70fea53cef0632f6b
python -I vendor_tau.py tau-bench .
python -I test_harness.py
python -I analyze.py results --set primary_40
```

To rerun the campaign: `KAGGLE_USERNAME=<you> python -I build_notebook.py`, then
`kaggle kernels push -p kernels/full` (GPU T4 x2, Internet on). Expect small differences
between runs: the noise floor above is part of the result.

## Exploratory analyses

The scripts in `exploratory/` produce the additional analyses of the article. They
were written after the full run, are not part of the frozen protocol, and the article
reports them as exploratory. They read the traces in `results/` and replay the executed
calls on the shop database, so run them from the repository root after `vendor_tau.py`.

| Script | Output | What it computes |
|--------|--------|------------------|
| `exploratory/extra_analysis.py` | `exploratory/outputs/extra_analysis.{json,txt}` | replays every executed call on a fresh database; provenance of the identifiers behind "not found" errors; identifiers the simulated customer invented; per-episode funnel; repeated failing calls; overlap of solved tasks; success by number of required writes; passive tasks |
| `exploratory/provenance_guard.py` | `exploratory/outputs/provenance_guard.{json,txt}` | offline estimate of what an identifier-provenance check would have blocked |
| `exploratory/textual_calls.py` | `exploratory/outputs/textual_calls.{json,txt}` | calls written as JSON text inside agent messages, and turns with several native calls (only the first is executed) |
| `exploratory/review_checks.py` | `exploratory/outputs/review_checks.txt` | agent and simulator time, tokens per turn, schema-invalid executed calls, noise floor and identical tool sequences on quiet tasks |
| `exploratory/make_figures.py` | `exploratory/figures/` | the three result figures of the article (needs matplotlib) |

```bash
python -I exploratory/extra_analysis.py . exploratory/outputs/extra_analysis.json > exploratory/outputs/extra_analysis.txt
python -I exploratory/provenance_guard.py . exploratory/outputs/provenance_guard.json > exploratory/outputs/provenance_guard.txt
python -I exploratory/textual_calls.py . exploratory/outputs/textual_calls.json > exploratory/outputs/textual_calls.txt
python -I exploratory/review_checks.py . > exploratory/outputs/review_checks.txt
python -I exploratory/make_figures.py exploratory/outputs results exploratory/figures
```

Rerun on the published traces, the four analysis scripts reproduce the files in
`exploratory/outputs/` exactly; the figure files differ only in their embedded metadata.

## Failure classes (one per failed episode, first rule that applies)

- W1 an executed write with wrong arguments; W2 an executed write the task did not ask for
- M1 a required write attempted but rejected by the shop; M2 a required write never attempted
- O correct database, required information missing from the replies
- X any other mismatch; E harness error

## Licence

Code: MIT (see `LICENSE`). Task data, tools and policy belong to tau-bench (MIT,
Copyright 2024 Sierra); they are not redistributed here, except as they appear inside
the episode traces.
