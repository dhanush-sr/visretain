# VisRetain — implementation plan

## The question

Run the same image and question **without rounding (FP16)** and **with rounding (INT4,
INT2)**, at the same memory budget. Does rounding change what the model throws away from its
memory — and does it lose the visual evidence the answer needs later?

## Method (agreed 4 Nov)

The project is a **measurement**, so the instrument is built and **proven one stage at a
time**, following the model's own flow. For every stage:

1. **Research** just enough for that stage (Qwen's source code for facts; papers for design
   choices).
2. **Decide and write down** the choices and the pass criteria in `STAGES.md` **before**
   building the checks.
3. **Build**, then **verify** on the real model, including a batch over many images.
4. **Show it** visually on the stages page, with each check's result.
5. Only then move to the next stage. Failures are reported, diagnosed, and never quietly
   re-defined; a revised check is pre-registered before it runs.

All settings live in `config.json`. Nothing is hard-coded.

## Stages

| Stage | What | How we know it is right | Status |
|---|---|---|---|
| 1 | Image → visual tokens | resize rule, patch layout, token count, Qwen's positions, causal routing test, slot check, overflow guard | **done; see STAGES.md** |
| 2 | Attention capture | our attention = Qwen's reference formula; reproduces the model's real output; keys = cache; non-interference; negative controls | **done; see STAGES.md** |
| 3 | Rounding the keys | error within half a step on real keys; 16-bit is exact; size per bit width | next |
| 4 | Eviction at FP16 | budget held, recent tokens protected, deleted never return | next |
| 5 | Eviction with rounding | 16-bit "rounding" reproduces FP16 exactly; dose response | next |
| 6 | Evidence | evidence tokens match what the answer uses (GQA boxes) | later |
| 7 | Metrics and statistics | metrics give known answers on constructed cases | later |

## Kill test (planned, not started) — decides go / pivot / stop before stages 3–7

**Question:** under realistic conditions, does forgetting cost correct answers, and does rounding
make it worse?

**Protocol — "describe first, ask later":** the model describes a photo (~150 tokens) with a
limited cache, evicting as it goes; *then* it is asked a GQA question about a detail and answers
from what is left. Answers are scored against GQA's human labels. Eviction is real (evicted
entries are masked from attention, so the answer can change); the cache's keys and values are
really rounded.

| Arm | Cache | Precision | Answers |
|---|---|---|---|
| A | full | FP16 | the ceiling; must reproduce the plain model exactly (sanity) |
| B | full | INT4, INT2 | does rounding alone hurt? |
| C | limited | FP16 | **does forgetting hurt?** |
| D | limited | INT4, INT2 | **does rounding make forgetting worse?** |

~100 questions, several budgets, paired per question (McNemar), 5-point threshold.

| Result | Decision |
|---|---|
| D ≥ 5 points worse than C, significant | **GO** — original thesis |
| C ≥ 5 points worse than A, but D ≈ C | **PIVOT** — the eviction rule, not rounding, is the problem |
| C ≈ A | **STOP** — forgetting doesn't cost answers here; report a negative result |

Prediction recorded before running: PIVOT.

Earlier exploratory runs (yesterday's full pipeline: rounding changes 8.8% / 33.5% of the
eviction set; evidence loss not increased) ran *ahead* of this verification. They are kept
and shown as "the full pipeline, being verified stage by stage", not as final results.

## Open decisions (team)

- Vision encoder in **bfloat16** instead of fp16 (overflow found in stage 1)?
- Scope: retrieval (RAG) strategies are out of scope for now.
