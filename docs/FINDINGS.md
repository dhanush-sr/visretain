# VisRetain — Stage 1 + Stage 2 gate, first run on a real VLM

**Date:** 3 November 2026 · **Project:** PW26_KT_03 · **Model:** Qwen2.5-VL-3B-Instruct (FP16, MPS)

## What was built

| File | What it is |
|---|---|
| `visretain.py` | Fake KV quantizers, attention scoring, the Jensen bias term, and the rank-agreement metrics. Model-agnostic, CPU-only, with a runnable self-check (`python3 visretain.py`). |
| `run_kill_test.py` | Attaches the harness to Qwen2.5-VL-3B via a custom attention function and runs the shadow-ranking experiment. |
| `analyze.py` | Aggregates logs into the gate answer. Runs on CPU from saved JSON, no model needed. |

**Method: shadow ranking.** One real decode pass produces one cache. Every arm scores
*that same cache* — FP16, INT4, INT2, and bias-corrected. Nothing is actually evicted.

This is deliberate. It makes the placebo bug that broke the earlier prototype
(arms drawing independent randomness) *structurally impossible*, because every arm sees
byte-identical `q` and `k`. It is also the cheapest design that answers the gate, and it
is what the Top-4 gaps note prescribes for stage one.

## Run

8 items (COCO val2017 photos, long-form prompts), 8 layers spread over depth
(0,5,10,15,20,25,30,35), up to 160 decode steps, 25% eviction budget.
**8,744 (layer, step) observations.** Greedy decoding, FP16, single seed.

## Results — KIVI-style per-channel block quantization (the realistic setting)

| Arm | Split | top-K overlap | **evict Jaccard** | Spearman | rank displacement |
|---|---|---|---|---|---|
| INT4 | visual | 0.9863 | **0.9210** | 0.9959 | 6.88 |
| INT4 | text   | 0.9876 | 0.9285 | 0.9964 | 1.99 |
| INT2 | visual | 0.9376 | **0.6869** | 0.9206 | 31.20 |
| INT2 | text   | 0.9461 | 0.7236 | 0.9406 | 8.73 |

### 1. The gate, as pre-registered, is NOT met at INT4

The rule fixed in advance was *visual top-K overlap below 0.95*. Measured INT4 value is
**0.9863** (bootstrap CI [0.9862, 0.9863]). INT2 gives 0.9376 and does pass.

### 2. But the gate metric looks like the wrong one, and this is the main finding

Eviction deletes from the **bottom** of the ranking. Top-K overlap measures the **top**,
which is dominated by stable heavy hitters and is therefore a conservative proxy.

Measured on the set eviction actually acts on:

- **INT4 already changes 7.9% of the visual eviction set** (Jaccard 0.9210), while top-K overlap says 0.9863.
- **INT2 changes 31.3% of it** (Jaccard 0.6869).

So at INT4 the ranking *is* being reordered where it matters, and the pre-registered
metric hides it. Recommendation: **re-specify the gate on bottom-k (eviction-set)
agreement, not top-K overlap** — and say plainly that the threshold was reconsidered
*because the original metric measures the wrong end of the ranking*, not because INT4
failed it.

### 3. Clean dose response

Rank displacement on visual tokens: 6.88 (INT4) → 31.20 (INT2), a 4.5× increase.
Eviction-set Jaccard 0.9210 → 0.6869. An effect that grows as precision falls is much
harder to dismiss as noise than a single difference.

### 4. Visual tokens are disrupted more than text — the modality claim holds

At INT2, rank displacement is **31.20 on visual vs 8.73 on text — 3.6×**.
Eviction-set agreement is also worse for visual (0.6869 vs 0.7236).

### 5. The Jensen bias correction recovers ~0% — on a real model

| Arm | Split | raw | corrected | gap recovered |
|---|---|---|---|---|
| INT4 | visual | 0.9863 | 0.9863 | −0.1% |
| INT2 | visual | 0.9376 | 0.9367 | −1.4% |

This **reproduces the Stage 0 synthetic finding on a real VLM**. The systematic Jensen
bias is real but far too small to flip rankings: σ² is 0.063% of logit scale at INT4 and
1.58% at INT2. The reordering is **variance, not bias**. Stage 4 targeting score variance
rather than porting the published correction is now supported by real-model evidence.

### 6. Correction to a number in the methodology note

The team note states the bias is "5.28× larger on visual than text tokens".
Measured here it is the **opposite direction**: σ²_visual / σ²_text = **0.66×** — visual
tokens carry a *smaller* predicted bias than text, consistently at both bit widths.
Visual tokens are still reordered more, but **not because the Jensen bias is larger on
them.** This should be corrected before the review; the current slide claim is wrong in
sign and a panelist checking it would find that.

## Instrument validation

`per_channel_global` (one step per channel over the whole cache) is the sanity check:
the Jensen term is constant across tokens there, so it cannot reorder anything. Measured:
σ²_visual / σ²_text = **exactly 1.00×**, correction recovers **exactly 0.0%**. The
instrument behaves as theory requires.

The self-check (`python3 visretain.py`) asserts four properties: placebo (same cache
scored twice agrees at exactly 1.000), quantization round-trip bounded by half a step,
dose response monotone in bit width, and Jensen rank-neutrality under a global step.

## A measurement bug found and fixed

The first run reported `text evict_jaccard = 1.0000` everywhere. Cause: a single global
budget (98 tokens) exceeded the text-token count (~48), so *every* text token was in the
bottom-k of both arms and Jaccard was trivially 1.0. Fixed by scaling the budget to each
split. Worth mentioning in the review — it is the kind of control the panel should see
being caught.

## Honest limitations

- One model, one seed, 8 items, greedy decoding. No statistical test across items yet
  (Wilcoxon/McNemar need more items; the CIs here are over observations, not items).
- COCO photographs with long-form prompts, **not GQA** — so there is no ground-truth
  evidence-region mapping yet. That is Gap 2 and the harder engineering problem
  (box-to-token mapping under Qwen's dynamic resolution).
- Shadow ranking means nothing is actually evicted, so there is **no accuracy number**
  and no deferred-token loss yet. This measures *ranking disagreement*, which is the
  premise the project rests on — not yet its task consequence.
- Fake (simulated) quantization, not a real low-bit kernel. Standard for this kind of
  study, but it is simulation.

## What this establishes for Review 2

The premise holds: **KV-cache precision does change which visual tokens eviction would
delete, it hits visual tokens harder than text, and the effect is not explained by the
published bias correction.** That is the whole foundation of the project, now measured on
a real VLM rather than a synthetic harness.

---

# Update — accumulated (H2O/EMA) scoring, the decisive test

The result above used **instantaneous** single-query scoring, which is TOVA-like.
H2O and EMA rank by attention **summed over decode steps**. Since our mechanism is
*variance*, and accumulation averages variance away, this was the single biggest threat
to the premise — your own reference 2609.03515 reports that EMA aggregation makes
eviction rankings largely insensitive to scorer changes.

Measured over 8 items, 8 layers, 160 decode steps, decode-only accumulation
(no prefill attention, which is computed on full-precision keys and would mask the effect).
**Statistics are per item (n = 8), not per observation.**

| Arm | Split | evict Jaccard | 95% CI | rank disp | top-K overlap |
|---|---|---|---|---|---|
| INT4 | visual | **0.9123** | [0.909, 0.917] | 7.49 | 0.9846 |
| INT4 | text   | 0.9404 | [0.928, 0.948] | 3.17 | 0.9897 |
| INT2 | visual | **0.6650** | [0.655, 0.678] | 33.38 | 0.9318 |
| INT2 | text   | 0.7645 | [0.732, 0.786] | 13.96 | 0.9543 |

## The effect survives accumulation

INT4 visual eviction-set agreement is **0.9123 accumulated** vs 0.9210 instantaneous;
INT2 is **0.6650** vs 0.6869. Accumulation did **not** wash the effect out — it is
marginally *stronger*. The main threat to the premise is defeated.

- **8.8% of the visual eviction set changes at INT4**, Wilcoxon p = 0.0039.
- **33.5% changes at INT2**, p = 0.0039.

p = 0.0039 is 2⁻⁸, the floor for n = 8 one-sided: **all eight items moved the same way.**
Read it as perfect directional consistency, not as a precise probability.

## Visual is hit harder than text — now with a paired test

| Arm | text − visual Jaccard | 95% CI | p |
|---|---|---|---|
| INT4 | +0.0281 | [+0.014, +0.038] | 0.0156 |
| INT2 | +0.0995 | [+0.064, +0.127] | 0.0156 |

Visual tokens are reordered significantly more than text, on the same items.

## Jensen correction, third confirmation

Recovers **−0.2%** (INT4) and **−1.3%** (INT2) of the gap — nothing, on real-model
accumulated scores. The variance pivot is now supported by synthetic data,
instantaneous real scores, and accumulated real scores.

## Gate verdict depends entirely on the metric

| Gate | INT4 visual | Verdict |
|---|---|---|
| top-K overlap < 0.95 (as written) | 0.9846 | **FAILS** |
| eviction-set disagreement > 0, paired | 8.8%, p = 0.0039 | **PASSES** |

INT2 passes both (top-K 0.9318).

## A protocol inconsistency to fix

The gate specifies "across three seeds", but decoding is **greedy and deterministic** —
there are no seeds to vary. Seed variation has to come from somewhere else (item
subsets, or sampled decoding), or the clause should be replaced with a paired test
across items, which is what was actually run here.

## Still missing (unchanged)

No real eviction, no eviction timeline, no accuracy, no B3/B4/B5 against the real model,
no GQA evidence regions, no deferred-importance measurement.
