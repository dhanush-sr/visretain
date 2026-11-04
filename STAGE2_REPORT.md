# Stage 2 batch report

Run 2026-11-04 09:25 on Qwen2.5-VL-3B-Instruct (local, MPS). Command: `python3 test_stage2.py`. Pass criteria: STAGES.md, stage 2.

**13 images; every check passed on 13 of 13.**

| Check | Images passing |
|---|---|
| 2.1 The keys we read are exactly the keys the KV cache stores | 13 / 13 |
| 2.2 Our attention equals Qwen's own reference formula | 13 / 13 |
| 2.3 Our attention reproduces the model's actual attention output | 13 / 13 |
| 2.N The checks can fail: deliberately broken versions are caught | 13 / 13 |
| 2.4 Every attention row is a valid probability distribution | 13 / 13 |
| 2.5 Watching attention does not change the model's answer | 13 / 13 |

## Per image

| Image | Answer tokens | 2.1 | 2.2 | 2.3 | 2.N | 2.4 | 2.5 |
|---|---|---|---|---|---|---|---|
| pilot 139 | 60 | PASS | PASS | PASS | PASS | PASS | PASS |
| pilot 285 | 60 | PASS | PASS | PASS | PASS | PASS | PASS |
| pilot 632 | 60 | PASS | PASS | PASS | PASS | PASS | PASS |
| pilot 724 | 60 | PASS | PASS | PASS | PASS | PASS | PASS |
| pilot 776 | 60 | PASS | PASS | PASS | PASS | PASS | PASS |
| pilot 785 | 60 | PASS | PASS | PASS | PASS | PASS | PASS |
| pilot 802 | 15 | PASS | PASS | PASS | PASS | PASS | PASS |
| pilot 872 | 60 | PASS | PASS | PASS | PASS | PASS | PASS |
| pilot 885 | 60 | PASS | PASS | PASS | PASS | PASS | PASS |
| pilot 39769 | 60 | PASS | PASS | PASS | PASS | PASS | PASS |
| large photo (shrink branch) | 60 | PASS | PASS | PASS | PASS | PASS | PASS |
| tiny image (enlarge branch) | 58 | PASS | PASS | PASS | PASS | PASS | PASS |
| panorama 8:1 (shape + shrink) | 60 | PASS | PASS | PASS | PASS | PASS | PASS |

## Evidence per check (first image)

- **2.1**: bit-identical for all 144 (layer, step) pairs checked: 36 layers × steps [1, 10, 30, 59]
- **2.2**: max difference 0.0e+00 (pass < 1e-5) across 144 (layer, step) pairs
- **2.3**: max difference 0.047% of the output's largest value (pass ≤ 1%) across 144 (layer, step) pairs; the model computes in fp16, we in fp32
- **2.N**: N1 wrong head mapping: caught (2.2 diff 1.0e+00, 2.3 diff 306.2%); N2 scaling × 1.1: caught (2.2 diff 1.3e-01, 2.3 diff 32.4%); N3 newest key missing: caught (2.2 diff 1.0e+00, 2.3 diff 141.1%)
- **2.4**: rows sum to 1 within 1.3e-06; no NaN/inf
- **2.5**: 60 tokens, identical to the answer with plain attention

## Where attention goes (measured, averaged over images)

| Layer | First token | Prompt text | Image | Answer so far |
|---|---|---|---|---|
| 0 | 0% | 19% | 2% | 79% |
| 5 | 79% | 2% | 1% | 18% |
| 10 | 57% | 7% | 5% | 30% |
| 15 | 46% | 5% | 4% | 44% |
| 20 | 52% | 10% | 9% | 29% |
| 25 | 59% | 5% | 6% | 31% |
| 30 | 55% | 6% | 7% | 32% |
| 35 | 0% | 44% | 6% | 50% |
