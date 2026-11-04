# Stage 1 batch report

Run 2026-11-04 07:52 on Qwen2.5-VL-3B-Instruct (local, MPS). Command: `python3 test_stage1.py`. Pass criteria: STAGES.md, stage 1.

**13 images; every pass/fail check passed on 13 of 13.**

| Check | Images passing |
|---|---|
| 1.1 Our resize rule matches Qwen's processor | 13 / 13 |
| 1.2 Image rebuilt from the model's input numbers matches the resized image | 13 / 13 |
| 1.3 Number of image tokens = rows × columns | 13 / 13 |
| 1.4 Qwen's own positions put token k at row k ÷ W, column k mod W | 13 / 13 |
| 1.5a Routing: with the encoder's mixing switched off, changing square k changes token k and nothing else | 13 / 13 |
| 1.6 Token count stays under the configured budget | 13 / 13 |
| 1.7 Encoder output k is what the language model receives in image slot k | 13 / 13 |
| 1.8 The vision encoder's output is finite (no fp16 overflow) | 13 / 13 |

## 1.5a routing (mixing off)

294 squares. Largest change of any token OTHER than the changed square's, relative to the mean token size: max 0.0e+00 (pass < 1e-3).

## 1.5b mixing (encoder unmodified) — measured, not judged

276 squares with finite outputs. Margin = response of the changed square's token ÷ the next-largest response.

- minimum 0.37×, 5th percentile 0.93×, median 1.67×, maximum 4.39×
- squares with margin below 1.10× (near-ties): 34 of 276

## Per image

| Image | Size | Tokens | Pass/fail checks | 1.5b min margin | Weakest square |
|---|---|---|---|---|---|
| pilot 139 | 640 × 426 | 345 (23 × 15) | PASS | 0.68× | row 2, col 4 |
| pilot 285 | 586 × 640 | 483 (21 × 23) | PASS | 0.94× | row 11, col 17 |
| pilot 632 | 640 × 483 | 391 (23 × 17) | PASS | 0.42× | row 16, col 18 |
| pilot 724 | 375 × 500 | 234 (13 × 18) | PASS | 0.78× | row 4, col 3 |
| pilot 776 | 428 × 640 | 345 (15 × 23) | PASS | 1.20× | row 13, col 12 |
| pilot 785 | 640 × 425 | 345 (23 × 15) | PASS | 1.18× | row 13, col 17 |
| pilot 802 | 424 × 640 | 345 (15 × 23) | PASS | 0.37× | row 13, col 12 |
| pilot 872 | 621 × 640 | 506 (22 × 23) | PASS | 0.96× | row 18, col 5 |
| pilot 885 | 640 × 427 | 345 (23 × 15) | PASS | 0.98× | row 10, col 14 |
| pilot 39769 | 640 × 480 | 391 (23 × 17) | PASS | 1.09× | row 11, col 5 |
| large photo (shrink branch) | 3840 × 2550 | 486 (27 × 18) | PASS | 0.84× | row 16, col 23 |
| tiny image (enlarge branch) | 40 × 30 | 6 (3 × 2) | PASS | 0.89× | row 1, col 2 |
| panorama 8:1 (shape + shrink) | 2400 × 300 | 441 (63 × 7) | PASS | 0.92× | row 6, col 23 |
