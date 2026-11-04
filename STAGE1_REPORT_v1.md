# Stage 1 batch report

Run 2026-11-04 01:25 on Qwen2.5-VL-3B-Instruct (local, MPS). Command: `python3 test_stage1.py`. Pass criteria: STAGES.md, stage 1.

**13 images; every check passed on 3 of 13.**

| Check | Images passing |
|---|---|
| 1.1 Our resize rule matches Qwen's processor | 13 / 13 |
| 1.2 Image rebuilt from the model's input numbers matches the resized image | 13 / 13 |
| 1.3 Number of image tokens = rows × columns | 13 / 13 |
| 1.4 Qwen's own positions put token k at row k ÷ W, column k mod W | 13 / 13 |
| 1.5 Changing one square changes that square's token the most | 3 / 13 |
| 1.6 Token count stays under the configured budget | 13 / 13 |
| 1.7 Encoder output k is what the language model receives in image slot k | 13 / 13 |

## Causal test (1.5) margins across all images

294 squares tested in total. Margin = response of the changed square's token ÷ the next-largest response.

- minimum nan×, 5th percentile nan×, median nan×, maximum nan×
- squares with margin below 1.10× (near-ties): 34 of 294

## Per image

| Image | Size | Tokens | All checks | 1.5 min margin | Weakest square |
|---|---|---|---|---|---|
| pilot 139 | 640 × 426 | 345 (23 × 15) | **FAIL** | 0.68× | row 2, col 4 |
| pilot 285 | 586 × 640 | 483 (21 × 23) | **FAIL** | 0.94× | row 11, col 17 |
| pilot 632 | 640 × 483 | 391 (23 × 17) | **FAIL** | 0.42× | row 16, col 18 |
| pilot 724 | 375 × 500 | 234 (13 × 18) | **FAIL** | 0.78× | row 4, col 3 |
| pilot 776 | 428 × 640 | 345 (15 × 23) | PASS | 1.20× | row 13, col 12 |
| pilot 785 | 640 × 425 | 345 (23 × 15) | PASS | 1.18× | row 13, col 17 |
| pilot 802 | 424 × 640 | 345 (15 × 23) | **FAIL** | 0.37× | row 13, col 12 |
| pilot 872 | 621 × 640 | 506 (22 × 23) | **FAIL** | 0.96× | row 18, col 5 |
| pilot 885 | 640 × 427 | 345 (23 × 15) | **FAIL** | 0.98× | row 10, col 14 |
| pilot 39769 | 640 × 480 | 391 (23 × 17) | PASS | 1.09× | row 11, col 5 |
| large photo (shrink branch) | 3840 × 2550 | 486 (27 × 18) | **FAIL** | 0.84× | row 16, col 23 |
| tiny image (enlarge branch) | 40 × 30 | 6 (3 × 2) | **FAIL** | 0.89× | row 1, col 2 |
| panorama 8:1 (shape + shrink) | 2400 × 300 | 441 (63 × 7) | **FAIL** | nan× | row 6, col 54 |

## Failures

- **pilot 139**, check 1.5: 23 / 24 random squares. The changed square's token responded 0.68× to 4.11× as much as the next-largest (median 1.69×); weakest: row 2, column 4 (0.68×). The runner-up was an immediate neighbour in 17 of 24.
- **pilot 285**, check 1.5: 23 / 24 random squares. The changed square's token responded 0.94× to 2.63× as much as the next-largest (median 1.74×); weakest: row 11, column 17 (0.94×). The runner-up was an immediate neighbour in 18 of 24.
- **pilot 632**, check 1.5: 19 / 24 random squares. The changed square's token responded 0.42× to 4.39× as much as the next-largest (median 1.40×); weakest: row 16, column 18 (0.42×). The runner-up was an immediate neighbour in 21 of 24.
- **pilot 724**, check 1.5: 22 / 24 random squares. The changed square's token responded 0.78× to 2.83× as much as the next-largest (median 1.70×); weakest: row 4, column 3 (0.78×). The runner-up was an immediate neighbour in 17 of 24.
- **pilot 802**, check 1.5: 21 / 24 random squares. The changed square's token responded 0.37× to 2.85× as much as the next-largest (median 1.45×); weakest: row 13, column 12 (0.37×). The runner-up was an immediate neighbour in 19 of 24.
- **pilot 872**, check 1.5: 23 / 24 random squares. The changed square's token responded 0.96× to 2.19× as much as the next-largest (median 1.62×); weakest: row 18, column 5 (0.96×). The runner-up was an immediate neighbour in 19 of 24.
- **pilot 885**, check 1.5: 23 / 24 random squares. The changed square's token responded 0.98× to 3.01× as much as the next-largest (median 2.02×); weakest: row 10, column 14 (0.98×). The runner-up was an immediate neighbour in 16 of 24.
- **large photo (shrink branch)**, check 1.5: 22 / 24 random squares. The changed square's token responded 0.84× to 2.49× as much as the next-largest (median 1.58×); weakest: row 16, column 23 (0.84×). The runner-up was an immediate neighbour in 24 of 24.
- **tiny image (enlarge branch)**, check 1.5: 4 / 6 random squares. The changed square's token responded 0.89× to 1.26× as much as the next-largest (median 1.04×); weakest: row 1, column 2 (0.89×). The runner-up was an immediate neighbour in 3 of 6.
- **panorama 8:1 (shape + shrink)**, check 1.5: 5 / 24 random squares. The changed square's token responded nan× to nan× as much as the next-largest (median nan×); weakest: row 6, column 54 (nan×). The runner-up was an immediate neighbour in 3 of 24.
