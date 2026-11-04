# VisRetain test report

Run 2026-11-04 08:10 on the real Qwen2.5-VL-3B-Instruct, local (Apple M-series, MPS). Command: `python3 test_pipeline.py`.

**14 / 14 checks passed.**

| Check | Result | Detail |
|---|---|---|
| T1 non-interference | PASS | answer generated with logging on is token-for-token identical to vanilla attention (80 tokens; identical=True) |
| T2 determinism | PASS | two runs, same input: identical eviction timelines across 8 layers x 3 arms (8280 token deaths compared) |
| T3 live == batch | PASS | with the live streamer attached, record identical to batch run (identical=True) |
| T3b stream reproduces timeline | PASS | 80 streamed steps for 80 generated tokens; replaying the stream gives the final layer-15 timeline |
| T3c one event per token | PASS | 80 step events vs 80 tokens |
| T4 image scale cap | PASS | 4000x3000 photo -> 494 visual tokens (cap ~512; uncapped it would be ~15,000); grid [19, 26] |
| T4b short answer flagged | PASS | 20-token answer flagged too_short=True |
| T5 record integrity | PASS | budgets respected, death steps in range, evidence sets well-formed |
| T6a rejects non-image | PASS | HTTP 400: could not read that image (UnidentifiedImageError) |
| T6b rejects tiny image | PASS | HTTP 400: image is too small (need at least 56 px per side) |
| T6c starts a run | PASS | HTTP 200 |
| T6d rejects a second concurrent run | PASS | HTTP 409: a run is already in progress, wait for it to finish |
| T6e full live run over HTTP | PASS | events: 1 setup, 80 steps, done in 14s; saved results_live/item_1104_081021.json |
| T6f empty question gets a default | PASS | question used: 'Describe this image.' |

## Browser checks (live app, real Chrome)

**10 / 10 passed.** Screenshots in `demo/screens/`.

| Check | Result | Detail |
|---|---|---|
| B1 live card shown | PASS | upload card visible when served by app.py |
| B2 pilot images present | PASS | 8 images in the picker |
| B3 playback | PASS | Play advanced to step 14 |
| B4 live updates arrive | PASS | mid-run: step 48, 48 tokens shown, evicted so far per arm [155, 155, 158] |
| B5 evidence withheld until the end | PASS | evidence counter mid-run reads 'after the answer' |
| B6 switching images mid-run is safe | PASS | live item kept updating while another image was shown (step 48 -> 72); pilot image untouched (160 tokens) |
| B7 evidence revealed at the end | PASS | FP16 evidence counter '6 / 34'; layer picker has 8 layers; status: Done in 62.3s. The outlined tokens are the evidence the answer relied on late; r |
| B8 charts drawn | PASS | 3 survival lines drawn |
| B9 results tab | PASS | results tab shows the FP16 evidence-loss figure computed from the data (80.6%) |
| B10 no JavaScript errors | PASS | none |
