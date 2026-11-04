# VisRetain — stage-by-stage build log

Method for every stage: **research → decide and write down the pass criteria → build →
verify → show**. The decision and the checks below were written *before* the code that
tests them. Results are filled in afterwards, pass or fail, and never edited to fit.

All settings live in `config.json`; nothing below is hard-coded in the scripts.

---

## Stage 1 — Image → visual tokens

### Question
When Qwen2.5-VL-3B reads an image, how exactly does it become the visual tokens the
language model stores in its KV cache, and which square of the image does each token
stand for? Every later stage draws on this mapping, so if it is wrong, every square in
every visual is in the wrong place.

### Research (from Qwen's own source, transformers 5.11)
1. **Resize** (`image_processing_qwen2_vl.py`, `smart_resize`): both sides rounded to a
   multiple of **28 px** (patch 14 × merge 2), aspect ratio kept, total pixels clamped to
   [`min_pixels`, `max_pixels`]. Default `max_pixels` is 12,845,056, so a 12 MP phone
   photo would become ~16,000 visual tokens. Bicubic resampling; normalised with the CLIP
   mean and std.
2. **Patches**: the resized image is cut into **14 × 14 px patches**. `pixel_values` stores
   them **in groups of four**: each 2 × 2 block of patches is consecutive, and blocks run
   row-major (`permute(0, 2, 5, 3, 6, 1, 4, 7)`). Each row is 3 colours × 2 frames (a still
   image is treated as a 2-frame clip by repeating it) × 14 × 14 = **1176 numbers**.
3. **Vision encoder** (`modeling_qwen2_5_vl.py`, vision `forward`): reorders patches into
   attention windows (`window_index`), runs 32 blocks (most windowed, a few full), merges
   each 2 × 2 block into one token (`merger`), then **undoes the reordering**
   (`reverse_indices = argsort(window_index)`).
4. **Position in the language model** (`get_vision_position_ids`): each image token gets a
   3D rotary position (time, row, column) with row = k ÷ W and column = k mod W, i.e.
   **row-major over the merged grid**.

### Prototype findings (real model, before building the checks)
- Rebuilding the image from `pixel_values` with the layout in research item 2 matches the
  processor's resize (torchvision bicubic, antialiased) to **6e-8** max abs difference.
- The two frames of each patch row are **identical** (the still image is duplicated).
- Vision encoder: **32 blocks**, attention windows of **112 px** (4 x 4 tokens) except blocks
  **7, 15, 23, 31**, which attend over the whole image. Output: one **2048-number** vector
  per token (345 tokens for a 640 x 426 photo).
- **Qwen's documented `max_pixels=` argument is silently ignored** in transformers 5.11: a
  4000 x 3000 photo still became 15,301 tokens. Only `size={"shortest_edge",
  "longest_edge"}` takes effect. Check 1.1 would have caught this.

### Decisions
- **D1.1 Token k is the 28 × 28 px square at row k ÷ W, column k mod W** of the resized
  image (W = width ÷ 28). Follows from research items 2–4. To be *verified*, not assumed.
- **D1.2 Cap image size with Qwen's own pixel budget**, not a custom resize: it is the
  model's own control and is how Qwen is normally deployed under a budget. Passed as
  `size={"shortest_edge": min_pixels, "longest_edge": max_pixels}` because the documented
  `max_pixels=` argument is ignored (finding above). Value in `config.json`, chosen so every
  image has at most ~512 visual tokens, the scale of the pilot images.
- **D1.3 How the causal test changes a square**: invert its colours (255 − value). Chosen
  because it is deterministic and changes every pixel substantially (only exact mid-grey
  is unchanged), unlike a solid fill that can match the square's existing colour. It is
  applied to the image *as Qwen already resized it*, so no further resizing can blur it
  into neighbouring squares. The response of a token is the relative change of its output
  vector, ‖e′ₖ − eₖ‖ / mean‖e‖.
- **D1.4 Images tested in batch**: the 10 pilot photos, plus three edge cases chosen to
  exercise each branch of the resize rule — a large photo (shrunk to the budget), a tiny
  image (enlarged to the minimum), and an extreme panorama.

### Checks (pass criteria fixed in advance)
| # | Check | Pass if |
|---|---|---|
| 1.1 | Geometry: our predicted resize and grid equal what the processor produces | exact match |
| 1.2 | Patch layout: rebuild the image from `pixel_values` using the layout in research item 2 | matches the resized image to within float error (max abs diff < 1e-4 after un-normalising) |
| 1.3 | Token count: image tokens in the prompt = rows × cols of the merged grid | exact match |
| 1.4 | Positions: Qwen's own 3D positions for the image tokens equal row-major (k ÷ W, k mod W) | exact match for every token |
| 1.5 | **Causal test**: invert one 28 × 28 square of the image, run the real vision encoder; the output token that changes most must be the token D1.1 assigns to that square | correct for **every** square tested (the configured number of random squares per image, on every batch image) |
| 1.6 | Size cap: a very large photo stays under the configured token ceiling | n tokens ≤ ceiling |
| 1.7 | *Added after review (see below).* The k-th image slot in the language model's input receives vision-encoder output k | every slot's input vector equals encoder output k (max abs diff < 1e-3 in fp16) |

Check 1.5 is the one that matters most: 1.1–1.4 confirm we read the code correctly; 1.5
confirms the model actually behaves that way.

**Why 1.7 was added.** Reviewing the proof chain after the first run: 1.5 proves square k →
encoder output k, and 1.4 proves prompt slot k → position (row k ÷ W, column k mod W). Nothing
yet proved encoder output k is what lands in prompt slot k (Qwen fills the slots in order
with `masked_scatter`, but that is reading code). 1.7 closes the chain: square k → encoder
output k → prompt slot k → the KV-cache entry every later stage tracks. The criterion was
written before the check was built.

### Results — single image (development run)
First run on the skier photo (COCO 785, 640 × 425): **all six checks pass.**
- 1.1 predicted 644 × 420 = processor 644 × 420. 1.2 max difference 6e-8; frames identical.
  1.3 345 = 15 × 23. 1.4 all 345 tokens row-major. 1.6 345 ≤ 512.
- **1.5: 24 / 24 squares.** The margin over the next-largest response ranged 1.2× to 3.3×
  (median 1.6×). Investigated rather than accepted: the runner-up was an *immediate
  neighbour* in 19 of 24 tests, and the mean response falls off with distance —
  1.00 (the square) → 0.27 (1 token away) → 0.11 → 0.07 → ≤ 0.04 (4+ away). So the effect
  is local and centred on the right square; thin margins come from neighbours, which the
  encoder's local windows and four full-image layers are expected to mix in.
- Also observed: image tokens occupy positions starting at 15, but the next text token is
  at 38 = 15 + 23 = start + max(rows, cols), not start + 345. Qwen advances its position
  counter by the image's larger side, not its token count.

### Results — batch, 13 images (`reports/STAGE1_REPORT.md`)
| Check | Images passing |
|---|---|
| 1.1 resize rule | 13 / 13 |
| 1.2 image rebuilt from the model's numbers | 13 / 13 |
| 1.3 token count | 13 / 13 |
| 1.4 Qwen's own positions are row-major | 13 / 13 |
| **1.5 causal test (v1)** | **3 / 13 — FAILED** |
| 1.6 size budget | 13 / 13 |
| 1.7 encoder output k reaches prompt slot k | 13 / 13 |

**1.5 failed as pre-registered and is reported as a failure.** 257 of 294 squares (87%) had
their own token react most; the misses were near-ties (margins 0.37×–0.98×) where an
immediate neighbour's output changed more. The panorama additionally produced `nan`
values (a separate bug, investigated below).

**Diagnosis.**
- *Not* an artefact of the edit: re-running the processor on an already-sized image is an
  exact no-op, and inverting a square changes zero pixels outside it (checked).
- *Not* flat patches: the first failing square has colour std 50.
- What v1 actually demanded was two things at once: (a) **routing** — token k's input is
  square k's four patches and token k reaches slot k; and (b) **self-dominance** — token
  k's *output* depends more on its own square than on any neighbour's. VisRetain relies on
  (a). (b) is a property of how much the encoder mixes context, and Qwen's encoder mixes a
  lot (112 px windows plus four whole-image layers). The data shows (b) does not hold for
  ~13% of squares. That is a finding about the encoder, not a failure of the mapping:
  1.2, 1.4 and 1.7 prove the data path exactly on all 13 images.

**Consequence for VisRetain (worth keeping):** a visual token's cached vector is not a pure
description of its own square; it carries its neighbours too (mean response 0.27 one token
away). Evicting token k removes mostly, but not only, square k's information.

### Check 1.5 v2 (pre-registered after the v1 failure, before running)
Separate the two properties instead of testing them together:

| # | Check | Pass if |
|---|---|---|
| 1.5a | **Routing, mixing switched off.** Temporarily make every vision-encoder attention layer attend only to the token itself (all other computation unchanged). Invert square k: | output token k changes, and **every** other token changes by < 1e-3 of the mean token norm — on every tested square of every image |
| 1.5b | **Mixing, measured (descriptive, no pass/fail).** With the encoder unmodified: share of squares whose own token reacts most, the margin distribution, and response vs distance | reported, not judged |

1.5a is a strictly sharper test of the claim VisRetain needs: with mixing off, an error in
routing would show up as a *different* token changing, with nothing to hide it.

Prototype of 1.5a on the first failing image (before the batch): for 6 squares including the
v1 failure (2, 4), the changed square's token moved by ~0.4 and **every other token by exactly
0**. The swap was confirmed to take effect and to be fully undone afterwards.

### The panorama `nan`: fp16 overflow in the vision encoder
- 18 of 24 *edited* panoramas produced NaN in at least one output token (the unedited one did
  not). The same input in fp32 on CPU gives no NaN.
- Largest activation per vision block (fp32): tens in early blocks, ~7,400 from block 17
  (the "massive activations" phenomenon), **67,070 in the last block — above fp16's maximum
  of 65,504**. In fp16 it overflows to infinity, which becomes NaN.
- **We load the model in fp16.** No pilot run overflowed (1.7 confirms finite, exact slot
  values on all 13 images), so existing results stand, but the vision encoder runs close to
  the fp16 ceiling. Qwen is normally run in bfloat16, which has fp32's range. Switching would
  change the instrument and require re-running everything: **a team decision, recorded as
  open.** Until then, check 1.8 guards every run.

| # | Check (pre-registered after the diagnosis, before running) | Pass if |
|---|---|---|
| 1.8 | The vision encoder's outputs for the image are all finite; the last block's largest activation is reported against the fp16 limit | no NaN / inf |

Squares whose *edited* image overflows are reported as failures of 1.5a, not dropped.

### Results — batch v2, 13 images (`reports/STAGE1_REPORT.md`; v1 kept as `reports/STAGE1_REPORT_v1.md`)
**Every pass/fail check passed on all 13 images.**

| Check | Images passing |
|---|---|
| 1.1 resize rule | 13 / 13 |
| 1.2 image rebuilt from the model's numbers (max diff ~6e-8) | 13 / 13 |
| 1.3 token count | 13 / 13 |
| 1.4 Qwen's own positions are row-major | 13 / 13 |
| **1.5a routing, mixing off** | **13 / 13** |
| 1.6 size budget | 13 / 13 |
| 1.7 encoder output k reaches prompt slot k (diff 0) | 13 / 13 |
| 1.8 encoder output finite | 13 / 13 |

- **1.5a**: 294 squares over 13 images. Every time, only the predicted token changed; the
  largest change of any *other* token was **exactly 0.0** (criterion < 1e-3). No edit
  overflowed with mixing off.
- **1.5b (measured)**: 276 squares with finite output (18 edited panoramas overflowed fp16,
  excluded and reported). Margin of the changed square's own token over the next-largest:
  min 0.37×, 5th percentile 0.93×, median 1.67×, max 4.39×; own token reacted most in
  ~88% of squares; the winner otherwise was nearly always an immediate neighbour. The
  values are identical to v1 for the same squares (a determinism check for free).

**Stage 1 conclusion.** We know exactly which square of the image each cached visual token
stands for, from the resize through to the language model's input and positions, on every
image shape tested. One finding carries forward: each token's cached vector also carries
its neighbours' information (the encoder mixes context), so evicting a token removes mostly,
but not only, its own square. And one open decision: fp16 vs bfloat16 for the vision encoder.

---

## Stage 2 — Attention: is what we capture what the model computes?

### Question
VisRetain decides what to evict from attention scores that **we** compute from the model's
queries and keys. If our computation differs from the model's — wrong scaling, wrong mapping
of the 16 query heads onto the 2 key heads, keys that are not the ones the cache stores —
every eviction decision downstream is wrong. Stage 2 proves the attention we capture is the
attention the model actually uses.

### Research (Qwen's own source, transformers 5.11)
1. **Language-model attention** (`Qwen2_5_VLAttention.forward`): q, k, v projections; 3D
   rotary positions applied to q and k; then `past_key_values.update(...)` appends the new key
   and returns the **whole cache**, which is what the attention function receives. So the keys
   our hook sees at a decode step should be exactly the cache's stored keys.
2. **Heads**: 16 query heads, 2 key/value heads (grouped-query attention), head size 128,
   `scaling = 128^-0.5`, passed in by the model. Sliding window off.
3. **Reference implementation** (`eager_attention_forward`): `softmax(q·kᵀ × scaling + mask)`,
   with the dot products in the model's dtype (fp16) and the softmax in fp32, then
   `weights · v`. Key heads are expanded with `repeat_kv`, documented as equal to
   `repeat_interleave` on the head axis (what our hook uses).
4. The model normally runs SDPA (a fused kernel), not the eager code; our hook passes the real
   computation through to SDPA untouched and computes scores on the side.

### Decisions
- **D2.1 Our scores use the model's own `scaling`** (read from the attention module), not a
  re-derived 1/√d: one source of truth.
- **D2.2 Scores are computed in fp32** from the fp16 queries and the fp16-stored keys. The
  keys are exactly what an FP16 cache holds; fp32 only removes arithmetic rounding from *our*
  measurement. Consequence: comparisons with the model's fp16 output use an fp16 tolerance.
- **D2.3 Checks run on every one of the 36 layers**, at decode steps spread over the answer.

### Checks (pass criteria fixed in advance)
| # | Check | Pass if |
|---|---|---|
| 2.1 | The keys our hook sees at a decode step are the KV cache's stored keys for that layer | identical (bit-for-bit) |
| 2.2 | Our attention weights equal Qwen's reference formula (`eager_attention_forward`, its `repeat_kv`, its `scaling`) on the same inputs in fp32 | max abs diff < 1e-5, every layer |
| 2.3 | Our weights applied to the values reproduce the model's **actual** attention output (SDPA, fp16) | max abs diff ≤ 1% of the output's largest value, every layer |
| 2.4 | Every attention row is a valid distribution | sums to 1 within 1e-4; no NaN or inf |
| 2.5 | Capturing attention does not change the model's answer | token-for-token identical to vanilla attention |

2.3 is the one that matters most: 2.2 shows we match the *formula*; 2.3 shows we match what
the model actually *computed*.

### First run (skier, 60-token answer) — all five pass
2.1 bit-identical on 144 (layer, step) pairs · 2.2 max diff **exactly 0.0** · 2.3 max diff
0.048% of the output's scale · 2.4 rows sum to 1 within 1.1e-6 · 2.5 identical answer.

### Negative controls (added after the first run, before running them)
A difference of exactly 0.0 could mean the code is right, or that the check compares
something with itself and cannot fail. A check is only evidence if it can fail. So the same
checks are run on deliberately broken versions of our computation, each of which must FAIL:

| Control | Deliberate error | Must fail |
|---|---|---|
| N1 | wrong head mapping: query head h uses key head h mod 2 instead of h ÷ 8 | 2.2 and 2.3 |
| N2 | wrong scaling: scores × 1.1 | 2.2 and 2.3 |
| N3 | stale keys: score against the cache minus its newest entry | 2.2 and 2.3 |

If any control passes, the corresponding check has no power and its result means nothing.


### Results — batch, 13 images (`reports/STAGE2_REPORT.md`)
**All six checks pass on all 13 images** (the 10 pilot photos and the three edge cases).

| Check | Images passing |
|---|---|
| 2.1 keys = the KV cache's stored keys (bit-identical) | 13 / 13 |
| 2.2 our attention = Qwen's reference formula (diff exactly 0) | 13 / 13 |
| 2.3 reproduces the model's real attention output (≤ 0.05% of scale) | 13 / 13 |
| 2.N deliberately broken versions are caught | 13 / 13 |
| 2.4 valid probability distributions | 13 / 13 |
| 2.5 watching does not change the answer | 13 / 13 |

The negative controls fail by wide margins (wrong head mapping: attention off by up to 1.0,
output off ~250–300%; scaling × 1.1: off 0.13 / ~33%; newest key missing: off 1.0 /
~140–160%), so the exact 0.0 in 2.2 is genuine agreement, not a check that cannot fail.

### What the stage 2 view shows (measured, not judged)
- **Per word**: where the model looked while choosing it. On the skier, while choosing
  "jacket" (layer 15), the most-attended patch is on the jacket — and that patch's attention
  stays near zero through the answer and **spikes exactly at the word "jacket"**.
- **Caution**: attention averaged over heads is noisy; a good share also lands on background
  patches. Present it as "the strongest attention lands on the object being named", not "it
  looks only at the object" (consistent with CVMA's finding that attention is an imperfect
  signal of what is needed).
- **Attention sink**: in layers ~3–30 most attention goes to the very first token. The photo
  receives its largest share in the later-middle layers (~17–33).

**Stage 2 conclusion.** The attention VisRetain computes, and will later recompute after
rounding, is exactly the attention the model computes, on every image tested. Stage 3 adds
FP16 / INT4 / INT2 side by side in this same view.