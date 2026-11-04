# VisRetain — project context

Read this first if you are new to the project. It explains the problem, the vocabulary, where
the project sits in the literature, how we work, what has been found so far, and what has been
decided. Detailed records are in [`STAGES.md`](STAGES.md) (stage-by-stage log),
[`PLAN.md`](PLAN.md) and [`FINDINGS.md`](FINDINGS.md).

## 1. The problem

A vision-language model (VLM) reads an image as a grid of **visual tokens** (one per 28 × 28 px
square for Qwen2.5-VL) and writes its answer one token at a time. For every new token it re-reads
everything it has stored: the **KV cache** (one key and one value vector per token, per layer).
For long or high-resolution inputs the cache does not fit in memory, so deployments:

1. **quantize** the cache — store keys and values in 8, 4 or 2 bits instead of 16; and
2. **evict** from the cache — delete the entries the model has paid least attention to, to stay
   within a fixed budget.

The field mostly treats these as independent ("orthogonal"). They are not: eviction ranks entries
by attention scores, and quantization perturbs exactly those scores.

**Deferred importance** (from RetentiveKV, Findings of ACL 2026): a part of the image can look
unimportant early in an answer and be essential later — e.g. a gun in someone's hand that only
matters when the answer reaches "what is the person holding". If it was evicted early, the model
can no longer see it.

**Research question.** At a fixed memory budget, does quantizing the KV cache change *which*
visual evidence is evicted and *when*, and does that make the model lose evidence it needs later?

## 2. Vocabulary

| Term | Meaning here |
|---|---|
| visual token | one 28 × 28 px square of the resized image (2 × 2 merged 14 px patches) |
| KV cache | the model's working memory: key and value vectors for every token so far |
| attention | how much the token being written "looks at" each stored token (softmax of q·k / √d) |
| eviction | deleting cache entries to stay within a budget; here H2O-style (lowest accumulated attention), most recent 32 tokens protected |
| FP16 / INT4 / INT2 | cache precision: 16-bit, or keys rounded to 4 or 2 bits (KIVI-style, per channel in blocks of 32 tokens) |
| arm | one experimental condition, e.g. "limited memory, INT4" |
| deferred-important evidence | visual tokens attended little early and a lot late in an answer |
| counterfactual eviction | each arm records what it *would* evict while the model answers from the full cache (all arms see identical inputs) |
| real eviction | the model is actually prevented from attending to evicted entries, so its answer can change |

## 3. Where this sits in the literature

- **Quantization changes token selection — in text:** Q-Hitter (MLSys 2024), DGAP.
- **Visual KV tolerates low bits:** VidKV, FIER; **CVMA** (2607.25467) finds visual KV near-lossless
  down to ~3 bits and that current attention is a poor predictor of future need (across dialog turns).
- **Deferred visual importance:** RetentiveKV, MM-ShiftKV.
- **Eviction + quantization together:** RDKV, QEvict (text).
- **Bias correction for quantized keys:** "Quantized Keys Steal Attention" (video diffusion).

What is not yet measured by others, as far as our searches found: the eviction *timeline* of
visual tokens within a single long answer, under quantization, with verified token-to-image
mapping. Novelty must be re-checked before any submission; this area publishes weekly.

## 4. How we work

The project is a **measurement**, so the instrument has to be trusted before its numbers are.

1. Build in **stages that follow the model's own flow**.
2. For each stage: **research** (the model's source code for facts, papers for design choices) →
   **write the decisions and pass criteria down first** → build → **verify on the real model** over
   13 images → show it visually.
3. A failed check is **reported, diagnosed, and never quietly redefined**. A revised check is
   pre-registered before it runs, and the original result is kept (e.g. `STAGE1_REPORT_v1.md`).
4. Checks include **negative controls**: deliberately broken versions that must fail, proving a
   check can catch an error.
5. Nothing is hard-coded: all settings in `config.json`; every result stores its settings.

## 5. What is verified, and what was found

**Stage 1 (image → tokens), verified.** Token *k* is the square at row *k* ÷ W, column *k* mod W,
proven from the resize rule through to the language model's input and positions. Key proof: with
the vision encoder's mixing switched off, inverting one square changes exactly one token — the
predicted one (294 squares; every other token changes by exactly 0).
Findings: each cached visual token also carries information from its **neighbours** (the encoder
mixes context; a neighbour moves ~0.27× as much); Qwen's documented `max_pixels=` is **silently
ignored** in transformers 5.11 (we pass `size=`); the vision encoder can **overflow fp16** on rare
inputs (guarded by a check).

**Stage 2 (attention), verified.** The attention we compute equals Qwen's own reference formula
exactly, reproduces the model's real attention output to within 0.05%, reads keys bit-identical
to the cache, and does not change the model's answer. Negative controls (wrong head mapping, wrong
scaling, missing key) are all caught.
Observations: the strongest attention while naming an object tends to land on that object, but
attention is noisy and partly on background; most attention in middle layers goes to the first
token (attention sink).

**Exploratory full pipeline (not yet verified stage by stage).** Rounding changes 8.8% (INT4) /
33.5% (INT2) of eviction decisions, more for visual than text tokens. But late-needed evidence is
lost ~80% of the time even at full precision, and rounding does not increase that (INT2 slightly
decreases it). Caveats: counterfactual eviction (no accuracy), attention-defined evidence, 7–8
images, one budget.

## 6. Decisions and scope

- **No RAG for now** (archive-and-recall, retrieve-first are out of scope).
- **Open:** fp16 vs bfloat16 for the vision encoder.
- **Next: the kill test**, before verifying stages 3–7 in full. "Describe first, ask later": the
  model describes a photo under limited memory, then gets a GQA question about a detail, scored
  against the human answer. Six arms: full memory at FP16 / INT4 / INT2 and limited memory at
  FP16 / INT4 / INT2; the full-memory FP16 arm must reproduce the plain model's answers exactly
  (sanity check). Decision rule (pre-registered, 5-point threshold, paired
  McNemar): rounding makes forgetting worse → **GO**; forgetting hurts but rounding doesn't add →
  **PIVOT** (the eviction rule is the problem); forgetting doesn't hurt → **STOP**.
