"""Phase A — the eviction TIMELINE, and whether late-needed evidence dies earlier.

This is the rung-2 experiment: not "does the ranking shuffle" (rung 1, done) but
"does the gun note get binned earlier when the notes are stored roughly".

## Counterfactual eviction

Each arm keeps its OWN accumulating evicted-set. Once a token is evicted in an arm it
stays evicted and is masked out of that arm's future scoring, so eviction decisions
compound exactly as they would in a real deployment. The model itself still runs on the
full cache, so its output never changes.

What that buys: a real per-token death timeline, cheaply, with every arm seeing identical
inputs (so arms cannot diverge through stray randomness).
What it costs: no accuracy number, because the model is never actually fed a reduced
cache. That is Phase C. We label this counterfactual and never call it otherwise.

## Defining "needed later" without annotations

An ORACLE arm runs at FP16 and never evicts. It is the ground truth for what the answer
actually used. Tokens carrying the most attention in the LAST quarter of decode steps,
under the oracle, are the "late-needed" tokens -- the gun.

Then the question is simply: under each arm, when did those tokens die?

Caveat, stated plainly: this defines need by attention, and CVMA reports attention is a
mediocre proxy for true future utility. GQA evidence boxes are the rigorous version
(Phase E). This is the cheap first measurement, and it is the same construct as QEvict's
Future Missed Mass.

Usage:
    python -m visretain.run_timeline --items 8   # all settings from config.json
"""

import argparse, json, time
from pathlib import Path

import numpy as np
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from . import core as vr

from .cfg import CFG, ROOT, RESULTS, snapshot
MODEL_ID = CFG["model_id"]

# Arms come from the configured bit widths, e.g. [16, 4, 2] -> fp16, int4, int2.
# "oracle" is full precision and never evicts: the ground truth for what was needed.
# The rest evict under an identical budget; only key precision differs.
BITS = {"oracle": 16, **{("fp16" if b >= 16 else f"int{b}"): b for b in CFG["quant"]["bits"]}}
ARMS = list(BITS)                 # oracle first, then the evicting arms

CTX = {"collect": False, "visual_mask": None, "layers": CFG["layers"], "width": 0,
       "keep_frac": CFG["eviction"]["keep_frac"], "recent": CFG["eviction"]["recent_window"],
       "state": {}, "prompt_len": 0, "late_frac": CFG["evidence"]["window_frac"], "n_steps": 0}


def _repeat_kv(k, n_heads):
    n_kv = k.shape[1]
    return k if n_kv == n_heads else k.repeat_interleave(n_heads // n_kv, dim=1)


def visretain_attention(module, query, key, value, attention_mask=None, **kw):
    out = sdpa_attention_forward(module, query, key, value, attention_mask, **kw)
    obs = CTX.get("observer")
    if obs is not None:                 # stage-2 verification watches the real inputs/outputs
        with torch.no_grad():
            obs(module, query, key, value, attention_mask, out, kw)
    if CTX["collect"] and query.shape[2] == 1:
        layer = getattr(module, "layer_idx", -1)
        if CTX["layers"] is None or layer in CTX["layers"]:
            with torch.no_grad():
                _step(module, layer, query, key)
    return out


def _new_state(H, W):
    return {a: {"acc": torch.zeros(H, W),
                "evicted": torch.zeros(W, dtype=torch.bool),
                "death": torch.full((W,), -1, dtype=torch.long)}
            for a in ARMS}


def pipeline_qk(module, query, key):
    """What the pipeline scores, from what the attention layer receives at a decode step:
    the step's query per head [H, d], the cached keys expanded to every query head
    [H, S, d], and the model's OWN scaling (D2.1). fp32 for our arithmetic (D2.2).
    Stage 2 verifies exactly this function against the model."""
    H = query.shape[1]
    return (query[0, :, 0, :].float().cpu(), _repeat_kv(key, H)[0].float().cpu(),
            float(module.scaling))


def _step(module, layer, query, key):
    q, k_fp, scale = pipeline_qk(module, query, key)
    H = q.shape[0]
    S = k_fp.shape[1]
    P = CTX["prompt_len"]
    step = S - P
    if S > CTX["width"]:
        return

    st = CTX["state"].setdefault(int(layer), _new_state(H, CTX["width"]))
    budget = max(8, int(CTX["keep_frac"] * P))

    # Precompute the quantized caches once per step, shared across arms of equal bits.
    kq = {16: k_fp}
    for b in set(BITS.values()) - {16}:
        kq[b] = vr.quantize_keys(k_fp, b, CFG["quant"]["scheme"], CFG["quant"]["group"])[0]

    for arm in ARMS:
        s = st[arm]
        scores = vr.attn_scores(q, kq[BITS[arm]], scale)   # [H, S]
        dead = s["evicted"][:S]
        if dead.any():
            scores = scores.masked_fill(dead.unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        s["acc"][:, :S] += probs

        if arm == "oracle":
            # per-step attention on each visual token: drives the demo's need-curve
            s.setdefault("steps", []).append(probs.mean(0)[CTX["vis_idx"]].clone())
            continue                                        # never evicts

        # H2O-style: rank by accumulated attention, averaged over heads.
        resident = ~dead
        n_res = int(resident.sum())
        if n_res <= budget:
            continue
        rank = s["acc"][:, :S].mean(0).clone()
        rank[dead] = float("inf")
        rank[max(0, S - CTX["recent"]):] = float("inf")     # protect recent window
        n_evict = min(n_res - budget, int(torch.isfinite(rank).sum()))
        if n_evict <= 0:
            continue
        victims = rank.topk(n_evict, largest=False).indices
        s["evicted"][victims] = True
        s["death"][victims] = step


def finalize():
    """Per layer: were the DEFERRED-IMPORTANT tokens already dead when needed?

    Deferred-important ("the gun") = visual tokens the oracle attends to a lot LATE but
    little EARLY. Defining them by late attention alone would be circular: H2O ranks by
    accumulated attention, so high-late-attention tokens are heavy hitters it protects by
    construction, and they would survive no matter what precision we used. The low-early
    condition is what makes them genuinely at risk, and it is what "deferred" means.

    Headline metric: fraction of them already evicted before the late window began --
    gone before they were needed.
    """
    vm_p = CTX["visual_mask"]
    P, W = len(vm_p), CTX["width"]
    vm = torch.cat([vm_p, torch.zeros(W - P, dtype=torch.bool)])
    # The oracle logs one row per DECODE step (the first answer token comes out of
    # prefill), so rows j = 0..n_dec-1 are decode steps 1..n_dec. Both the evidence
    # windows and need_step come from this one count, so they line up exactly.
    n_dec = max(1, CTX["n_steps"] - 1)
    w = max(1, int(n_dec * CTX["late_frac"]))
    need_step = n_dec - w + 1                                  # first step of the late window
    out = []
    for layer, st in sorted(CTX["state"].items()):
        vis_idx = vm.nonzero(as_tuple=True)[0]
        if len(vis_idx) < 32:
            continue
        o = st["oracle"]
        # Early/late windows from the ACTUAL answer length, after it has ended. (Doing
        # this on the fly would assume max_new_tokens, so an answer that stops early
        # would get a much narrower "late" window than the need_step check below.)
        A = torch.stack(o["steps"])                       # [decode steps, n_visual]
        assert A.shape[0] == n_dec, (A.shape[0], n_dec)
        early, late = A[:w].sum(0), A[n_dec - w:].sum(0)

        # rank-normalise both, then score = needed late AND ignored early
        def nrank(x):
            r = torch.empty_like(x)
            r[x.argsort()] = torch.arange(len(x), dtype=x.dtype)
            return r / max(1, len(x) - 1)
        deferred = nrank(late) - nrank(early)
        k = max(8, int(CFG["evidence"]["top_frac"] * len(vis_idx)))
        gun = vis_idx[deferred.topk(k).indices]

        row = {"layer": layer, "n_visual": int(vm.sum()), "n_gun": int(len(gun)),
               "need_step": need_step,
               "deferred_gap": float(deferred.topk(k).values.mean())}
        for arm in ARMS[1:]:
            d = st[arm]["death"]
            dv, dg = d[vis_idx], d[gun]
            died = dg >= 0
            # THE metric: dead before the answer needed it
            gone_when_needed = float(((dg >= 0) & (dg < need_step)).float().mean())
            row[arm] = {
                "visual_death_mean": float(dv[dv >= 0].float().mean()) if (dv >= 0).any() else -1.0,
                "visual_alive_frac": float((dv < 0).float().mean()),
                "gun_death_mean": float(dg[died].float().mean()) if died.any() else -1.0,
                "gun_alive_frac": float((dg < 0).float().mean()),
                "gun_died": int(died.sum()),
                "gone_when_needed": gone_when_needed,
            }
        # attention share on the evidence tokens at each decode step (full cache)
        if o.get("steps"):
            A = torch.stack(o["steps"])                       # [steps, n_visual]
            local = torch.searchsorted(vis_idx, gun)
            row["need_curve"] = (A[:, local].sum(1) / A.sum(1).clamp(min=1e-12)).tolist()
            row["gun_local"] = local.tolist()
        row["curves"] = {a: st[a]["death"][vis_idx].tolist() for a in ARMS[1:]}
        row["gun_idx"] = gun.tolist()
        row["gun_curves"] = {a: st[a]["death"][gun].tolist() for a in ARMS[1:]}
        out.append(row)
    return out


def visual_token_mask(model, input_ids):
    cfg = model.config
    ids = [getattr(cfg, n, None) for n in ("image_token_id", "video_token_id")]
    m = torch.zeros(input_ids.shape[-1], dtype=torch.bool)
    for i in [i for i in ids if i is not None]:
        m |= (input_ids[0].cpu() == i)
    return m


def build_items(n):
    """The first n pilot images, ORIGINAL size. Downloaded once into the cache folder,
    then read from disk, so everything works offline afterwards."""
    from PIL import Image
    cache = ROOT / CFG["pilot"]["cache_dir"]
    cache.mkdir(parents=True, exist_ok=True)
    out = []
    for cid, q in CFG["pilot"]["items"][:n]:
        f = cache / f"{cid}.jpg"
        if not f.exists():
            import urllib.request
            with urllib.request.urlopen(CFG["pilot"]["url"].format(id=cid), timeout=30) as r:
                f.write_bytes(r.read())
        out.append({"image": Image.open(f).convert("RGB"), "question": q, "id": cid})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=CFG["generation"]["max_new_tokens"])
    ap.add_argument("--keep-frac", type=float, default=CFG["eviction"]["keep_frac"],
                    help="cache budget as a fraction of prompt length")
    ap.add_argument("--recent", type=int, default=CFG["eviction"]["recent_window"])
    ap.add_argument("--layers", type=str, default=",".join(map(str, CFG["layers"])))
    ap.add_argument("--image", type=str, default="",
                    help="your own image: a file path or URL (overrides the pilot set)")
    ap.add_argument("--question", type=str, default="",
                    help="question to ask about --image")
    ap.add_argument("--out", type=str, default=str(RESULTS / "timeline"))
    args = ap.parse_args()

    CTX["keep_frac"], CTX["recent"] = args.keep_frac, args.recent
    CTX["layers"] = [int(x) for x in args.layers.split(",")] if args.layers else None
    model, proc, dev = load_model()

    outdir = Path(args.out); outdir.mkdir(exist_ok=True)
    if args.image:
        items = [{"image": load_image(args.image),
                  "question": args.question or CFG["generation"]["default_question"]}]
    else:
        items = build_items(args.items)

    for i, item in enumerate(items):
        print(f"[{i}] answering: {item['question'][:70]}", flush=True)
        rec = run_item(model, proc, dev, item, i, args.max_new_tokens)
        (outdir / f"item_{i:03d}.json").write_text(json.dumps(rec))
        g = rec["layers"][0] if rec["layers"] else {}
        print(f"[{i}] {rec['n_steps']} steps, {rec['n_visual']} vis, budget "
              f"{rec['budget']}, {rec['secs']}s"
              + (f" :: L{g['layer']} gone-when-needed " + " ".join(
                 f"{a}={g[a]['gone_when_needed']:.2f}" for a in ARMS[1:]) if g else ""), flush=True)
    print("wrote", outdir)


def load_model():
    """Load Qwen2.5-VL-3B with our attention shim on the LANGUAGE model only."""
    ALL_ATTENTION_FUNCTIONS["visretain"] = visretain_attention
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    dev = ("cuda" if torch.cuda.is_available() else
           "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"loading {MODEL_ID} on {dev} ...", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.float16,
        # a bare string only reaches the vision tower, which has no KV cache
        attn_implementation={"text_config": "visretain",
                             "vision_config": "sdpa"}).to(dev).eval()
    return model, proc, dev


def pixel_budget():
    """Qwen's own pixel budget. Passed as size=: max_pixels= is silently ignored in
    transformers 5.11 (checked; see STAGES.md stage 1)."""
    return {"shortest_edge": CFG["image"]["min_pixels"], "longest_edge": CFG["image"]["max_pixels"]}


def load_image(src):
    import io, urllib.request
    from PIL import Image
    if src.startswith("http"):
        with urllib.request.urlopen(src, timeout=30) as r:
            return Image.open(io.BytesIO(r.read())).convert("RGB")
    return Image.open(src).convert("RGB")


def run_item(model, proc, dev, item, i, max_new, streamer=None, on_setup=None):
    """One image + question through the model, all arms logged. Returns the record.

    on_setup(info) is called before decoding starts, with the image and its token
    grid, so a live viewer can draw the grid while the answer is generated.
    streamer is passed straight to model.generate (one call per generated token).
    """
    import base64, io as _io
    # Cap image scale so every run has a comparable number of visual tokens. Qwen's
    # dynamic resolution would otherwise turn a 12 MP phone photo into ~15k tokens:
    # slow, likely out of memory on 16 GB, and not comparable with the pilot set.
    msgs = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": item["question"] +
         CFG["generation"]["prompt_suffix"]}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[item["image"]], return_tensors="pt",
                  size=pixel_budget()).to(dev)

    P = inputs["input_ids"].shape[1]
    vmask = visual_token_mask(model, inputs["input_ids"])
    CTX["vis_idx"] = vmask.nonzero(as_tuple=True)[0]
    CTX.update(visual_mask=vmask, prompt_len=P, width=P + max_new,
               n_steps=max_new, state={}, collect=True)

    # grid geometry: Qwen patches are 14 px, merged 2x2 -> one visual token per 28 px
    _, gh, gw = [int(x) for x in inputs["image_grid_thw"][0]]
    rows_, cols_ = gh // 2, gw // 2
    assert rows_ * cols_ == int(vmask.sum()), (rows_, cols_, int(vmask.sum()))
    buf = _io.BytesIO()
    item["image"].resize((gw * 14, gh * 14)).save(buf, "JPEG", quality=82)
    info = {"grid": [rows_, cols_], "image_px": [gw * 14, gh * 14],
            "image_b64": base64.b64encode(buf.getvalue()).decode(),
            "question": item["question"], "n_visual": int(vmask.sum()),
            "budget": int(CTX["keep_frac"] * P), "prompt_len": P, "max_new": max_new}
    if on_setup:
        on_setup(info)

    t0 = time.time()
    try:
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                 streamer=streamer)
    finally:
        CTX["collect"] = False
    CTX["n_steps"] = int(out.shape[1] - P)

    gen_ids = out[0, P:].tolist()
    tokens = [proc.tokenizer.decode([t], skip_special_tokens=True) for t in gen_ids]
    return {**info, "item": i, "tokens": tokens, "n_steps": CTX["n_steps"],
            # with too few steps there is no meaningful "late" window to measure
            "too_short": CTX["n_steps"] < CFG["evidence"]["min_steps"],
            "config": snapshot(),
            "secs": round(time.time() - t0, 1), "layers": finalize()}


if __name__ == "__main__":
    main()
