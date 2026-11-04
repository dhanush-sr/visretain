"""Stage 1 + Stage 2 gate: attach the VisRetain harness to a real VLM and ask the
one question the project rests on --

    At an identical budget, does KV-cache rounding change WHICH visual tokens
    eviction would delete, and does it hit visual tokens harder than text?

Shadow ranking: we run ONE real decode pass. At every decode step we take the real
post-RoPE query and the real key cache, and score that same cache several ways
(FP16, INT4, INT2, and bias-corrected). Nothing is actually evicted, so all arms see
byte-identical inputs and the decode path cannot diverge. This is the cheapest design
that answers the gate, and it is what the Top-4 gaps note prescribes for stage one.

Usage:
    python3 run_kill_test.py --items 20 --max-new-tokens 200
"""

import argparse, json, time
from pathlib import Path

import numpy as np
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.integrations.sdpa_attention import sdpa_attention_forward

import visretain as vr

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
ARMS = [("B1_int4", 4), ("B2_int2", 2)]   # B0 (FP16) is the reference scoring
SCHEMES = ["kivi", "per_channel_global"]  # realistic + the noise sanity check

# Filled by the driver before each item; the attention shim reads them.
CTX = {"collect": False, "visual_mask": None, "records": [], "step": 0,
       "budget_frac": 0.25, "layers": None, "every": 20, "width": 0,
       "acc": {}}

# Arms carried in the ACCUMULATED scorer. Instantaneous (single-query) scoring is
# TOVA-like; H2O and EMA rank by attention summed over decode steps, and accumulation
# suppresses per-step noise. Since our effect is variance-driven, surviving
# accumulation is the real test of the premise -- hence both are measured.
ACC_ARMS = ["fp", "int4", "int2", "int4_corr", "int2_corr"]


def _repeat_kv(k, n_heads):
    """Expand GQA key heads to query heads so every query head scores its own keys."""
    n_kv = k.shape[1]
    if n_kv == n_heads:
        return k
    return k.repeat_interleave(n_heads // n_kv, dim=1)


def visretain_attention(module, query, key, value, attention_mask=None, **kw):
    """Drop-in attention that computes the real output via SDPA, and on the side
    records the shadow-ranking comparison for this (layer, step). Read-only w.r.t.
    the model: the returned tensors are exactly SDPA's (NFR 'non-interference')."""
    out = sdpa_attention_forward(module, query, key, value, attention_mask, **kw)

    if CTX["collect"] and query.shape[2] == 1:  # decode steps only, not prefill
        layer = getattr(module, "layer_idx", -1)
        if CTX["layers"] is None or layer in CTX["layers"]:
            with torch.no_grad():
                _record(layer, query, key)
    return out


def _record(layer, query, key):
    """Score one cache several ways and store the per-arm agreement metrics."""
    H = query.shape[1]
    q = query[0, :, 0, :].float().cpu()            # [H, d] this step's query
    k = _repeat_kv(key, H)[0].float().cpu()        # [H, S, d] the whole cache
    S = k.shape[1]

    # The mask covers the prompt; the cache also holds tokens generated so far,
    # which are text by construction. Pad with False to the current cache length.
    vmask = CTX["visual_mask"]
    if vmask is None:
        return
    P = len(vmask)
    if S < P:
        return
    if S > P:
        vmask = torch.cat([vmask, torch.zeros(S - P, dtype=torch.bool)])
    if vmask.sum() < 16:
        return  # need enough visual tokens for the split to mean anything

    budget = max(1, int(S * CTX["budget_frac"]))   # how many eviction would delete
    s_fp = vr.attn_scores(q, k)

    # ---- accumulated (H2O / EMA-style) scoring, every step ----
    # Decode-only: accumulation starts at the first decode step, so no prefill
    # attention (computed on full-precision keys) leaks in and masks the effect.
    acc = CTX["acc"].setdefault(int(layer),
                                {a: torch.zeros(H, CTX["width"]) for a in ACC_ARMS})
    k4, st4 = vr.quantize_keys(k, 4, "kivi")
    k2, st2 = vr.quantize_keys(k, 2, "kivi")
    s4, s2 = vr.attn_scores(q, k4), vr.attn_scores(q, k2)
    for arm, sc in (("fp", s_fp), ("int4", s4), ("int2", s2),
                    ("int4_corr", vr.correct(s4, vr.jensen_term(q, st4))),
                    ("int2_corr", vr.correct(s2, vr.jensen_term(q, st2)))):
        acc[arm][:, :S] += torch.softmax(sc, dim=-1)

    # Per-step instantaneous metrics are expensive and highly redundant between
    # neighbouring steps; subsample them. Accumulation above runs every step.
    if (S - P) % CTX["every"] != 0:
        return

    row = {"layer": int(layer), "step": S - P, "S": S,
           "n_visual": int(vmask.sum()), "budget": budget}

    for scheme in SCHEMES:
        for name, bits in ARMS:
            k_hat, step = vr.quantize_keys(k, bits, scheme)
            s_q = vr.attn_scores(q, k_hat)
            sigma2 = vr.jensen_term(q, step)
            s_c = vr.correct(s_q, sigma2)

            for split, m in (("all", None), ("visual", vmask), ("text", ~vmask)):
                if m is not None and m.sum() < 8:
                    continue
                # Budget must scale with the split, else a split smaller than the
                # global budget has its whole set in the bottom-k of both arms and
                # Jaccard is a trivial 1.0 rather than a measurement.
                n_split = S if m is None else int(m.sum())
                b = max(1, min(int(n_split * CTX["budget_frac"]), n_split - 1))
                tag = f"{scheme}|{name}|{split}"
                row[tag] = vr.compare(s_fp, s_q, b, mask=m)
                # bias-vs-noise: does removing the predicted Jensen term help?
                row[tag + "|corrected"] = vr.compare(s_fp, s_c, b, mask=m)
            # magnitude of the predicted bias, visual vs text
            row[f"{scheme}|{name}|sigma2_visual"] = float(sigma2[:, vmask].mean())
            row[f"{scheme}|{name}|sigma2_text"] = float(sigma2[:, ~vmask].mean())
            row[f"{scheme}|{name}|logit_scale"] = float(s_fp.abs().mean())

    CTX["records"].append(row)


def finalize_accumulated(S_final):
    """End-of-sequence comparison of ACCUMULATED rankings, per layer.

    This is the decisive test. Instantaneous scoring is TOVA-like; H2O/EMA rank by
    attention summed over decode steps. Our effect is variance-driven and
    accumulation averages variance away, so if the eviction sets still disagree
    here, the premise survives the scoring rule real policies actually use.
    """
    out = []
    vmask = CTX["visual_mask"]
    P = len(vmask)
    vm = torch.cat([vmask, torch.zeros(S_final - P, dtype=torch.bool)])[:S_final]
    for layer, acc in sorted(CTX["acc"].items()):
        fp = acc["fp"][:, :S_final]
        row = {"layer": layer, "S": S_final, "n_visual": int(vm.sum())}
        for arm in ACC_ARMS[1:]:
            for split, m in (("visual", vm), ("text", ~vm)):
                if m.sum() < 8:
                    continue
                n_split = int(m.sum())
                b = max(1, min(int(n_split * CTX["budget_frac"]), n_split - 1))
                row[f"acc|{arm}|{split}"] = vr.compare(
                    fp, acc[arm][:, :S_final], b, mask=m)
        out.append(row)
    return out


def visual_token_mask(model, input_ids):
    """True at positions holding image patch tokens. Tagging these is FR6 and the
    whole visual-vs-text split depends on it."""
    cfg = model.config
    ids = [getattr(cfg, n, None) for n in ("image_token_id", "video_token_id")]
    ids = [i for i in ids if i is not None]
    m = torch.zeros(input_ids.shape[-1], dtype=torch.bool)
    for i in ids:
        m |= (input_ids[0].cpu() == i)
    return m


# Pilot images (COCO val2017, public URLs). The gate only needs real photographs
# with enough visual tokens; GQA's value is its ground-truth evidence regions, which
# Gap 2 needs and the gate does not. `--dataset gqa` switches over once that lands.
_PILOT = [
    (139, "Describe this room and every object you can see in it."),
    (285, "What animal is this and what is its surroundings like?"),
    (632, "Describe the scene, the furniture and any objects on the surfaces."),
    (724, "What is in this image, and what is the weather and setting?"),
    (776, "Describe everything in this picture and how the items relate."),
    (785, "What is the person doing and what equipment are they using?"),
    (802, "Describe the people, their clothing and what they are doing."),
    (872, "What objects are in this scene and where is each one placed?"),
    (885, "Describe the environment and any animals or people present."),
    (39769, "What animals are shown, where are they, and what surrounds them?"),
]


def build_items(n, dataset="pilot"):
    """Items for the run. Prompts force long outputs on purpose: a one-word answer
    gives one eviction event and no timeline (why NExT-QA/MSVD-QA were dropped
    after Review 1)."""
    if dataset == "gqa":
        from datasets import load_dataset
        instr = load_dataset("lmms-lab/GQA", "testdev_balanced_instructions",
                             split="testdev")
        imgs = load_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev")
        by_id = {r["id"]: r["image"] for r in imgs}
        out = []
        for r in instr:
            if r["imageId"] in by_id:
                out.append({"image": by_id[r["imageId"]], "question": r["question"]})
            if len(out) >= n:
                break
        return out

    import io, urllib.request
    from PIL import Image
    out = []
    for cid, q in _PILOT[:n]:
        url = f"http://images.cocodataset.org/val2017/{cid:012d}.jpg"
        with urllib.request.urlopen(url, timeout=30) as r:
            img = Image.open(io.BytesIO(r.read())).convert("RGB")
        out.append({"image": img, "question": q})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--budget-frac", type=float, default=0.25)
    ap.add_argument("--layers", type=str, default="",
                    help="comma-separated layer indices; empty = all")
    ap.add_argument("--every", type=int, default=20,
                    help="compute instantaneous metrics every N decode steps")
    ap.add_argument("--dataset", type=str, default="pilot", choices=["pilot","gqa"])
    ap.add_argument("--out", type=str, default="results")
    args = ap.parse_args()

    CTX["budget_frac"] = args.budget_frac
    CTX["every"] = args.every
    CTX["layers"] = ([int(x) for x in args.layers.split(",")] if args.layers else None)

    ALL_ATTENTION_FUNCTIONS["visretain"] = visretain_attention

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    dev = ("cuda" if torch.cuda.is_available() else
           "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"loading {MODEL_ID} on {dev} ...", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.float16,
        # per-subconfig: the LANGUAGE model is the one with the KV cache we study.
        # Passing a bare string only reaches the vision tower, which has no KV cache.
        attn_implementation={"text_config": "visretain",
                             "vision_config": "sdpa"}).to(dev).eval()

    items = build_items(args.items, args.dataset)
    outdir = Path(args.out); outdir.mkdir(exist_ok=True)

    for i, item in enumerate(items):
        img, q = item["image"], item["question"]
        msgs = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": q + " Explain your reasoning step by step, "
                                         "describing what you see in detail."}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to(dev)

        P = inputs["input_ids"].shape[1]
        CTX["visual_mask"] = visual_token_mask(model, inputs["input_ids"])
        CTX["records"], CTX["step"], CTX["collect"] = [], 0, True
        CTX["acc"] = {}                       # fresh accumulators per item
        CTX["width"] = P + args.max_new_tokens

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False)  # greedy: reproducibility (NFR)
        CTX["collect"] = False
        ans = proc.batch_decode(out[:, P:], skip_special_tokens=True)[0]

        rec = {"item": i, "question": q, "answer": ans,
               "n_visual": int(CTX["visual_mask"].sum()), "prompt_len": P,
               "n_steps": int(out.shape[1] - P),
               "secs": round(time.time() - t0, 1), "records": CTX["records"],
               "accumulated": finalize_accumulated(int(out.shape[1]))}
        (outdir / f"item_{i:03d}.json").write_text(json.dumps(rec))
        print(f"[{i}] {len(CTX['records'])} obs, {rec['n_steps']} steps, "
              f"{rec['n_visual']} visual tok, {rec['secs']}s :: {ans[:50]}...",
              flush=True)

    print("wrote", outdir)


if __name__ == "__main__":
    main()
