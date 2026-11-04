"""Stage-by-stage verification of the VisRetain instrument, on the real model.

Each stage returns (a) the data the stages page draws and (b) its checks, with the pass
criteria written in STAGES.md BEFORE this code. Nothing here is hard-coded: model sizes come
from the model's own processor/config, settings from config.json.

Stage 1 — how Qwen2.5-VL turns an image into the visual tokens the language model caches.
"""

import base64, io, math, random

import numpy as np
import torch
import torchvision.transforms.v2.functional as TF
from PIL import Image

import run_timeline as rt
import visretain as vr
from cfg import CFG


def _b64(img, fmt="PNG"):
    buf = io.BytesIO(); img.save(buf, fmt)
    return base64.b64encode(buf.getvalue()).decode()


def predict_resize(H, W, F, min_px, max_px):
    """The resize rule as WE wrote it down from Qwen's source (STAGES.md, research 1),
    returned with each arithmetic step so the page can show the math with real numbers.
    Check 1.1 compares this against what the processor actually does."""
    steps = []
    h, w = round(H / F) * F, round(W / F) * F
    steps.append({"what": f"Round each side to the nearest multiple of {F} px",
                  "math": f"height: round({H} / {F}) × {F} = {h}   ·   width: round({W} / {F}) × {F} = {w}"})
    if h * w > max_px:
        beta = math.sqrt(H * W / max_px)
        h = max(F, math.floor(H / beta / F) * F)
        w = max(F, math.floor(W / beta / F) * F)
        steps.append({"what": f"Too many pixels ({round(H/F)*F*round(W/F)*F:,} > budget {max_px:,}), so shrink, keeping the shape",
                      "math": f"β = √({H}·{W} / {max_px:,}) = {beta:.3f}   →   height ⌊{H} / {beta:.3f} / {F}⌋ × {F} = {h},  width ⌊{W} / {beta:.3f} / {F}⌋ × {F} = {w}"})
    elif h * w < min_px:
        beta = math.sqrt(min_px / (H * W))
        h = math.ceil(H * beta / F) * F
        w = math.ceil(W * beta / F) * F
        steps.append({"what": f"Too few pixels (< minimum {min_px:,}), so enlarge, keeping the shape",
                      "math": f"β = √({min_px:,} / ({H}·{W})) = {beta:.3f}   →   height ⌈{H} × {beta:.3f} / {F}⌉ × {F} = {h},  width ⌈{W} × {beta:.3f} / {F}⌉ × {F} = {w}"})
    else:
        steps.append({"what": "Within the pixel budget, so no further shrinking",
                      "math": f"{h} × {w} = {h*w:,} px  ≤  budget {max_px:,} px"})
    return h, w, steps


def _prompt(proc, question):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


CHECK15A_NAME = "Routing: with the encoder's mixing switched off, changing square k changes token k and nothing else"
CHECK15B_NAME = "Mixing: how much a change in one square spreads to other tokens (measured, not pass/fail)"
FP16_MAX = float(torch.finfo(torch.float16).max)


from contextlib import contextmanager


@contextmanager
def self_only_vision_attention():
    """Switch off token mixing in the VISION encoder only (check 1.5a): every attention
    layer returns each token's own value vector, i.e. attention that can only see itself.
    Projections, rotary positions, MLPs, norms and the merger are untouched. The vision
    tower looks attention up by the name "sdpa" at each call; the language model goes
    through our own hook, which calls SDPA directly, so it is unaffected."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    orig = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def self_only(module, query, key, value, attention_mask=None, **kw):
        return value.transpose(1, 2).contiguous(), None
    ALL_ATTENTION_FUNCTIONS["sdpa"] = self_only
    try:
        yield
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = orig


class Stage1:
    """Holds one image's stage-1 state so the page can ask for occlusion maps on click."""

    def __init__(self, model, proc, dev):
        self.model, self.proc, self.dev = model, proc, dev
        ip = proc.image_processor
        self.P, self.M, self.T = ip.patch_size, ip.merge_size, ip.temporal_patch_size
        self.F = self.P * self.M                                  # px per visual token side
        self.mean = torch.tensor(ip.image_mean)[:, None, None]
        self.std = torch.tensor(ip.image_std)[:, None, None]
        self.V = model.model.visual
        self.vcfg = self.V.config

    # ---------------------------------------------------------------- helpers
    def _process(self, img, question):
        return self.proc(text=[_prompt(self.proc, question)], images=[img],
                         size=rt.pixel_budget(), return_tensors="pt")

    def _rebuild(self, pixel_values, gh, gw):
        """Image rebuilt from the model's input rows, using OUR layout (research 2)."""
        P, M = self.P, self.M
        pv = pixel_values.reshape(gh // M, gw // M, M, M, 3, self.T, P, P)
        frames_equal = all(torch.equal(pv[..., 0, :, :], pv[..., t, :, :]) for t in range(1, self.T))
        x = pv[..., 0, :, :].permute(4, 0, 2, 5, 1, 3, 6).reshape(3, gh * P, gw * P)
        return x * self.std + self.mean, frames_equal

    @torch.no_grad()
    def _encode(self, pixel_values, grid, peak=False):
        """Encoder output [n_tokens, out_dim]; with peak=True also the largest |activation|
        in the last vision block (compared with fp16's limit in check 1.8)."""
        got = {}
        h = self.V.blocks[-1].register_forward_hook(
            lambda m, i, o: got.__setitem__("peak", float(o.float().abs().max()))) if peak else None
        try:
            out = self.V(pixel_values.to(self.dev, self.model.dtype), grid_thw=grid.to(self.dev))
        finally:
            if h: h.remove()
        E = out.pooler_output.float().cpu()
        return (E, got["peak"]) if peak else E

    # ---------------------------------------------------------------- main
    def run(self, img, question=None):
        question = question or CFG["generation"]["default_question"]
        H0, W0 = img.height, img.width
        mn, mx = CFG["image"]["min_pixels"], CFG["image"]["max_pixels"]
        h, w, steps = predict_resize(H0, W0, self.F, mn, mx)
        # what the model's OWN default budget would have done (shows why the cap exists)
        ip = self.proc.image_processor
        h_raw, w_raw, _ = predict_resize(H0, W0, self.F, ip.size.shortest_edge, ip.size.longest_edge)

        inp = self._process(img, question)
        t, gh, gw = [int(v) for v in inp["image_grid_thw"][0]]
        R, C = gh // self.M, gw // self.M
        ids = inp["input_ids"][0]
        img_id = self.model.config.image_token_id
        img_pos = (ids == img_id).nonzero(as_tuple=True)[0]

        checks = []

        def check(cid, name, ok, detail):
            checks.append({"id": cid, "name": name, "pass": bool(ok), "detail": detail})

        # 1.1 geometry
        check("1.1", "Our resize rule matches Qwen's processor", (gh * self.P, gw * self.P) == (h, w),
              f"predicted {w} × {h} px; processor produced {gw*self.P} × {gh*self.P} px")

        # 1.2 patch layout: rebuild the image from the model's numbers
        rebuilt, frames_equal = self._rebuild(inp["pixel_values"], gh, gw)
        ref = TF.resize(TF.pil_to_tensor(img.convert("RGB"))[None], [h, w],
                        interpolation=TF.InterpolationMode.BICUBIC, antialias=True).float()[0] / 255
        diff = float((rebuilt - ref).abs().max())
        check("1.2", "Image rebuilt from the model's input numbers matches the resized image",
              diff < 1e-4 and frames_equal,
              f"max difference {diff:.1e} on a 0–1 scale (pass < 1e-4); "
              f"the {self.T} frames per patch are {'identical' if frames_equal else 'NOT identical'}")

        # 1.3 token count
        check("1.3", "Number of image tokens = rows × columns", len(img_pos) == R * C,
              f"{len(img_pos)} image tokens in the prompt; grid {R} × {C} = {R*C}")

        # 1.4 Qwen's own 3D positions
        pos, _ = self.model.model.get_rope_index(inp["input_ids"], inp["mm_token_type_ids"],
                                                 inp["image_grid_thw"])
        p = pos[:, 0, img_pos]                                     # [3, n] (t, row, col)
        rows, cols = p[1] - p[1].min(), p[2] - p[2].min()
        k = torch.arange(len(img_pos))
        ok14 = torch.equal(rows, k // C) and torch.equal(cols, k % C) and bool((p[0] == p[0][0]).all())
        check("1.4", "Qwen's own positions put token k at row k ÷ W, column k mod W", ok14,
              f"all {len(img_pos)} tokens match row-major order" if ok14 else "MISMATCH")

        # 1.7 encoder output k lands in prompt slot k (captured from a real forward pass)
        E_img, peak = self._encode(inp["pixel_values"], inp["image_grid_thw"], peak=True)

        # 1.8 the encoder's output is finite; how close the last block gets to fp16's limit
        finite = bool(torch.isfinite(E_img).all())
        check("1.8", "The vision encoder's output is finite (no fp16 overflow)", finite,
              f"{'all' if finite else 'NOT all'} {E_img.shape[0]} token vectors finite; largest activation in the "
              f"last vision block {peak:,.0f} = {100*peak/FP16_MAX:.0f}% of the fp16 limit ({FP16_MAX:,.0f})")
        got = {}

        def grab(_mod, args, kwargs):
            got["emb"] = kwargs.get("inputs_embeds")
        hook = self.model.model.language_model.register_forward_pre_hook(grab, with_kwargs=True)
        try:
            with torch.no_grad():
                self.model(**{k_: v.to(self.dev) for k_, v in inp.items()})
        finally:
            hook.remove()
        slots = got["emb"][0, img_pos.to(got["emb"].device)].float().cpu()      # [n, dim]
        d17 = float((slots - E_img).abs().max())
        # and, independently of tolerance: each slot's closest encoder output is its own index
        nearest = torch.cdist(slots, E_img).argmin(dim=1)
        in_order = bool(torch.equal(nearest, torch.arange(len(img_pos))))
        check("1.7", "Encoder output k is what the language model receives in image slot k",
              d17 < 1e-3 and in_order,
              f"max difference {d17:.1e} (pass < 1e-3); each of the {len(img_pos)} slots is closest to "
              f"{'its own' if in_order else 'NOT its own'} encoder output")

        # 1.6 size cap
        n_raw = (h_raw // self.F) * (w_raw // self.F)
        cap = mx // (self.F * self.F)
        check("1.6", "Token count stays under the configured budget", R * C <= cap,
              f"{R*C} tokens ≤ {cap}" + (f" (without the cap this image would be {n_raw:,} tokens)" if n_raw != R * C else ""))

        # ---- the image exactly as Qwen sees it, kept for the causal test
        base = Image.fromarray((rebuilt.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy())
        self.base, self.question, self.grid_rc = base, question, (R, C)
        b_inp = self._process(base, question)
        assert [int(v) for v in b_inp["image_grid_thw"][0]] == [t, gh, gw], "base image re-gridded"
        self.grid_thw = b_inp["image_grid_thw"]
        self.E0 = self._encode(b_inp["pixel_values"], self.grid_thw)              # mixing on
        self.E0_norm = float(self.E0.norm(dim=1).mean())
        with self_only_vision_attention():
            self.E0s = self._encode(b_inp["pixel_values"], self.grid_thw)         # mixing off
        self.E0s_norm = float(self.E0s.norm(dim=1).mean())

        # 1.5 causal test: squares chosen now, tested one by one through occlude(record=True)
        n_sq = min(CFG["stage1"]["occlusion_squares"], R * C)
        rng = random.Random(CFG["stage1"]["occlusion_seed"])
        self.squares = [(s_ // C, s_ % C) for s_ in rng.sample(range(R * C), n_sq)]
        self.trials = {}

        # ---- the prompt as a token strip, image tokens collapsed into one block
        tok = self.proc.tokenizer
        strip, i = [], 0
        while i < len(ids):
            if ids[i] == img_id:
                j = i
                while j < len(ids) and ids[j] == img_id:
                    j += 1
                strip.append({"kind": "image", "count": j - i, "pos": [int(pos[1, 0, i]), int(pos[1, 0, j-1]), int(pos[2, 0, i]), int(pos[2, 0, j-1])]})
                i = j
            else:
                txt = tok.decode([int(ids[i])])
                kind = "special" if txt.startswith("<|") and txt.endswith("|>") else "text"
                strip.append({"kind": kind, "text": txt, "pos": int(pos[0, 0, i])})
                i += 1
        # position of the first token after the image (it does NOT advance by n_tokens)
        last = int(img_pos[-1])
        after = int(pos[0, 0, last + 1]) if last + 1 < len(ids) else None
        start = int(p[1].min())

        win_px = self.vcfg.window_size
        return {
            "question": question,
            "original": {"w": W0, "h": H0, "img": _b64(img.convert("RGB"), "JPEG")},
            "resize_steps": steps,
            "resized": {"w": w, "h": h, "img": _b64(base)},
            "budget": {"min_px": mn, "max_px": mx, "cap_tokens": cap, "uncapped_tokens": n_raw},
            "patch": {"px": self.P, "cols": gw, "rows": gh, "n": gh * gw,
                      "numbers": 3 * self.T * self.P * self.P, "frames": self.T},
            "token": {"px": self.F, "merge": self.M, "cols": C, "rows": R, "n": R * C,
                      "dim": int(self.E0.shape[1])},
            "encoder": {"layers": self.vcfg.depth, "window_px": win_px,
                        "window_tokens": win_px // self.F, "full_layers": list(self.vcfg.fullatt_block_indexes),
                        "hidden": self.vcfg.hidden_size},
            "positions": {"start": start, "t": int(p[0][0]), "after_image": after,
                          "jump": (after - start) if after is not None else None},
            "strip": strip,
            "react_frac": CFG["stage1"]["react_frac"],
            "squares": self.squares,
            "checks": checks,
            "pending_checks": [{"id": "1.5a", "name": CHECK15A_NAME}, {"id": "1.5b", "name": CHECK15B_NAME}],
        }

    @torch.no_grad()
    def occlude(self, r, c, record=False):
        """Invert one token-sized square of the image Qwen sees (D1.3) and measure how much
        every output token changes, twice: with the encoder's mixing switched off (1.5a,
        routing) and with the encoder unmodified (1.5b, mixing). The square is located in
        IMAGE space, never through our layout assumption."""
        R, C = self.grid_rc
        F = self.F
        arr = np.array(self.base)
        arr[r * F:(r + 1) * F, c * F:(c + 1) * F] = 255 - arr[r * F:(r + 1) * F, c * F:(c + 1) * F]
        inp = self._process(Image.fromarray(arr), self.question)
        target = r * C + c

        # 1.5a routing: mixing off -> only token `target` may change
        with self_only_vision_attention():
            E1s = self._encode(inp["pixel_values"], inp["image_grid_thw"])
        ch_s = (E1s - self.E0s).norm(dim=1) / self.E0s_norm
        fin_s = bool(torch.isfinite(ch_s).all())
        others = torch.cat([ch_s[:target], ch_s[target + 1:]])
        routing_ok = fin_s and int(ch_s.argmax()) == target and float(others.max()) < 1e-3

        # 1.5b mixing: the real encoder
        E1 = self._encode(inp["pixel_values"], inp["image_grid_thw"])
        change = (E1 - self.E0).norm(dim=1) / self.E0_norm
        fin = bool(torch.isfinite(change).all())
        safe = torch.nan_to_num(change, nan=-1.0)
        order = safe.argsort(descending=True)
        top = int(order[0])
        nxt = float(safe[order[1]]) if top == target else float(safe[top])
        runner = int(order[1]) if top == target else top
        res = {"r": r, "c": c, "target": target,
               # routing (mixing off)
               "routing_ok": routing_ok, "routing_finite": fin_s,
               "routing_own": float(ch_s[target]), "routing_other_max": float(others.max()),
               "routing_changed": int((ch_s > 1e-3).sum()),
               "routing_map": [round(float(v), 4) for v in torch.nan_to_num(ch_s, nan=-1.0)],
               # mixing (encoder as is)
               "finite": fin, "top": top, "correct": fin and top == target,
               "target_change": float(change[target]),
               "margin": float(change[target]) / max(nxt, 1e-9) if fin else None,
               "runner_up_distance": max(abs(runner // C - r), abs(runner % C - c)),
               "map": [round(float(v), 4) for v in safe]}
        if record and (r, c) in self.squares:
            self.trials[(r, c)] = res
            if len(self.trials) == len(self.squares):
                res["final"] = self.verdict()
        return res

    def verdict(self):
        """1.5a (pass/fail) and 1.5b (measured) from the recorded trials."""
        R, C = self.grid_rc
        T = list(self.trials.values())
        n = len(T)
        ok_a = sum(t["routing_ok"] for t in T)
        nonfin_a = sum(not t["routing_finite"] for t in T)
        worst_other = max(t["routing_other_max"] for t in T if t["routing_finite"]) if n > nonfin_a else float("nan")
        own = [t["routing_own"] for t in T if t["routing_finite"]]
        a = {"id": "1.5a", "name": CHECK15A_NAME, "pass": ok_a == n,
             "detail": f"{ok_a} / {n} random squares: only the changed square's token moved "
                       f"(its change {min(own):.2f}–{max(own):.2f}; largest change of any OTHER token "
                       f"{worst_other:.1e}, pass < 1e-3)" + (f"; {nonfin_a} square(s) overflowed fp16 (counted as failures)" if nonfin_a else "")}

        V = [t for t in T if t["finite"]]
        nonfin = n - len(V)
        m = [t["margin"] for t in V]
        maxd = max(R, C)
        sums, cnts = np.zeros(maxd), np.zeros(maxd)
        for t in V:
            for i, v in enumerate(t["map"]):
                d = max(abs(i // C - t["r"]), abs(i % C - t["c"]))
                sums[d] += v / t["target_change"]; cnts[d] += 1
        ring = [round(float(sums[d] / cnts[d]), 4) for d in range(maxd) if cnts[d]]
        top1 = sum(t["correct"] for t in V)
        weak = min(V, key=lambda t: t["margin"]) if V else None
        losers = [t for t in V if not t["correct"]]
        losers_adj = sum(t["runner_up_distance"] == 1 for t in losers)   # where the winner was
        one_away = ring[1] if len(ring) > 1 else 0.0
        if V:
            pl = lambda n, word: f"{n} {word}{'' if n == 1 else 's'}"
            detail = (f"its own token reacted most in {top1} of {len(V)} squares; margin over the next-largest "
                      f"{min(m):.2f}×–{max(m):.2f}× (median {float(np.median(m)):.2f}×); "
                      + (f"in the {pl(len(losers), 'square')} where it did not, the winner was an immediate "
                         f"neighbour in {losers_adj}; " if losers else "")
                      + f"on average a token 1 square away moves {one_away:.2f}× as much as the changed square's own")
            if nonfin:
                detail += f"; {nonfin} square(s) overflowed fp16"
        else:
            detail = "no finite squares"
        b = {"id": "1.5b", "name": CHECK15B_NAME, "pass": None, "detail": detail}
        return {"checks": [a, b], "ring": ring, "n": n,
                "weakest": {"r": weak["r"], "c": weak["c"], "margin": weak["margin"]} if weak else None}


# =====================================================================================
# Stage 2 — is the attention we capture the attention the model computes?
# =====================================================================================

class Stage2:
    """Runs one real answer with an observer on every attention layer.

    For the page: at every decode step and layer, how attention splits between the image,
    the prompt text, the first token and the answer so far, plus the full map over image
    tokens. For the proof: at the configured check steps, our pipeline's own computation
    (run_timeline.pipeline_qk + visretain.attn_scores) is compared with Qwen's reference
    implementation and with the model's actual attention output (STAGES.md, stage 2).
    """

    def __init__(self, model, proc, dev):
        self.model, self.proc, self.dev = model, proc, dev
        from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as qm
        self.qm = qm                                           # Qwen's eager reference + repeat_kv
        self.layers = None

    def _observer(self, module, query, key, value, mask, out, kw):
        if query.shape[2] != 1:                                # decode steps only
            return
        L = module.layer_idx
        S = key.shape[2]
        step = S - self.P
        q, k, scale = rt.pipeline_qk(module, query, key)       # EXACTLY what the pipeline scores
        a = torch.softmax(vr.attn_scores(q, k, scale), dim=-1)  # [H, S]
        am = a.mean(0)
        vm = torch.zeros(S, dtype=torch.bool); vm[:self.P] = self.vmask    # cache is longer than the prompt
        gen = torch.zeros(S, dtype=torch.bool); gen[self.P:] = True
        txt = ~vm & ~gen; txt[0] = False                                   # prompt text, minus the first token
        self.split.setdefault(L, []).append([float(am[0]), float(am[:S][txt].sum()),
                                              float(am[vm].sum()), float(am[gen].sum())])
        self.img_attn.setdefault(L, []).append(am[vm].numpy().astype(np.float32))

        if step in self.check_steps:
            # 2.4 rows are distributions
            row_err = float((a.sum(-1) - 1).abs().max())
            finite = bool(torch.isfinite(a).all())
            # 2.2 Qwen's reference formula, same inputs, fp32
            if mask is None:
                m = None
            elif mask.dtype == torch.bool:                  # True = may attend -> additive 0 / -inf
                m = torch.zeros(mask.shape).masked_fill(~mask.cpu(), float("-inf"))
            else:
                m = mask.float().cpu()
            ref_out, ref_w = self.qm.eager_attention_forward(
                module, query.float().cpu(), key.float().cpu(), value.float().cpu(), m,
                scaling=module.scaling)
            d22 = float((ref_w[0, :, 0, :] - a).abs().max())
            # 2.3 our weights x values vs the model's real (SDPA, fp16) output
            v = rt._repeat_kv(value, query.shape[1])[0].float().cpu()       # [H, S, d]
            ours = torch.einsum("hs,hsd->hd", a, v)
            real = out[0][0, 0].float().cpu() if out[0].shape[1] == 1 else out[0][0, :, 0].float().cpu()
            scale_out = float(real.abs().max())
            d23 = float((ours - real).abs().max()) / max(scale_out, 1e-9)
            # Negative controls: deliberately broken versions, scored by the SAME comparisons.
            # Each must fail, or the check that let it pass has no power (STAGES.md, N1-N3).
            Hq, kv = query.shape[1], key.shape[1]
            ctrl = {}
            k_n1 = key.repeat(1, Hq // kv, 1, 1)[0].float().cpu()       # head h -> key head h mod kv
            variants = {"N1": (q, k_n1, scale, v),
                        "N2": (q, k, scale * 1.1, v),
                        "N3": (q, k[:, :-1], scale, v[:, :-1])}         # cache without its newest key
            for name, (qq, kk, sc, vv) in variants.items():
                ac = torch.softmax(vr.attn_scores(qq, kk, sc), dim=-1)
                if ac.shape[-1] < a.shape[-1]:                          # pad to the full cache length
                    ac = torch.cat([ac, torch.zeros(ac.shape[0], a.shape[-1] - ac.shape[-1])], dim=-1)
                    vv = v
                ctrl[name] = (float((ref_w[0, :, 0, :] - ac).abs().max()),
                              float((torch.einsum("hs,hsd->hd", ac, vv) - real).abs().max()) / max(scale_out, 1e-9))
            self.rec.append({"layer": L, "step": step, "d22": d22, "d23": d23,
                             "row_err": row_err, "finite": finite, "controls": ctrl})
            self.keys[(L, step)] = key.detach().cpu().clone()               # for 2.1

    def run(self, img, question=None, max_new=None):
        question = question or CFG["generation"]["default_question"]
        max_new = max_new or CFG["stage2"]["max_new_tokens"]
        text = _prompt(self.proc, question + CFG["generation"]["prompt_suffix"])
        inp = self.proc(text=[text], images=[img], size=rt.pixel_budget(), return_tensors="pt").to(self.dev)
        ids = inp["input_ids"][0].cpu()
        self.P = len(ids)
        self.vmask = rt.visual_token_mask(self.model, inp["input_ids"])
        self.check_steps = set(CFG["stage2"]["check_steps"])
        self.split, self.img_attn, self.rec, self.keys = {}, {}, [], {}

        rt.CTX["observer"] = self._observer
        try:
            with torch.no_grad():
                out = self.model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                                          return_dict_in_generate=True)
        finally:
            rt.CTX["observer"] = None
        gen_ids = out.sequences[0, self.P:].tolist()
        tokens = [self.proc.tokenizer.decode([t], skip_special_tokens=True) for t in gen_ids]
        cache = out.past_key_values

        checks = []

        def check(cid, name, ok, detail):
            checks.append({"id": cid, "name": name, "pass": bool(ok), "detail": detail})

        n_layers = len(self.split)
        steps_checked = sorted({r["step"] for r in self.rec})
        # 2.1 hook keys == the cache's stored keys (the cache only appends, so the prefix must match)
        bad21 = []
        for (L, step), k in self.keys.items():
            stored = cache.layers[L].keys if hasattr(cache, "layers") else cache.key_cache[L]
            stored = stored[:, :, :k.shape[2]].cpu()
            if not torch.equal(stored, k):
                bad21.append((L, step))
        check("2.1", "The keys we read are exactly the keys the KV cache stores", not bad21,
              f"bit-identical for all {len(self.keys)} (layer, step) pairs checked: {n_layers} layers × "
              f"steps {steps_checked}" if not bad21 else f"MISMATCH at {bad21[:5]}")
        d22 = max(r["d22"] for r in self.rec)
        check("2.2", "Our attention equals Qwen's own reference formula", d22 < 1e-5,
              f"max difference {d22:.1e} (pass < 1e-5) across {len(self.rec)} (layer, step) pairs")
        d23 = max(r["d23"] for r in self.rec)
        check("2.3", "Our attention reproduces the model's actual attention output", d23 <= 0.01,
              f"max difference {d23*100:.3f}% of the output's largest value (pass ≤ 1%) across "
              f"{len(self.rec)} (layer, step) pairs; the model computes in fp16, we in fp32")
        # negative controls: every broken version must fail BOTH 2.2 and 2.3
        ctrl_txt, ctrl_ok = [], True
        for name, what in (("N1", "wrong head mapping"), ("N2", "scaling × 1.1"), ("N3", "newest key missing")):
            # judged by exactly the rule the real checks use (worst case over all layers/steps)
            c22 = max(r["controls"][name][0] for r in self.rec)
            c23 = max(r["controls"][name][1] for r in self.rec)
            caught = c22 >= 1e-5 and c23 > 0.01
            ctrl_ok &= caught
            ctrl_txt.append(f"{name} {what}: {'caught' if caught else 'NOT caught'} (2.2 diff {c22:.1e}, 2.3 diff {c23*100:.1f}%)")
        check("2.N", "The checks can fail: deliberately broken versions are caught", ctrl_ok, "; ".join(ctrl_txt))
        re_ = max(r["row_err"] for r in self.rec); fin = all(r["finite"] for r in self.rec)
        check("2.4", "Every attention row is a valid probability distribution", re_ < 1e-4 and fin,
              f"rows sum to 1 within {re_:.1e}; {'no' if fin else 'SOME'} NaN/inf")

        # 2.5 non-interference: same prompt, plain attention, same answer?
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        shim = ALL_ATTENTION_FUNCTIONS["visretain"]
        ALL_ATTENTION_FUNCTIONS["visretain"] = sdpa_attention_forward
        try:
            with torch.no_grad():
                plain = self.model.generate(**inp, max_new_tokens=max_new, do_sample=False)
        finally:
            ALL_ATTENTION_FUNCTIONS["visretain"] = shim
        same = plain[0, self.P:].tolist() == gen_ids
        check("2.5", "Watching attention does not change the model's answer", same,
              f"{len(gen_ids)} tokens, {'identical' if same else 'DIFFERENT'} to the answer with plain attention")

        checks.sort(key=lambda c: (c["id"] == "2.N", c["id"]))   # the control check reads best last

        # ---- data for the page (token geometry from the model's own processor)
        ip = self.proc.image_processor
        F = ip.patch_size * ip.merge_size
        R, C = [int(v) // ip.merge_size for v in inp["image_grid_thw"][0][1:]]
        layers = sorted(self.split)
        split = {L: np.array(self.split[L]).mean(0).round(4).tolist() for L in layers}
        self.maps = {L: np.stack(self.img_attn[L]) for L in layers}   # [steps, n_img]
        H = self.model.config.text_config if hasattr(self.model.config, "text_config") else self.model.config
        return {
            "question": question, "tokens": tokens, "grid": [R, C], "n_image": int(self.vmask.sum()),
            "prompt_len": self.P, "layers": layers,
            "split": split,        # per layer: [first token, prompt text, image, answer so far]
            "heads": {"q": H.num_attention_heads, "kv": H.num_key_value_heads,
                      "dim": H.hidden_size // H.num_attention_heads},
            "check_steps": steps_checked, "checks": checks,
            # map j is the attention while choosing word j+1: the first word comes from reading
            # the prompt (prefill), which is not a decode step, so it has no map of its own
            "n_maps": len(self.img_attn[layers[0]]),
            "display_layer": CFG["display_layer"],
            "image": _b64(img.resize((C * F, R * F), Image.BICUBIC), "JPEG"),
        }

    def layer_map(self, L):
        """One layer's attention on the image tokens, every step (fetched on demand).
        share = exact fraction of that step's attention on the image (sums of real weights).
        q = the map at each step scaled to 0-255 of that step's maximum: a heat map needs only
        relative intensity, and this keeps one layer to ~20k small integers."""
        m = self.maps[L]                                        # [steps, n_img]
        mx = m.max(axis=1, keepdims=True)
        q = np.round(255 * m / np.maximum(mx, 1e-12)).astype(int)
        # mx lets the page recover each patch's real weight (q / 255 * mx) to compare across words
        return {"layer": L, "share": m.sum(axis=1).round(5).tolist(), "q": q.tolist(),
                "mx": [float(v) for v in mx[:, 0]]}
