"""VisRetain core: fake KV quantization, eviction scoring, and the rank-shift metrics
that answer the Stage-2 gate ("does rounding reorder the eviction ranking?").

Model-agnostic on purpose: everything here takes plain tensors, so it is unit-testable
without a VLM and the metrics layer re-runs on CPU from saved logs.

Method note: we use SHADOW RANKING. One real decode pass produces one cache; every arm
scores that same cache. Nothing is actually evicted. This structurally prevents the bug
that broke the earlier prototype's placebo arm (arms drawing independent randomness),
because every arm sees byte-identical q and k.
"""

import numpy as np
import torch

# ---------------------------------------------------------------- quantizers


def _asym_quant(x, bits, dim):
    """Asymmetric min/max fake-quant along `dim`. Returns (x_hat, step).

    step is broadcast to x's shape; error is ~U(-step/2, step/2) so Var = step^2/12.
    """
    levels = 2**bits - 1
    lo = x.amin(dim=dim, keepdim=True)
    hi = x.amax(dim=dim, keepdim=True)
    step = (hi - lo).clamp(min=1e-8) / levels
    x_hat = ((x - lo) / step).round().clamp(0, levels) * step + lo
    return x_hat, step.expand_as(x)


def quantize_keys(k, bits, scheme="kivi", group=32):
    """Fake-quantize a key cache. k: [..., seq, d] -> (k_hat, step) same shape.

    schemes (the three setups the kill test compares):
      per_channel_global : one step per channel over the whole cache. Sanity check --
                           the Jensen term is constant across tokens here, so it cannot
                           reorder anything. Any reordering seen is pure noise.
      kivi               : KIVI-style, per-channel within blocks of `group` tokens.
                           The realistic setting.
      per_token          : one step per token across channels.
    """
    if bits >= 16:
        return k, torch.zeros_like(k)
    if scheme == "per_channel_global":
        return _asym_quant(k, bits, dim=-2)
    if scheme == "per_token":
        return _asym_quant(k, bits, dim=-1)
    if scheme != "kivi":
        raise ValueError(f"unknown scheme {scheme}")

    seq = k.shape[-2]
    pad = (-seq) % group
    if pad:  # replicate the tail so the last block is full; truncate after
        k = torch.cat([k, k[..., -1:, :].expand(*k.shape[:-2], pad, k.shape[-1])], dim=-2)
    blocked = k.unflatten(-2, (-1, group))          # [..., nblk, group, d]
    q_hat, step = _asym_quant(blocked, bits, dim=-2)  # per-channel within block
    return q_hat.flatten(-3, -2)[..., :seq, :], step.flatten(-3, -2)[..., :seq, :]


# ------------------------------------------------------------------- scoring


def attn_scores(q, k, scale=None):
    """Pre-softmax scores. q: [H, d] (one decode step), k: [H, S, d] -> [H, S].
    scale: the model's own scaling factor; falls back to 1/sqrt(d) (synthetic tests only)."""
    return torch.einsum("hd,hsd->hs", q, k) * (scale if scale is not None else q.shape[-1] ** -0.5)


def jensen_term(q, step):
    """Per-score Jensen bias sigma^2_i (Tuncer et al. 2026, ported to decoding).

    s_i = q . k_i / sqrt(d). Quantization gives k_hat = k + e, e ~ U(-step/2, step/2),
    so the error in s_i has variance sigma^2_i = sum_c q_c^2 step_ic^2 / 12 / d.
    Softmax convexity inflates a rounded key's weight by exp(sigma^2/2), hence the
    correction s_corrected = s_hat - sigma^2 / 2.

    q: [H, d], step: [H, S, d] -> [H, S]
    """
    d = q.shape[-1]
    return torch.einsum("hd,hsd->hs", q**2, step**2) / (12.0 * d)


def correct(s_hat, sigma2):
    """Remove the predicted systematic bias, leaving the noise part."""
    return s_hat - 0.5 * sigma2


# ------------------------------------------------------------------- metrics


def _bottom_k(scores, k, mask=None):
    """Indices of the k lowest-scoring tokens -- i.e. who eviction would delete."""
    s = scores.clone()
    if mask is not None:
        s[~mask] = float("inf")  # excluded tokens can never be the lowest
        k = min(k, int(mask.sum()))
    if k <= 0:
        return torch.empty(0, dtype=torch.long, device=s.device)
    return s.topk(k, largest=False).indices


def jaccard(a, b):
    a, b = set(a.tolist()), set(b.tolist())
    return 1.0 if not (a | b) else len(a & b) / len(a | b)


def overlap(a, b):
    """Fraction of set a also in b. The gate metric is top-K overlap < 0.95."""
    a, b = set(a.tolist()), set(b.tolist())
    return 1.0 if not a else len(a & b) / len(a)


def spearman(x, y):
    """Rank correlation, no scipy needed."""
    def rank(v):
        idx = v.argsort()
        r = torch.empty_like(idx, dtype=torch.float64)
        r[idx] = torch.arange(len(v), dtype=torch.float64, device=v.device)
        return r
    rx, ry = rank(x.double()), rank(y.double())
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = (rx.norm() * ry.norm()).clamp(min=1e-12)
    return float((rx @ ry) / denom)


def rank_displacement(x, y):
    """Mean |rank change| between two scorings, in token positions."""
    def rank(v):
        idx = v.argsort()
        r = torch.empty_like(idx, dtype=torch.float64)
        r[idx] = torch.arange(len(v), dtype=torch.float64, device=v.device)
        return r
    return float((rank(x.double()) - rank(y.double())).abs().mean())


def _compare_head(scores_fp, scores_q, budget, mask, topk):
    """One head. Eviction policies (H2O, TOVA) rank per head, so this is the unit."""
    n = int(mask.sum()) if mask is not None else scores_fp.shape[-1]
    topk = topk if topk is not None else max(1, n - budget)

    ev_fp, ev_q = _bottom_k(scores_fp, budget, mask), _bottom_k(scores_q, budget, mask)
    keep_fp = _bottom_k(-scores_fp, topk, mask)
    keep_q = _bottom_k(-scores_q, topk, mask)

    if mask is not None:  # rank stats only over the tokens in this modality
        scores_fp, scores_q = scores_fp[mask], scores_q[mask]

    return {
        "n": n,
        "evict_jaccard": jaccard(ev_fp, ev_q),     # who gets deleted
        "topk_overlap": overlap(keep_fp, keep_q),  # the gate: < 0.95 at INT4?
        "spearman": spearman(scores_fp, scores_q),
        "rank_disp": rank_displacement(scores_fp, scores_q),
    }


def compare(scores_fp, scores_q, budget, mask=None, topk=None):
    """Compare a full-precision against a rounded scoring of the SAME cache.

    scores_*: [H, S] (or [S] for one head). `mask` ([S] bool) restricts to one
    modality, which is how we get the visual-vs-text split the gap turns on.
    Metrics are computed per head and averaged, since eviction ranks per head.

    budget : how many tokens eviction would delete -> bottom-k agreement
    topk   : how many it would keep -> top-k overlap (the gate metric)
    """
    if scores_fp.dim() == 1:
        return _compare_head(scores_fp, scores_q, budget, mask, topk)
    per_head = [_compare_head(scores_fp[h], scores_q[h], budget, mask, topk)
                for h in range(scores_fp.shape[0])]
    return {k: float(np.mean([r[k] for r in per_head])) for k in per_head[0]}


# ---------------------------------------------------------------- self-check


def _demo():
    """Runnable check of the whole metrics path on synthetic data.

    Asserts the three properties the real experiment depends on:
      1. PLACEBO  - scoring the same cache twice at FP16 agrees exactly (1.000).
      2. ROUNDTRIP- quantization error stays inside the half-step bound.
      3. DOSE     - INT2 damages the ranking strictly more than INT4.
    """
    torch.manual_seed(0)
    H, S, d, budget = 4, 256, 64, 32
    q = torch.randn(H, d)
    k = torch.randn(H, S, d)

    # 1. placebo: identical inputs must give identical rankings
    s = attn_scores(q, k)
    p = compare(s, s.clone(), budget)
    assert p["evict_jaccard"] == 1.0 and p["topk_overlap"] == 1.0, p
    assert abs(p["spearman"] - 1.0) < 1e-9 and p["rank_disp"] == 0.0, p

    # 2. round-trip error bounded by half a step, for every scheme
    for scheme in ("per_channel_global", "kivi", "per_token"):
        for bits in (8, 4, 2):
            kh, st = quantize_keys(k, bits, scheme)
            assert kh.shape == k.shape and st.shape == k.shape, scheme
            assert ((k - kh).abs() <= st / 2 + 1e-5).all(), (scheme, bits)

    # 3. dose response: more aggressive rounding must reorder more
    res = {}
    for bits in (8, 4, 2):
        kh, st = quantize_keys(k, bits, "kivi")
        res[bits] = compare(s, attn_scores(q, kh), budget)
    assert res[2]["rank_disp"] > res[4]["rank_disp"] > res[8]["rank_disp"], res
    assert res[2]["evict_jaccard"] < res[4]["evict_jaccard"], res

    # 4. the Jensen correction must be rank-neutral under a global per-channel step
    #    (constant sigma^2 across tokens cannot reorder) -- the sanity check the
    #    Top-4 gaps note calls for.
    kh, st = quantize_keys(k, 2, "per_channel_global")
    sq = attn_scores(q, kh)
    sig = jensen_term(q, st)
    assert sig.std(dim=-1).max() < 1e-6, "global step should give constant sigma^2"
    assert abs(spearman(sq[0], correct(sq, sig)[0]) - 1.0) < 1e-9

    print("self-check passed")
    for bits in (8, 4, 2):
        r = res[bits]
        print(f"  INT{bits}: evict_jaccard={r['evict_jaccard']:.3f} "
              f"topk_overlap={r['topk_overlap']:.3f} "
              f"spearman={r['spearman']:.4f} rank_disp={r['rank_disp']:.2f}")


if __name__ == "__main__":
    _demo()
