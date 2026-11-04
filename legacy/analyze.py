"""Aggregate the shadow-ranking logs into the Stage-2 gate answer.

The gate, fixed in advance: does INT4 push visual top-K overlap below 0.95?
Also reports the visual-vs-text split (the modality claim) and the bias-vs-noise
split (does subtracting the predicted Jensen term recover the ranking?).

Runs on CPU from saved JSON, no model needed -- NFR 'portability'.
"""

import json, sys
from pathlib import Path

import numpy as np

GATE = 0.95


def load(d):
    rows = []
    for f in sorted(Path(d).glob("item_*.json")):
        item = json.loads(f.read_text())
        for r in item["records"]:
            r["item"] = item["item"]
            rows.append(r)
    return rows


def agg(rows, tag, metric):
    v = [r[tag][metric] for r in rows if tag in r]
    return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (float("nan"),) * 2 + (0,)


def boot_ci(vals, n=2000, seed=0):
    """Bootstrap 95% CI of the mean -- NFR 'transparency' asks for intervals."""
    if not vals:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    a = np.asarray(vals, dtype=float)
    m = rng.choice(a, size=(n, len(a)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main(d="results/ranking"):
    rows = load(d)
    if not rows:
        print("no records in", d); return
    items = len({r["item"] for r in rows})
    layers = sorted({r["layer"] for r in rows})
    print(f"{len(rows)} observations | {items} items | layers {layers} | "
          f"steps up to {max(r['step'] for r in rows)}")
    print(f"cache {rows[0]['S']}-{rows[-1]['S']} tokens, "
          f"{rows[0]['n_visual']} visual, budget {rows[0]['budget']}\n")

    for scheme in ("kivi", "per_channel_global"):
        if not any(k.startswith(scheme) for r in rows for k in r):
            continue
        print(f"=== scheme: {scheme} ===")
        hdr = f"{'arm':<12}{'split':<8}{'topk_ovl':>10}{'evict_jac':>11}{'spearman':>10}{'rank_disp':>11}"
        print(hdr); print("-" * len(hdr))
        for arm in ("B1_int4", "B2_int2"):
            for split in ("visual", "text"):
                tag = f"{scheme}|{arm}|{split}"
                vals = [r[tag]["topk_overlap"] for r in rows if tag in r]
                if not vals:
                    continue
                lo, hi = boot_ci(vals)
                print(f"{arm:<12}{split:<8}"
                      f"{np.mean(vals):>10.4f}"
                      f"{agg(rows, tag, 'evict_jaccard')[0]:>11.4f}"
                      f"{agg(rows, tag, 'spearman')[0]:>10.4f}"
                      f"{agg(rows, tag, 'rank_disp')[0]:>11.2f}")
                if split == "visual":
                    gate = "PASS (effect present)" if np.mean(vals) < GATE else "no effect at this threshold"
                    print(f"{'':<20}  gate: visual top-K overlap {np.mean(vals):.4f} "
                          f"CI [{lo:.4f},{hi:.4f}] vs {GATE} -> {gate}")
        # bias vs noise: does removing the predicted Jensen term restore the ranking?
        print(f"\n  bias-vs-noise (does subtracting the Jensen term help?)")
        for arm in ("B1_int4", "B2_int2"):
            for split in ("visual", "text"):
                t = f"{scheme}|{arm}|{split}"
                raw = [r[t]["topk_overlap"] for r in rows if t in r]
                cor = [r[t + "|corrected"]["topk_overlap"] for r in rows if t + "|corrected" in r]
                if raw and cor:
                    d_ = np.mean(cor) - np.mean(raw)
                    share = d_ / max(1e-9, 1 - np.mean(raw)) * 100
                    print(f"    {arm:<10}{split:<8} raw {np.mean(raw):.4f} -> corrected "
                          f"{np.mean(cor):.4f}  (recovers {share:5.1f}% of the gap)")
        # magnitude of the predicted bias, visual vs text
        for arm in ("B1_int4", "B2_int2"):
            sv = [r[f"{scheme}|{arm}|sigma2_visual"] for r in rows
                  if f"{scheme}|{arm}|sigma2_visual" in r]
            st = [r[f"{scheme}|{arm}|sigma2_text"] for r in rows
                  if f"{scheme}|{arm}|sigma2_text" in r]
            ls = [r[f"{scheme}|{arm}|logit_scale"] for r in rows
                  if f"{scheme}|{arm}|logit_scale" in r]
            if sv and st:
                ratio = np.mean(sv) / max(1e-12, np.mean(st))
                print(f"    {arm:<10}sigma^2 visual {np.mean(sv):.3e} vs text "
                      f"{np.mean(st):.3e}  ratio {ratio:.2f}x, "
                      f"{np.mean(sv)/max(1e-12,np.mean(ls))*100:.3f}% of logit scale")
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/ranking")
