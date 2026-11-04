"""The decisive test: does the eviction-ranking effect survive ACCUMULATED scoring?

Instantaneous scoring is TOVA-like. H2O/EMA rank by attention summed over decode
steps, and accumulation averages away per-step variance -- which is exactly the
mechanism we measured. If the effect dies here, the premise is weak.

Statistics are PER ITEM (n = items), not per observation. Observations from the same
item at neighbouring decode steps are nearly identical, so pooling them inflates the
sample size and produces meaninglessly tight intervals.
"""

import json, sys
from pathlib import Path

import numpy as np

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None


def per_item(d, key, metric):
    """One value per item: mean over layers of the accumulated metric."""
    out = []
    for f in sorted(Path(d).glob("item_*.json")):
        it = json.loads(f.read_text())
        v = [r[key][metric] for r in it.get("accumulated", []) if key in r]
        if v:
            out.append(float(np.mean(v)))
    return np.array(out)


def ci(a, n=10000, seed=0):
    if len(a) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    m = rng.choice(a, size=(n, len(a)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main(d="results_acc"):
    print(f"=== ACCUMULATED (H2O/EMA-style) scoring :: {d} ===")
    print("statistics per item; n = number of items\n")

    hdr = f"{'arm':<12}{'split':<8}{'n':>3}{'evict_jac':>11}{'95% CI':>18}{'rank_disp':>11}"
    print(hdr); print("-" * len(hdr))
    store = {}
    for arm in ("int4", "int2"):
        for split in ("visual", "text"):
            k = f"acc|{arm}|{split}"
            ej = per_item(d, k, "evict_jaccard")
            rd = per_item(d, k, "rank_disp")
            if not len(ej):
                continue
            store[(arm, split)] = ej
            lo, hi = ci(ej)
            print(f"{arm:<12}{split:<8}{len(ej):>3}{ej.mean():>11.4f}"
                  f"{f'[{lo:.3f}, {hi:.3f}]':>18}{rd.mean():>11.2f}")

    print("\n--- is the eviction set changed at all? (paired vs perfect agreement) ---")
    for (arm, split), ej in store.items():
        if wilcoxon is None or len(ej) < 6:
            print(f"  {arm:<6}{split:<8} mean disagreement "
                  f"{(1-ej.mean())*100:5.1f}%  (n={len(ej)}, no test)")
            continue
        # H0: eviction sets identical (Jaccard = 1). One-sided: agreement < 1.
        try:
            st, p = wilcoxon(ej - 1.0, alternative="less")
            print(f"  {arm:<6}{split:<8} mean disagreement {(1-ej.mean())*100:5.1f}%"
                  f"   Wilcoxon p = {p:.4f}  {'SIGNIFICANT' if p < 0.05 else 'n.s.'}")
        except ValueError as e:
            print(f"  {arm:<6}{split:<8} {e}")

    print("\n--- is visual hit harder than text? (paired, same items) ---")
    for arm in ("int4", "int2"):
        v, t = store.get((arm, "visual")), store.get((arm, "text"))
        if v is None or t is None or len(v) != len(t):
            continue
        diff = t - v  # positive => visual agrees LESS => visual disrupted more
        lo, hi = ci(diff)
        line = (f"  {arm:<6} text-minus-visual Jaccard {diff.mean():+.4f} "
                f"[{lo:+.3f}, {hi:+.3f}]")
        if wilcoxon is not None and len(diff) >= 6:
            try:
                st, p = wilcoxon(diff)
                line += f"  p = {p:.4f}"
            except ValueError:
                pass
        print(line)

    print("\n--- does the Jensen correction recover anything? ---")
    for arm in ("int4", "int2"):
        raw = per_item(d, f"acc|{arm}|visual", "evict_jaccard")
        cor = per_item(d, f"acc|{arm}_corr|visual", "evict_jaccard")
        if len(raw) and len(cor):
            gap = 1 - raw.mean()
            rec = (cor.mean() - raw.mean()) / max(1e-9, gap) * 100
            print(f"  {arm:<6}visual  raw {raw.mean():.4f} -> corrected "
                  f"{cor.mean():.4f}   recovers {rec:+5.1f}% of the gap")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results_acc")
