"""Stage 1 batch verification: every check on every test image (STAGES.md, D1.4).

    python3 test_stage1.py          # ~10 minutes; needs the GPU free (stop app.py first)

Images: all pilot photos at original size, plus three edge cases built to exercise each
branch of Qwen's resize rule. Writes STAGE1_REPORT.md and results_stage1/*.json.
"""

import json, time
from pathlib import Path

import numpy as np
from PIL import Image

import run_timeline as rt
import stages
from cfg import CFG


def edge_cases(base):
    """Three images that force the shrink, enlarge and extreme-shape branches."""
    w, h = base.size
    return [
        ("large photo (shrink branch)", base.resize((w * 6, h * 6), Image.BICUBIC)),
        ("tiny image (enlarge branch)", base.resize((40, 30), Image.BICUBIC)),
        ("panorama 8:1 (shape + shrink)", base.resize((2400, 300), Image.BICUBIC)),
    ]


def main():
    model, proc, dev = rt.load_model()
    S = stages.Stage1(model, proc, dev)
    pilots = rt.build_items(len(CFG["pilot"]["items"]))
    images = [(f"pilot {it['id']}", it["image"]) for it in pilots] + edge_cases(pilots[5]["image"])

    out = Path("results_stage1"); out.mkdir(exist_ok=True)
    rows, all_margins = [], []
    for name, img in images:
        t0 = time.time()
        d = S.run(img)
        json.dumps(d)                    # exactly what the page receives: must serialise cleanly
        res = None
        for (r, c) in S.squares:
            res = S.occlude(r, c, record=True)
            json.dumps(res)
        checks = {c["id"]: c for c in d["checks"] + res["final"]["checks"]}
        margins = [t["margin"] for t in S.trials.values() if t["finite"]]   # 1.5b, mixing on
        routing_other = [t["routing_other_max"] for t in S.trials.values()]
        all_margins += margins
        rec = {"name": name, "size": [img.width, img.height], "grid": [d["token"]["rows"], d["token"]["cols"]],
               "tokens": d["token"]["n"], "checks": checks, "ring": res["final"]["ring"],
               "margins": margins, "weakest": res["final"]["weakest"], "routing_other": routing_other,
               "secs": round(time.time() - t0, 1), "config": rt.snapshot()}
        (out / (name.split(" (")[0].replace(" ", "_").replace(":", "x") + ".json")).write_text(json.dumps(rec))
        ok = all(c["pass"] for c in checks.values() if c["pass"] is not None)   # 1.5b is measured, not judged
        rows.append(rec)
        print(f"{'PASS' if ok else 'FAIL'}  {name:32s} {img.width}x{img.height} -> {rec['grid'][1]}x{rec['grid'][0]} = "
              f"{rec['tokens']} tokens; 1.5a {checks['1.5a']['detail'].split(':')[0]}; "
              f"1.5b own token top in {sum(t['correct'] for t in S.trials.values())}/{len(S.trials)} "
              f"({rec['secs']}s)", flush=True)

    # ---- report
    def key(i):                                                     # "1.5a" sorts after "1.5"
        return (float(i.rstrip("ab")), i)
    ids = sorted({i for r in rows for i in r["checks"]}, key=key)   # every check that ran
    judged = [i for i in ids if rows[0]["checks"][i]["pass"] is not None]
    n_pass = {i: sum(bool(r["checks"][i]["pass"]) for r in rows) for i in judged}
    m = np.array(all_margins)
    lines = ["# Stage 1 batch report", "",
             f"Run {time.strftime('%Y-%m-%d %H:%M')} on Qwen2.5-VL-3B-Instruct (local, MPS). "
             f"Command: `python3 test_stage1.py`. Pass criteria: STAGES.md, stage 1.", "",
             f"**{len(rows)} images; every pass/fail check passed on "
             f"{sum(all(r['checks'][i]['pass'] for i in judged) for r in rows)} of {len(rows)}.**", "",
             "| Check | Images passing |", "|---|---|"]
    names = {c["id"]: c["name"] for c in rows[0]["checks"].values()}
    lines += [f"| {i} {names[i]} | {n_pass[i]} / {len(rows)} |" for i in judged]
    ro = np.array([x for r in rows for x in r["routing_other"]])
    lines += ["", "## 1.5a routing (mixing off)", "",
              f"{len(ro)} squares. Largest change of any token OTHER than the changed square's, "
              f"relative to the mean token size: max {np.nanmax(ro):.1e} (pass < 1e-3).", "",
              "## 1.5b mixing (encoder unmodified) — measured, not judged", "",
              f"{len(m)} squares with finite outputs. Margin = response of the changed square's token "
              f"÷ the next-largest response.", "",
              f"- minimum {m.min():.2f}×, 5th percentile {np.percentile(m, 5):.2f}×, "
              f"median {np.median(m):.2f}×, maximum {m.max():.2f}×",
              f"- squares with margin below 1.10× (near-ties): {int((m < 1.10).sum())} of {len(m)}", "",
              "## Per image", "", "| Image | Size | Tokens | Pass/fail checks | 1.5b min margin | Weakest square |",
              "|---|---|---|---|---|---|"]
    for r in rows:
        ok = all(c["pass"] for c in r["checks"].values() if c["pass"] is not None)
        lines.append(f"| {r['name']} | {r['size'][0]} × {r['size'][1]} | {r['tokens']} ({r['grid'][1]} × {r['grid'][0]}) | "
                     f"{'PASS' if ok else '**FAIL**'} | "
                     + (f"{min(r['margins']):.2f}× | row {r['weakest']['r']}, col {r['weakest']['c']} |"
                        if r["margins"] else "— | all edits overflowed |"))
    fails = [(r["name"], c) for r in rows for c in r["checks"].values() if c["pass"] is False]
    if fails:
        lines += ["", "## Failures", ""] + [f"- **{n}**, check {c['id']}: {c['detail']}" for n, c in fails]
    Path("STAGE1_REPORT.md").write_text("\n".join(lines) + "\n")
    print("\nwrote STAGE1_REPORT.md")


if __name__ == "__main__":
    main()
