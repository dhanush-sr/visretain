"""Stage 2 batch verification: every check, plus the negative controls, on every test image.

    python -m tests.test_stage2     # ~10 minutes; needs the GPU free (stop app.py first)

Same images as stage 1 (all pilot photos at original size, plus the three edge cases), each
with its pilot question. Writes reports/STAGE2_REPORT.md and results/stage2/*.json.
"""

import json, time
from pathlib import Path

import numpy as np

from visretain import run_timeline as rt
from visretain import stages
from visretain.cfg import CFG, RESULTS, REPORTS
from tests.test_stage1 import edge_cases


def main():
    model, proc, dev = rt.load_model()
    S2 = stages.Stage2(model, proc, dev)
    pilots = rt.build_items(len(CFG["pilot"]["items"]))
    images = [(f"pilot {it['id']}", it["image"], it["question"]) for it in pilots] + \
             [(n, im, CFG["generation"]["default_question"]) for n, im in edge_cases(pilots[5]["image"])]

    out = RESULTS / "stage2"; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, img, q in images:
        t0 = time.time()
        d = S2.run(img, q)
        json.dumps(d)                                   # what the page receives must serialise
        checks = {c["id"]: c for c in d["checks"]}
        rec = {"name": name, "question": q, "n_tokens": len(d["tokens"]), "checks": checks,
               "split": d["split"], "secs": round(time.time() - t0, 1), "config": rt.snapshot()}
        (out / (name.split(" (")[0].replace(" ", "_").replace(":", "x") + ".json")).write_text(json.dumps(rec))
        rows.append(rec)
        ok = all(c["pass"] for c in checks.values())
        print(f"{'PASS' if ok else 'FAIL'}  {name:32s} {rec['n_tokens']} answer tokens; "
              + " ".join(f"{i}:{'ok' if c['pass'] else 'FAIL'}" for i, c in checks.items())
              + f" ({rec['secs']}s)", flush=True)

    ids = list(rows[0]["checks"])
    names = {i: rows[0]["checks"][i]["name"] for i in ids}
    lines = ["# Stage 2 batch report", "",
             f"Run {time.strftime('%Y-%m-%d %H:%M')} on Qwen2.5-VL-3B-Instruct (local, MPS). "
             f"Command: `python -m tests.test_stage2`. Pass criteria: STAGES.md, stage 2.", "",
             f"**{len(rows)} images; every check passed on "
             f"{sum(all(c['pass'] for c in r['checks'].values()) for r in rows)} of {len(rows)}.**", "",
             "| Check | Images passing |", "|---|---|"]
    lines += [f"| {i} {names[i]} | {sum(r['checks'][i]['pass'] for r in rows)} / {len(rows)} |" for i in ids]
    lines += ["", "## Per image", "", "| Image | Answer tokens | " + " | ".join(ids) + " |",
              "|---|---|" + "---|" * len(ids)]
    for r in rows:
        lines.append(f"| {r['name']} | {r['n_tokens']} | " +
                     " | ".join("PASS" if r["checks"][i]["pass"] else "**FAIL**" for i in ids) + " |")
    lines += ["", "## Evidence per check (first image)", ""]
    lines += [f"- **{i}**: {rows[0]['checks'][i]['detail']}" for i in ids]
    # where attention goes, averaged over images, a few layers (measured, not judged)
    layers = sorted({int(L) for r in rows for L in r["split"]})
    lines += ["", "## Where attention goes (measured, averaged over images)", "",
              "| Layer | First token | Prompt text | Image | Answer so far |", "|---|---|---|---|---|"]
    for L in layers[::5] + ([layers[-1]] if layers[-1] not in layers[::5] else []):
        m = np.mean([r["split"][L] if L in r["split"] else r["split"][str(L)] for r in rows], axis=0)
        lines.append(f"| {L} | " + " | ".join(f"{v*100:.0f}%" for v in m) + " |")
    fails = [(r["name"], c) for r in rows for c in r["checks"].values() if not c["pass"]]
    if fails:
        lines += ["", "## Failures", ""] + [f"- **{n}**, check {c['id']}: {c['detail']}" for n, c in fails]
    (REPORTS / "STAGE2_REPORT.md").write_text("\n".join(lines) + "\n")
    print("\nwrote reports/STAGE2_REPORT.md")


if __name__ == "__main__":
    main()
