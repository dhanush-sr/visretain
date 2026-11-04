"""Build the OFFLINE stages page: real stage-1 and stage-2 runs on one image, replayed without a model.

    python -m visretain.build_stages              # pilot image from config (default: the skier)
    python -m visretain.build_stages --pilot 3    # another pilot image (index into config)

Every square's causal test is precomputed, so "click any square" works offline too.
Takes ~10 minutes (one vision-encoder pass per square). Needs the GPU free.
"""

import argparse, json, time
from pathlib import Path

from . import run_timeline as rt
from . import stages
from . import stages_page as sp
from .cfg import CFG, DEMO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=CFG["stage1"]["offline_pilot"])
    ap.add_argument("--out", default=str(DEMO / "stages.html"))
    a = ap.parse_args()

    model, proc, dev = rt.load_model()
    S = stages.Stage1(model, proc, dev)
    item = rt.build_items(a.pilot + 1)[a.pilot]
    t0 = time.time()
    d = S.run(item["image"])
    trials = [S.occlude(r, c, record=True) for (r, c) in S.squares]     # the recorded 24
    R, C = S.grid_rc
    allmaps = []
    for k in range(R * C):                                               # every square, for clicks
        allmaps.append(S.occlude(k // C, k % C))
        if k % 50 == 0:
            print(f"  square {k} / {R*C}  ({time.time()-t0:.0f}s)", flush=True)
    # stage 2: the same image, its pilot question, every layer's attention maps
    S2 = stages.Stage2(model, proc, dev)
    d2 = S2.run(item["image"], item["question"])
    layers2 = {L: S2.layer_map(L) for L in d2["layers"]}
    print(f"  stage 2: {sum(c['pass'] for c in d2['checks'])}/{len(d2['checks'])} checks passed", flush=True)
    data = {"stage1": d, "trials": trials, "allmaps": allmaps, "stage2": d2, "stage2_layers": layers2}
    out = Path(a.out); out.parent.mkdir(exist_ok=True)
    out.write_text(sp.PAGE.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
                   .replace("__LIVE__", "false"))
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
