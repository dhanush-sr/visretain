"""Build the VisRetain demo: one self-contained HTML file, opens offline in any browser.

    python3 build_demo.py                       # reads results_timeline/, results_acc/
    python3 build_demo.py --runs results_mine   # e.g. a run on your own image

Everything shown comes from real Qwen2.5-VL-3B runs. Eviction is counterfactual (the
model is not fed the reduced cache) and evidence is attention-defined; the page says so.
"""

import argparse, json
from pathlib import Path

import numpy as np

from cfg import CFG
# the evicting arms, from the configured bit widths: [16, 4, 2] -> fp16, int4, int2
ARMS = [("fp16" if b >= 16 else f"int{b}") for b in CFG["quant"]["bits"]]


def arm_meta():
    """Labels and colours for the page, in arm order."""
    colours = ["#0f766e", "#2563eb", "#c026d3", "#ea580c", "#4f46e5"]
    out = []
    for i, (a, b) in enumerate(zip(ARMS, CFG["quant"]["bits"])):
        label = f"FP16 · no rounding" if b >= 16 else f"INT{b} · rounded to {b} bits"
        out.append({"key": a, "bits": b, "label": label, "color": colours[i % len(colours)]})
    return out


def to_item(r):
    """One run record (from run_timeline.run_item) -> one demo item, or None."""
    if "image_b64" not in r:
        return None
    if True:
        layers = {}
        for L in r["layers"]:
            if "need_curve" not in L:
                continue
            layers[str(L["layer"])] = {
                "deaths": {a: L["curves"][a] for a in ARMS},
                "gun": L["gun_local"],
                "need": [round(x, 5) for x in L["need_curve"]],
                "stats": {a: {k: L[a][k] for k in ("gone_when_needed", "gun_alive_frac",
                                                   "visual_alive_frac")} for a in ARMS},
            }
        if not layers:
            return None
        return {"id": r["item"], "question": r["question"], "grid": r["grid"],
                "img": r["image_b64"], "tokens": r["tokens"],
                "steps": r["n_steps"], "budget": r["budget"],
                "n_visual": r["n_visual"], "need_step": r["layers"][0]["need_step"],
                "layers": layers}


def load_runs(d):
    items = []
    for f in sorted(Path(d).glob("item_*.json")):
        it = to_item(json.loads(f.read_text()))
        if it:
            items.append(it)
    return items


def summarise_timeline(d):
    """Paired per item: share of needed evidence already gone when needed."""
    per = {}
    for f in sorted(Path(d).glob("item_*.json")):
        r = json.loads(f.read_text())
        if r.get("too_short") or r["n_steps"] < CFG["evidence"]["min_steps"]:
            continue  # short answers have no late window worth measuring
        vals = {a: np.mean([L[a]["gone_when_needed"] for L in r["layers"]]) for a in ARMS}
        per[r["item"]] = vals
    if not per:
        return None
    out = {"n": len(per)}
    for a in ARMS:
        out[a] = float(np.mean([v[a] for v in per.values()]))
    for a in ARMS[1:]:
        diffs = np.array([v[a] - v["fp16"] for v in per.values()])
        out[a + "_delta"] = float(diffs.mean())
        out[a + "_less"] = int((diffs < 0).sum())
        try:
            from scipy.stats import wilcoxon
            out[a + "_p"] = float(wilcoxon(diffs).pvalue) if len(diffs) >= 6 else None
        except Exception:
            out[a + "_p"] = None
    return out


def summarise_ranking(d):
    """Rung 1: how much of the eviction set changes, from the accumulated run."""
    rows = {}
    files = sorted(Path(d).glob("item_*.json"))
    if not files:
        return None
    for arm in ARMS[1:]:
        for split in ("visual", "text"):
            vals = []
            for f in files:
                r = json.loads(f.read_text())
                v = [L[f"acc|{arm}|{split}"]["evict_jaccard"]
                     for L in r.get("accumulated", []) if f"acc|{arm}|{split}" in L]
                if v:
                    vals.append(np.mean(v))
            if vals:
                rows[f"{arm}_{split}"] = float(np.mean(vals))
    rows["n"] = len(files)
    return rows


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VisRetain: does rounding the KV cache change what a VLM forgets?</title>
<style>
:root{
  --bg:#f6f7f9; --card:#fff; --ink:#111827; --sub:#4b5563; --mute:#9ca3af; --line:#e5e7eb;
  --ev:#f59e0b; --lost:#dc2626; --kept:#16a34a;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif}
header{background:var(--card);border-bottom:1px solid var(--line);padding:18px 32px;
  display:flex;align-items:center;gap:24px}
header h1{font-size:20px;margin:0;font-weight:650;letter-spacing:-.01em}
header p{margin:2px 0 0;color:var(--sub);font-size:13px}
.tabs{margin-left:auto;display:flex;gap:4px;background:var(--bg);padding:4px;border-radius:10px}
.tabs button{border:0;background:transparent;padding:8px 16px;border-radius:7px;font:inherit;
  font-size:14px;color:var(--sub);cursor:pointer}
.tabs button.on{background:var(--card);color:var(--ink);box-shadow:0 1px 2px rgba(0,0,0,.08);font-weight:600}
main{max-width:1440px;margin:0 auto;padding:24px 32px 48px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px}
.row{display:flex;gap:20px;align-items:stretch}
.controls{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:20px}
select,button.ctl{font:inherit;font-size:14px;padding:8px 12px;border:1px solid var(--line);
  border-radius:8px;background:var(--card);color:var(--ink);cursor:pointer}
button.play{background:var(--ink);color:#fff;border-color:var(--ink);min-width:92px}
input[type=range]{flex:1;min-width:240px;accent-color:var(--ink)}
.stepnum{font-variant-numeric:tabular-nums;color:var(--sub);min-width:120px;text-align:right}
.qa{display:grid;grid-template-columns:auto 1fr;gap:6px 14px;margin-bottom:20px}
.qa b{color:var(--mute);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.06em;padding-top:2px}
.answer{min-height:48px;color:var(--ink)} .answer .cur{background:#fde68a;border-radius:3px}
.answer .rest{color:#d1d5db}
.phases{display:flex;gap:6px;margin-bottom:20px}
.phase{flex:1;padding:10px 12px;border-radius:10px;background:var(--card);border:1px solid var(--line);
  font-size:13px;color:var(--mute)}
.phase b{display:block;font-size:11px;letter-spacing:.06em;text-transform:uppercase}
.phase.on{border-color:var(--ink);color:var(--ink)}
.panels{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin-bottom:20px}
.panel h3{margin:0 0 10px;font-size:15px;display:flex;align-items:center;gap:8px}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block}
.panel h3 span.tag{margin-left:auto;font-weight:500;font-size:12px;color:var(--sub)}
canvas.grid{width:100%;border-radius:8px;display:block;background:#000}
.meter{display:flex;justify-content:space-between;margin-top:10px;font-size:13px;color:var(--sub)}
.meter strong{color:var(--ink);font-variant-numeric:tabular-nums}
.bigline{font-size:13px;margin-top:6px;font-variant-numeric:tabular-nums}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:20px}
.charts h4{margin:0 0 4px;font-size:14px} .charts p{margin:0 0 10px;color:var(--sub);font-size:13px}
svg{width:100%;height:200px;display:block}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;color:var(--sub);margin:4px 0 20px}
.legend i{display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:-2px;margin-right:6px}
.note{color:var(--sub);font-size:13px;margin-top:20px;border-top:1px solid var(--line);padding-top:14px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:10px 14px;text-align:right;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
th{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--mute);font-weight:600}
.verdict{font-weight:600} .yes{color:var(--lost)} .no{color:var(--sub)} .better{color:var(--kept)}
.headline{font-size:22px;line-height:1.35;margin:0 0 6px;font-weight:600;letter-spacing:-.01em}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:20px}
.hidden{display:none !important}
.livecard{display:grid;grid-template-columns:220px 1fr auto;gap:20px;align-items:center;margin-bottom:20px}
.drop{border:2px dashed #cbd5e1;border-radius:12px;height:130px;display:flex;align-items:center;justify-content:center;
  text-align:center;color:var(--sub);font-size:13px;cursor:pointer;background:#fafbfc;overflow:hidden;position:relative}
.drop.over{border-color:var(--ink);background:#f1f5f9}
.drop img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover}
.livecard textarea{width:100%;height:70px;font:inherit;font-size:15px;padding:10px 12px;border:1px solid var(--line);border-radius:10px;resize:vertical}
.livecard .status{font-size:13px;color:var(--sub);margin-top:8px;min-height:20px}
.livecard .status.err{color:var(--lost)}
button.run{background:var(--ink);color:#fff;border:0;border-radius:10px;padding:14px 26px;font:inherit;font-size:15px;font-weight:600;cursor:pointer}
button.run:disabled{opacity:.4;cursor:default}
.warn{background:#fef3c7;border:1px solid #fcd34d;color:#92400e;border-radius:10px;padding:10px 14px;font-size:13px;margin-bottom:20px}
</style>
</head>
<body>
<header>
  <div>
    <h1>VisRetain: does rounding the KV cache change what a VLM forgets?</h1>
    <p>Qwen2.5-VL-3B &middot; real image, real question, real answer &middot; same memory budget in every panel; only the precision of the stored keys changes</p>
  </div>
  <div class="tabs"><button id="t-live" class="on">Live run</button><button id="t-res">Results</button></div>
</header>

<main>
<section id="live">
  <div class="card livecard hidden" id="livecard">
    <label class="drop" id="drop"><span id="droptext">Drop an image here<br>or click to choose</span>
      <input type="file" id="file" accept="image/*" hidden></label>
    <div>
      <textarea id="lq" placeholder="Ask a question about the image, e.g. Who is wearing the red shirt and what are they holding?"></textarea>
      <div class="status" id="lstatus">Runs on this machine with Qwen2.5-VL-3B. About a minute per image.</div>
    </div>
    <button class="run" id="lrun" disabled>Run live</button>
  </div>
  <div class="warn hidden" id="shortwarn">This answer was too short to have a meaningful "late" window, so the evidence numbers for it are not reliable. Ask for a longer, descriptive answer.</div>
  <div class="controls">
    <select id="item"></select>
    <select id="layer"></select>
    <button class="ctl play" id="play">&#9654; Play</button>
    <input type="range" id="step" min="0" value="0">
    <span class="stepnum" id="stepnum"></span>
    <label style="font-size:14px;color:var(--sub)"><input type="checkbox" id="diff" checked> highlight differences vs FP16</label>
  </div>

  <div class="card qa">
    <b>Question</b><div id="q"></div>
    <b>Answer</b><div class="answer" id="ans"></div>
  </div>
  <div style="height:20px"></div>

  <div class="phases" id="phases"></div>

  <div class="legend">
    <span><i style="background:#fff;border:1px solid #ccc"></i>still in memory</span>
    <span><i style="background:#1f2937"></i>evicted (thrown away)</span>
    <span><i style="background:transparent;border:3px solid var(--ev)"></i>evidence the answer needs later</span>
    <span><i style="background:transparent;border:3px solid var(--lost)"></i>evidence already thrown away</span>
    <span id="lg-diff"><i style="background:rgba(220,38,38,.55)"></i>thrown away here, kept in FP16 &nbsp; <i style="background:rgba(22,163,74,.55)"></i>kept here, thrown away in FP16</span>
  </div>

  <div class="panels" id="panels"></div>

  <div class="charts">
    <div class="card"><h4>When did the answer need the evidence?</h4>
      <p>Share of the model's attention on the evidence tokens, full cache, at each step. Shaded: the late window where it is needed.</p>
      <svg id="needsvg"></svg></div>
    <div class="card"><h4>How much of that evidence is still in memory?</h4>
      <p>Evidence tokens still cached at each step, per precision. Lines apart = rounding made a difference.</p>
      <svg id="survsvg"></svg></div>
  </div>

  <div class="note">How to read this. Each panel is the same image cut into the model's real visual tokens (one per 28&times;28 px). All three panels keep the same number of tokens; the only difference is whether the stored keys were rounded. Eviction is <b>counterfactual</b>: we record what each precision <i>would</i> throw away while the model answers from the full cache, so all three see identical inputs. "Evidence" is defined from the model itself: visual tokens it attends to a lot late in the answer but barely early on. Exact object boxes (GQA) replace this in a later phase.</div>
</section>

<section id="res" class="hidden"></section>
</main>

<script>
const DATA = __DATA__;
const LIVE = __LIVE__;
const ARMS = DATA.arms.map(a=>[a.key, a.label, a.color]);   // from config.json
const COL = k => DATA.arms.find(a=>a.key===k).color;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const $ = id => document.getElementById(id);
let it = DATA.items[0], layer = null, step = 0, timer = null;
const imgs = {};

// ---------- setup ----------
DATA.items.forEach((d,i)=>{
  const o=document.createElement("option"); o.value=i;
  o.textContent=`Image ${i+1}: ${d.question.length>60?d.question.slice(0,58)+"…":d.question}`;
  $("item").appendChild(o);
  const im=new Image(); im.src="data:image/jpeg;base64,"+d.img; imgs[i]=im;
});
ARMS.forEach(([k,label,c])=>{
  const p=document.createElement("div"); p.className="card panel";
  p.innerHTML=`<h3><span class="dot" style="background:${c}"></span>${label}<span class="tag" id="tag-${k}"></span></h3>
    <canvas class="grid" id="cv-${k}"></canvas>
    <div class="meter"><span>tokens in memory <strong id="mem-${k}"></strong></span><span>evidence kept <strong id="ev-${k}"></strong></span></div>`;
  $("panels").appendChild(p);
});
const PH=[["Prefill","image becomes visual tokens, all cached"],["Decode","each step: score tokens, evict the lowest"],
          ["Evidence needed","the answer now relies on the evidence"],["Measure","was it still there when needed?"]];
PH.forEach(([a,b],i)=>{const d=document.createElement("div");d.className="phase";d.id="ph"+i;d.innerHTML=`<b>${i+1} · ${a}</b>${b}`;$("phases").appendChild(d);});

function pickItem(i){
  it=DATA.items[i];
  $("layer").innerHTML="";
  Object.keys(it.layers).forEach(l=>{const o=document.createElement("option");o.value=l;o.textContent="Layer "+l+" of 36";$("layer").appendChild(o);});
  layer = it.layers[String(DATA.display_layer)] ? String(DATA.display_layer) : Object.keys(it.layers)[0];
  $("layer").value=layer;
  $("step").max=it.steps; setStep(0);
  $("q").textContent=it.question;
  const im=imgs[i]; if(!im.complete) im.onload=()=>draw();
}
$("item").onchange=e=>{stop();pickItem(+e.target.value)};
$("layer").onchange=e=>{layer=e.target.value;draw()};
$("step").oninput=e=>setStep(+e.target.value);
$("diff").onchange=()=>{draw()};
$("play").onclick=()=>timer?stop():play();
document.addEventListener("keydown",e=>{
  if(e.target.tagName==="SELECT")return;
  if(e.code==="Space"){e.preventDefault();timer?stop():play();}
  if(e.code==="ArrowRight")setStep(Math.min(it.steps,step+1));
  if(e.code==="ArrowLeft")setStep(Math.max(0,step-1));
});
function play(){ if(step>=it.steps)setStep(0);
  $("play").innerHTML="&#10074;&#10074; Pause";
  timer=setInterval(()=>{ if(step>=it.steps){stop();return;} setStep(step+1); },110); }
function stop(){ clearInterval(timer); timer=null; $("play").innerHTML="&#9654; Play"; }
function setStep(s){ step=s; $("step").value=s; draw(); }

// ---------- drawing ----------
const alive = (d,t) => d<0 || d>t;

function draw(){
  const L=it.layers[layer], [R,C]=it.grid, im=imgs[DATA.items.indexOf(it)];
  const gun=new Set(L.gun), nGun=L.gun.length, pending=!L.stats;
  $("shortwarn").classList.toggle("hidden", !it.too_short);
  $("stepnum").textContent = step===0 ? "prefill" : `step ${step} / ${it.steps}`;

  // answer stream
  const toks=it.tokens, upto=Math.min(step,toks.length);
  const esc=s=>s.replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
  $("ans").innerHTML = step===0 ? '<span style="color:var(--mute)">…the model has read the image and question; nothing generated yet</span>'
    : esc(toks.slice(0,upto-1).join(""))+'<span class="cur">'+esc(toks[upto-1]||"")+'</span>'+'<span class="rest">'+esc(toks.slice(upto).join(""))+'</span>';

  // phase strip
  const ph = step===0?0 : step<it.need_step?1 : step<it.steps?2 : 3;
  PH.forEach((_,i)=>$("ph"+i).classList.toggle("on",i===ph));

  const fp=L.deaths.fp16, showDiff=$("diff").checked;
  $("lg-diff").style.opacity=showDiff?1:.35;

  ARMS.forEach(([k])=>{
    const cv=$("cv-"+k), W=cv.clientWidth||400, H=Math.round(W*R/C), dpr=window.devicePixelRatio||1;
    if(cv.width!==W*dpr){cv.width=W*dpr;cv.height=H*dpr;cv.style.height=H+"px";}
    const g=cv.getContext("2d"); g.setTransform(dpr,0,0,dpr,0,0);
    if(im.complete) g.drawImage(im,0,0,W,H);
    const cw=W/C, ch=H/R, D=L.deaths[k];
    let inMem=0, evKept=0;
    for(let i=0;i<D.length;i++){
      const r=Math.floor(i/C), c=i%C, x=c*cw, y=r*ch, a=alive(D[i],step);
      if(a) inMem++;
      if(!a){ g.fillStyle="rgba(17,24,39,.86)"; g.fillRect(x,y,cw,ch); }
      if(showDiff && k!=="fp16"){
        const af=alive(fp[i],step);
        if(af && !a){ g.fillStyle="rgba(220,38,38,.55)"; g.fillRect(x,y,cw,ch); }
        if(!af && a){ g.fillStyle="rgba(22,163,74,.50)"; g.fillRect(x,y,cw,ch); }
      }
    }
    // grid lines, faint
    g.strokeStyle="rgba(255,255,255,.18)"; g.lineWidth=.5; g.beginPath();
    for(let c=1;c<C;c++){g.moveTo(c*cw,0);g.lineTo(c*cw,H);} for(let r=1;r<R;r++){g.moveTo(0,r*ch);g.lineTo(W,r*ch);} g.stroke();
    // evidence outlines on top
    L.gun.forEach(i=>{
      const r=Math.floor(i/C), c=i%C, a=alive(D[i],step); if(a) evKept++;
      g.lineWidth=2.5; g.strokeStyle = a ? css("--ev") : css("--lost");
      g.strokeRect(c*cw+1.5,r*ch+1.5,cw-3,ch-3);
    });
    $("mem-"+k).textContent=`${inMem} / ${D.length}`;
    if(pending){
      $("ev-"+k).textContent="after the answer"; $("ev-"+k).style.color="";
      $("tag-"+k).textContent="";
    } else {
      $("ev-"+k).textContent=`${evKept} / ${nGun}`;
      $("ev-"+k).style.color = evKept<nGun ? css("--lost") : "";
      $("tag-"+k).textContent=`lost before needed: ${(L.stats[k].gone_when_needed*100).toFixed(0)}%`;
    }
  });
  drawNeed(L); drawSurv(L);
}

function axes(svg,W,H,pad,maxY,fmtY){
  let s=`<rect x="${pad.l}" y="${pad.t}" width="${W-pad.l-pad.r}" height="${H-pad.t-pad.b}" fill="none"/>`;
  for(let j=0;j<=4;j++){const y=pad.t+(H-pad.t-pad.b)*(1-j/4);
    s+=`<line x1="${pad.l}" x2="${W-pad.r}" y1="${y}" y2="${y}" stroke="#eef0f3"/>`+
       `<text x="${pad.l-6}" y="${y+4}" font-size="11" text-anchor="end" fill="#9ca3af">${fmtY(maxY*j/4)}</text>`;}
  return s;
}
function pendingChart(svg){
  const W=svg.clientWidth||600, H=200; svg.setAttribute("viewBox",`0 0 ${W} ${H}`);
  svg.innerHTML=`<text x="${W/2}" y="${H/2}" font-size="13" text-anchor="middle" fill="#9ca3af">Shown once the answer finishes: only then do we know what it needed late.</text>`;
}
function drawNeed(L){
  if(!L.need.length) return pendingChart($("needsvg"));
  const svg=$("needsvg"), W=svg.clientWidth||600, H=200, pad={l:44,r:12,t:10,b:26};
  const n=L.need.length, mx=Math.max(...L.need,1e-9);
  const X=t=>pad.l+(W-pad.l-pad.r)*(t/it.steps), Y=v=>pad.t+(H-pad.t-pad.b)*(1-v/mx);
  let s=axes(svg,W,H,pad,mx,v=>(v*100).toFixed(1)+"%");
  s+=`<rect x="${X(it.need_step)}" y="${pad.t}" width="${X(it.steps)-X(it.need_step)}" height="${H-pad.t-pad.b}" fill="#fef3c7" opacity=".7"/>`;
  s+=`<text x="${X(it.need_step)+6}" y="${pad.t+14}" font-size="11" fill="#b45309">evidence needed</text>`;
  let d=""; L.need.forEach((v,j)=>{d+=(j?"L":"M")+X(j+1).toFixed(1)+","+Y(v).toFixed(1);});
  s+=`<path d="${d}" fill="none" stroke="${css('--ev')}" stroke-width="2.2"/>`;
  s+=`<line x1="${X(step)}" x2="${X(step)}" y1="${pad.t}" y2="${H-pad.b}" stroke="#111827" stroke-width="1.2"/>`;
  s+=`<text x="${(W)/2}" y="${H-6}" font-size="11" text-anchor="middle" fill="#9ca3af">decode step</text>`;
  svg.setAttribute("viewBox",`0 0 ${W} ${H}`); svg.innerHTML=s;
}
function drawSurv(L){
  if(!L.gun.length) return pendingChart($("survsvg"));
  const svg=$("survsvg"), W=svg.clientWidth||600, H=200, pad={l:44,r:12,t:10,b:26}, n=L.gun.length;
  const X=t=>pad.l+(W-pad.l-pad.r)*(t/it.steps), Y=v=>pad.t+(H-pad.t-pad.b)*(1-v);
  let s=axes(svg,W,H,pad,1,v=>(v*100).toFixed(0)+"%");
  s+=`<rect x="${X(it.need_step)}" y="${pad.t}" width="${X(it.steps)-X(it.need_step)}" height="${H-pad.t-pad.b}" fill="#fef3c7" opacity=".7"/>`;
  ARMS.forEach(([k,,c])=>{
    const D=L.deaths[k]; let d="";
    for(let t=0;t<=it.steps;t++){const v=L.gun.filter(i=>alive(D[i],t)).length/n; d+=(t?"L":"M")+X(t).toFixed(1)+","+Y(v).toFixed(1);}
    s+=`<path d="${d}" fill="none" stroke="${c}" stroke-width="2.2" ${k==="fp16"?'':'stroke-dasharray="6 4"'}/>`;
  });
  s+=`<line x1="${X(step)}" x2="${X(step)}" y1="${pad.t}" y2="${H-pad.b}" stroke="#111827" stroke-width="1.2"/>`;
  let lx=pad.l+8; ARMS.forEach(([k,label,c])=>{s+=`<rect x="${lx}" y="${H-pad.b-22}" width="10" height="10" fill="${c}"/><text x="${lx+14}" y="${H-pad.b-13}" font-size="11" fill="#4b5563">${k.toUpperCase()}</text>`; lx+=70;});
  s+=`<text x="${(W)/2}" y="${H-6}" font-size="11" text-anchor="middle" fill="#9ca3af">decode step</text>`;
  svg.setAttribute("viewBox",`0 0 ${W} ${H}`); svg.innerHTML=s;
}
window.addEventListener("resize",()=>{ARMS.forEach(([k])=>{$("cv-"+k).width=0;});draw();});

// ---------- results tab ----------
function results(){
  // Every row and every sentence here is computed from DATA. Nothing is written in by hand,
  // so a run that came out differently would produce different words.
  const T=DATA.timeline, R=DATA.ranking, pct=v=>(v*100).toFixed(1)+"%", A=DATA.arms;
  const rounded=A.filter(a=>a.bits<16), alpha=DATA.alpha;
  const pv=p=>p==null?"n/a":(p<0.001?"&lt;0.001":p.toFixed(3));
  const name=a=>a.bits>=16?"FP16 (no rounding)":"INT"+a.bits;
  const sig=p=>p!=null && p<alpha;

  // ---- 1. which tokens get evicted
  let c1="", v1="", r1=null;
  if(R && rounded.every(a=>R[a.key+"_visual"]!=null)){
    let rows=`<tr><td>FP16 (no rounding)</td><td>100%</td><td>100%</td></tr>`;
    rounded.forEach(a=>rows+=`<tr><td>${name(a)}</td><td>${pct(R[a.key+"_visual"])}</td><td>${pct(R[a.key+"_text"])}</td></tr>`);
    const changes=rounded.map(a=>`${name(a)} changes ${pct(1-R[a.key+"_visual"])}`).join(", ");
    const visWorse=rounded.filter(a=>R[a.key+"_visual"]<R[a.key+"_text"]).length;
    const anyChange=rounded.some(a=>R[a.key+"_visual"]<1);
    r1=anyChange;
    v1=`${anyChange?"Yes":"No"}. ${changes} of the visual eviction set. `+
       (visWorse===rounded.length?"Visual tokens are affected more than text at every bit width."
        : visWorse===0?"Text tokens are affected at least as much as visual."
        : `Visual tokens are affected more than text at ${visWorse} of ${rounded.length} bit widths.`);
    c1=`<div class="card"><h4 style="margin:0 0 12px">1 · Does rounding change which tokens get evicted?</h4>
      <table><tr><th>Precision</th><th>Visual: same eviction set as FP16</th><th>Text</th></tr>${rows}</table>
      <p class="verdict ${anyChange?"yes":"no"}" style="margin:14px 0 0">${v1}</p>
      <p style="color:var(--sub);font-size:13px;margin:6px 0 0">n = ${R.n} images, accumulated (H2O-style) scoring, averaged over layers.</p></div>`;
  }

  // ---- 2. is needed evidence lost earlier
  let c2="", r2=null, dir2="";
  if(T){
    let rows=`<tr><td>FP16 (no rounding)</td><td>${pct(T.fp16)}</td><td>—</td><td></td></tr>`;
    rounded.forEach(a=>rows+=`<tr><td>${name(a)}</td><td>${pct(T[a.key])}</td><td>${(T[a.key+"_delta"]*100>=0?"+":"")+(T[a.key+"_delta"]*100).toFixed(1)} pts</td><td>${pv(T[a.key+"_p"])}</td></tr>`);
    const more=rounded.filter(a=>T[a.key+"_delta"]>0 && sig(T[a.key+"_p"]));
    const less=rounded.filter(a=>T[a.key+"_delta"]<0 && sig(T[a.key+"_p"]));
    r2 = more.length>0;
    // per precision, only what the test supports
    dir2 = rounded.map(a=>{
      const d=T[a.key+"_delta"];
      return sig(T[a.key+"_p"]) ? `${name(a)} loses significantly ${d>0?"more":"less"}`
                                : `${name(a)} shows no significant difference`;
    }).join("; ");
    const per=rounded.map(a=>{
      const d=T[a.key+"_delta"], w=d>0?"more":d<0?"less":"the same";
      return `${name(a)} loses ${w} (${(d*100>=0?"+":"")+(d*100).toFixed(1)} pts, ${sig(T[a.key+"_p"])?"significant":"not significant"} at p &lt; ${alpha}; lower than FP16 on ${T[a.key+"_less"]} of ${T.n} images)`;
    }).join("; ");
    const v2=`${more.length?"Yes":"No"}. ${pct(T.fp16)} of the needed evidence is already gone at full precision. ${per}.`;
    c2=`<div class="card"><h4 style="margin:0 0 12px">2 · Does rounding make needed evidence get lost earlier?</h4>
      <table><tr><th>Precision</th><th>Evidence lost before needed</th><th>vs FP16</th><th>p (paired)</th></tr>${rows}</table>
      <p class="verdict ${more.length?"yes":"no"}" style="margin:14px 0 0">${v2}</p>
      <p style="color:var(--sub);font-size:13px;margin:6px 0 0">n = ${T.n} images with answers of at least ${DATA.min_steps} tokens, averaged over layers. Wilcoxon signed-rank, paired per image.</p></div>`;
  }

  // ---- headline, assembled from the two verdicts above
  const head = (r1==null||!T) ? "Results will appear here once both measurements have been run."
    : `Rounding ${r1?"clearly changes":"does not change"} <i>which</i> visual tokens get thrown away. `+
      (r2 ? `It also makes the model lose more of the evidence it needs later (${dir2}).`
          : `It does <i>not</i> make the model lose more of the evidence it needs later (${dir2}); ${pct(T.fp16)} of that evidence is lost even without rounding.`);
  let h=`<div class="card"><p class="headline">${head}</p>
    <p style="color:var(--sub);margin:0">Same image, same question, same memory budget. The only thing that changes is how precisely the cache is stored.</p></div>
    <div class="grid2">${c1}${c2}</div>`;
  h+=`<div class="card" style="margin-top:20px"><h4 style="margin:0 0 10px">What this does and does not show yet</h4>
    <table><tr><th>Question</th><th>Status</th></tr>
    <tr><td>Does rounding change which tokens are evicted?</td><td class="verdict">${r1==null?"not run":(r1?"Yes":"No")+", measured"}</td></tr>
    <tr><td>Does it make needed evidence get lost earlier?</td><td class="verdict">${r2==null?"not run":(r2?"Yes":"No")+", measured"}</td></tr>
    <tr><td>Does it hold at other budgets and eviction policies?</td><td>Next (phase A2)</td></tr>
    <tr><td>Is it the cache, not the model weights? (falsifier)</td><td>Next (phase B)</td></tr>
    <tr><td>Does the answer get worse?</td><td>Later: needs real eviction (phase C)</td></tr>
    <tr><td>Does it hold with real object boxes as evidence?</td><td>Later: GQA (phase D)</td></tr></table>
    <div class="note">Counterfactual eviction: each precision's evictions are recorded while the model answers from the full cache, so all arms see identical inputs. Evidence is attention-defined (high late, low early attention). One model, small n; treat as a first measurement.</div></div>`;
  $("res").innerHTML=h;
}
$("t-live").onclick=()=>{$("t-live").classList.add("on");$("t-res").classList.remove("on");$("live").classList.remove("hidden");$("res").classList.add("hidden");draw();};
$("t-res").onclick=()=>{stop();$("t-res").classList.add("on");$("t-live").classList.remove("on");$("res").classList.remove("hidden");$("live").classList.add("hidden");};

// ---------- live mode (only when served by app.py) ----------
if(LIVE){
  $("livecard").classList.remove("hidden");
  let dataURL=null;
  const setStatus=(t,err)=>{$("lstatus").textContent=t;$("lstatus").classList.toggle("err",!!err);};
  const take=f=>{
    if(!f) return;
    if(!f.type.startsWith("image/")){setStatus("That file is not an image.",true);return;}
    const r=new FileReader(); r.onload=()=>{dataURL=r.result;
      $("drop").querySelectorAll("img").forEach(x=>x.remove());
      const im=document.createElement("img"); im.src=dataURL; $("drop").appendChild(im);
      $("droptext").style.display="none"; $("lrun").disabled=false;
      setStatus("Ready. Press Run live.");};
    r.readAsDataURL(f);
  };
  $("file").onchange=e=>take(e.target.files[0]);
  ["dragover","dragenter"].forEach(ev=>$("drop").addEventListener(ev,e=>{e.preventDefault();$("drop").classList.add("over");}));
  ["dragleave","drop"].forEach(ev=>$("drop").addEventListener(ev,e=>{e.preventDefault();$("drop").classList.remove("over");}));
  $("drop").addEventListener("drop",e=>take(e.dataTransfer.files[0]));

  $("lrun").onclick=async()=>{
    if(!dataURL) return;
    stop(); $("lrun").disabled=true; setStatus("Starting...");
    let r;
    try{ r=await fetch("/api/run",{method:"POST",headers:{"Content-Type":"application/json"},
           body:JSON.stringify({image:dataURL,question:$("lq").value})}); }
    catch(e){ setStatus("Could not reach the app. Is app.py still running?",true); $("lrun").disabled=false; return; }
    if(!r.ok){ const j=await r.json().catch(()=>({})); setStatus(j.error||("Error "+r.status),true); $("lrun").disabled=false; return; }

    let idx=-1, t0=Date.now();
    const es=new EventSource("/api/events");
    es.onmessage=m=>{
      const ev=JSON.parse(m.data);
      if(ev.type==="setup"){
        const n=ev.n_visual, blank=()=>Array(n).fill(-1);
        const item={id:"live",question:ev.question,grid:ev.grid,img:ev.image_b64,tokens:[],steps:0,
          budget:ev.budget,n_visual:n,need_step:1e9,
          layers:{[String(DATA.display_layer)]:{deaths:Object.fromEntries(ARMS.map(([k])=>[k,blank()])),gun:[],need:[],stats:null}}};
        DATA.items.push(item); idx=DATA.items.length-1;
        const im=new Image(); im.src="data:image/jpeg;base64,"+ev.image_b64; imgs[idx]=im;
        const o=document.createElement("option"); o.value=idx; o.textContent="Live: "+ev.question.slice(0,58);
        $("item").appendChild(o); $("item").value=idx;
        im.onload=()=>{pickItem(idx);};
        setStatus(`Answering live: ${n} visual tokens, memory budget ${ev.budget} tokens...`);
      } else if(ev.type==="step" && idx>=0){
        const L=DATA.items[idx].layers[String(DATA.display_layer)];
        Object.entries(ev.died||{}).forEach(([a,arr])=>arr.forEach(([i,st])=>{L.deaths[a][i]=st;}));
        const cur=DATA.items[idx];                 // the live item, not whatever is selected
        cur.tokens.push(ev.token); cur.steps=ev.step;
        if(cur===it){ $("step").max=ev.step; setStep(ev.step); }
        setStatus(`Answering live: step ${ev.step} of up to ${DATA.max_new} · ${((Date.now()-t0)/1000).toFixed(0)}s`);
      } else if(ev.type==="done"){
        es.close();
        if(ev.item && idx>=0){
          ev.item.too_short=ev.too_short; DATA.items[idx]=ev.item; pickItem(idx); setStep(ev.item.steps);
          setStatus(`Done in ${ev.secs}s. The outlined tokens are the evidence the answer relied on late; red outlines were already thrown away. Saved to ${ev.saved}.`);
        } else {
          setStatus("Done, but the image was too small to analyse.",true);
        }
        $("lrun").disabled=false;
      } else if(ev.type==="error"){
        es.close(); setStatus("The run failed: "+ev.message,true); $("lrun").disabled=false;
      }
    };
    es.onerror=()=>{ if(es.readyState===2) return; };
  };
}

results(); pickItem(0);
// deep links: #results, #step-80, #item-3 (combinable: #item-3-step-80)
(function(){const h=location.hash.slice(1); if(!h) return;
  const mi=h.match(/item-(\d+)/); if(mi){const i=Math.min(DATA.items.length-1,+mi[1]-1); $("item").value=i; pickItem(i);}
  const ms=h.match(/step-(\d+)/); if(ms) setStep(Math.min(it.steps,+ms[1]));
  if(h.includes("results")) $("t-res").click();})();
</script>
</body>
</html>
"""


def page_data(runs, ranking):
    """Everything the page needs. Used by the static build AND the live app."""
    return {"items": load_runs(runs), "timeline": summarise_timeline(runs),
            "ranking": summarise_ranking(ranking), "arms": arm_meta(),
            "display_layer": CFG["display_layer"], "max_new": CFG["generation"]["max_new_tokens"],
            "alpha": CFG["stats"]["alpha"], "min_steps": CFG["evidence"]["min_steps"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results_timeline")
    ap.add_argument("--ranking", default="results_acc")
    ap.add_argument("--out", default="demo/visretain_demo.html")
    a = ap.parse_args()

    data = page_data(a.runs, a.ranking)
    items = data["items"]
    if not items:
        raise SystemExit(f"no demo-ready runs in {a.runs} (re-run run_timeline.py)")
    out = Path(a.out); out.parent.mkdir(exist_ok=True)
    # </ inside JSON would close the script tag
    out.write_text(PAGE.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
                   .replace("__LIVE__", "false"))
    print(f"wrote {out}  ({out.stat().st_size/1e6:.1f} MB, {len(items)} images)")


if __name__ == "__main__":
    main()
