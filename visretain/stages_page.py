"""The stages page: how Qwen2.5-VL processes an image, step by step, with the proof for each.

Served live by app.py at / (upload any image), or built offline by build_stages.py.
All numbers shown come from the run's data; nothing on the page is hand-written.
"""

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VisRetain: how the model sees your image, verified stage by stage</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#111827;--sub:#4b5563;--mute:#9ca3af;--line:#e5e7eb;
  --acc:#2563eb;--ok:#15803d;--okbg:#dcfce7;--bad:#b91c1c;--badbg:#fee2e2;--heat:220,38,38;--ev:#f59e0b}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif}
header{background:var(--card);border-bottom:1px solid var(--line);padding:18px 32px;display:flex;align-items:center;gap:24px}
header h1{font-size:20px;margin:0;font-weight:650;letter-spacing:-.01em}
header p{margin:2px 0 0;color:var(--sub);font-size:13px}
nav{margin-left:auto;display:flex;gap:4px;background:var(--bg);padding:4px;border-radius:10px}
nav a,nav span{padding:8px 14px;border-radius:7px;font-size:14px;color:var(--sub);text-decoration:none;white-space:nowrap}
nav a.on{background:var(--card);color:var(--ink);font-weight:600;box-shadow:0 1px 2px rgba(0,0,0,.08)}
nav span{color:#c4c8cf;cursor:default}
main{max-width:1180px;margin:0 auto;padding:24px 32px 64px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px 24px;margin-bottom:18px}
.upload{display:grid;grid-template-columns:200px 1fr auto;gap:20px;align-items:center}
.drop{border:2px dashed #cbd5e1;border-radius:12px;height:120px;display:flex;align-items:center;justify-content:center;
  text-align:center;color:var(--sub);font-size:13px;cursor:pointer;background:#fafbfc;overflow:hidden;position:relative}
.drop.over{border-color:var(--ink)} .drop img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover}
.samples{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px;align-items:center;font-size:13px;color:var(--sub)}
.samples img{width:56px;height:40px;object-fit:cover;border-radius:6px;cursor:pointer;border:2px solid transparent}
.samples img:hover{border-color:var(--ink)}
button.run{background:var(--ink);color:#fff;border:0;border-radius:10px;padding:14px 24px;font:inherit;font-weight:600;cursor:pointer}
button.run:disabled{opacity:.4;cursor:default}
.qbox{width:100%;height:52px;margin-top:10px;font:inherit;font-size:14px;padding:8px 10px;border:1px solid var(--line);border-radius:8px;resize:vertical}
.words{display:flex;flex-wrap:wrap;gap:2px;font-size:16px;line-height:1.9}
.words span{padding:1px 2px;border-radius:4px;cursor:pointer;white-space:pre}
.words span:hover{background:#e0e7ff}
.words span.on{background:var(--acc);color:#fff}
.words span.first{color:var(--mute);cursor:default}
.words span.first:hover{background:none}
.toplist{font-size:14px;margin-top:10px;display:grid;grid-template-columns:auto 1fr;gap:3px 12px}
.toplist b{font-variant-numeric:tabular-nums}
input[type=range].layer{width:100%;accent-color:var(--ink)}
.splitrow{display:grid;grid-template-columns:62px 1fr;gap:8px;align-items:center;font-size:12px;color:var(--sub);margin:2px 0;cursor:pointer}
.splitrow.on{color:var(--ink);font-weight:700}
.splitbar{display:flex;height:14px;border-radius:3px;overflow:hidden}
.splitrow.on .splitbar{outline:2px solid var(--ink)}
.status{font-size:13px;color:var(--sub);margin-top:8px;min-height:20px} .status.err{color:var(--bad)}
.intro{color:var(--sub);margin:0 0 18px;max-width:880px}
.step{display:grid;grid-template-columns:44px 1fr;gap:6px 16px}
.num{width:34px;height:34px;border-radius:50%;background:var(--ink);color:#fff;display:flex;align-items:center;justify-content:center;font-weight:650}
.step h2{margin:4px 0 4px;font-size:18px;letter-spacing:-.01em}
.step .lead{color:var(--sub);margin:0 0 14px}
.math{font:13px/1.7 ui-monospace,SFMono-Regular,Menlo,monospace;background:#f8fafc;border:1px solid var(--line);
  border-radius:10px;padding:12px 14px;margin:0 0 14px;white-space:pre-wrap}
.math b{color:var(--ink)} .math .c{color:var(--mute)}
.proof{display:flex;gap:10px;align-items:flex-start;border-radius:10px;padding:10px 14px;font-size:14px;margin-top:14px}
.proof.ok{background:var(--okbg);color:#14532d} .proof.bad{background:var(--badbg);color:#7f1d1d}
.proof .tick{font-weight:800;font-size:16px;line-height:1.3}
.proof small{display:block;color:inherit;opacity:.85}
.figrow{display:flex;gap:18px;align-items:flex-start;flex-wrap:wrap}
canvas.view{border-radius:8px;display:block;max-width:100%;cursor:crosshair}
.side{flex:1;min-width:220px;font-size:14px;color:var(--sub)}
.side .big{font-size:26px;font-weight:650;color:var(--ink);letter-spacing:-.01em;line-height:1.2}
.kv{display:grid;grid-template-columns:auto 1fr;gap:4px 12px;font-size:14px;margin-top:10px}
.kv b{color:var(--ink);font-variant-numeric:tabular-nums}
canvas.zoom{image-rendering:pixelated;border-radius:6px;border:1px solid var(--line);background:#fff}
.toggles{display:flex;gap:16px;font-size:13px;color:var(--sub);margin:0 0 10px;flex-wrap:wrap}
.strip{display:flex;flex-wrap:wrap;gap:4px;font:12px ui-monospace,SFMono-Regular,Menlo,monospace}
.strip span{padding:3px 6px;border-radius:5px;background:#f1f5f9;color:#334155;white-space:pre}
.strip span.sp{background:#e0e7ff;color:#3730a3}
.strip span.im{background:var(--acc);color:#fff;font-weight:600;padding:3px 14px}
.bars{display:flex;align-items:flex-end;gap:6px;height:110px;margin-top:28px}
.bars div{flex:1;background:rgb(var(--heat));border-radius:4px 4px 0 0;position:relative;min-width:18px}
.bars div span{position:absolute;bottom:-20px;left:0;right:0;text-align:center;font-size:11px;color:var(--sub)}
.bars div i{position:absolute;top:-18px;left:0;right:0;text-align:center;font-size:11px;font-style:normal;color:var(--sub)}
.twin{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.twin canvas{width:100%}
.twinhead{font-weight:600;margin-bottom:8px}
.result{font-size:15px;font-weight:600;margin-top:10px;min-height:44px;line-height:1.45}
.result.ok{color:var(--ok)} .result.bad{color:var(--bad)}
.tallyrow{display:flex;align-items:baseline;gap:10px;margin-top:8px}
.tallyrow .tally{white-space:nowrap}
.proof.info{background:#eef2ff;color:#312e81}
.pill.info{background:#e0e7ff;color:#3730a3}
.tally{font-size:30px;font-weight:700;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:9px 12px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--mute);font-weight:600}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12px;font-weight:700}
.pill.ok{background:var(--okbg);color:var(--ok)} .pill.bad{background:var(--badbg);color:var(--bad)}
.pill.wait{background:#f1f5f9;color:var(--mute)}
.hidden{display:none !important}
.note{color:var(--sub);font-size:13px}
#banner .big2,#banner2 .big2{font-size:22px;font-weight:700;letter-spacing:-.01em}
#banner a,#banner2 a{color:var(--acc);font-size:14px;margin-left:auto}
.legend{display:inline-flex;align-items:center;gap:6px;font-size:12px;color:var(--sub)}
.legend i{display:inline-block;width:80px;height:10px;border-radius:3px;background:linear-gradient(90deg,rgba(var(--heat),0),rgba(var(--heat),1))}
</style>
</head>
<body>
<header>
  <div><h1>VisRetain: how the model sees your image</h1>
    <p>Qwen2.5-VL-3B, running on this machine. Each stage shows what the model does, the math with your image's own numbers, and the check that proves it.</p></div>
  <nav><a class="on" href="#" data-stage="1">1 · Image → tokens</a><a href="#" data-stage="2">2 · Attention</a><span title="next">3 · Rounding</span><a id="fulllink" href="/full">Full pipeline</a></nav>
</header>
<main>
  <div class="card upload" id="uploadcard">
    <label class="drop" id="drop"><span id="droptext">Drop an image here<br>or click to choose</span><input type="file" id="file" accept="image/*" hidden></label>
    <div>
      <div style="font-weight:600" id="uptitle">Stage 1 · Image → visual tokens</div>
      <div class="note" id="upnote">How Qwen cuts your image into the pieces it remembers, and the proof that we know exactly which piece is which.</div>
      <textarea id="question" class="qbox" placeholder="Your question about the image, e.g. What is the person wearing and what are they doing?"></textarea>
      <div class="samples" id="samples"></div>
      <div class="status" id="status"></div>
    </div>
    <button class="run" id="run" disabled>Run stage 1</button>
  </div>

  <div id="out" class="hidden">
    <div class="card" id="banner" style="display:flex;align-items:center;gap:18px;padding:16px 22px"></div>
    <p class="intro">Every later stage of VisRetain colours these squares (which ones the model keeps, which it throws away). So before anything else we need to know, for certain, which square of your image each stored token stands for.</p>

    <div class="card step"><div class="num">1</div><div>
      <h2>Your image</h2><p class="lead" id="s1lead"></p>
      <canvas class="view" id="cv-orig"></canvas></div></div>

    <div class="card step"><div class="num">2</div><div>
      <h2>Resize to a grid of <span class="v-F"></span> px squares</h2>
      <p class="lead">Qwen needs both sides to be a whole number of squares, and keeps the total within a pixel budget.</p>
      <div class="math" id="s2math"></div>
      <div class="figrow"><canvas class="view" id="cv-resized"></canvas>
        <div class="side"><div class="big" id="s2size"></div><div id="s2sub"></div></div></div>
      <div id="p-1.1"></div><div id="p-1.6"></div></div></div>

    <div class="card step"><div class="num">3</div><div>
      <h2>Cut into <span class="v-P"></span> × <span class="v-P"></span> px patches</h2>
      <p class="lead">Each patch becomes one row of numbers in the model's input. Hover over the image to inspect a patch.</p>
      <div class="math" id="s3math"></div>
      <div class="figrow"><canvas class="view" id="cv-patch"></canvas>
        <div class="side"><canvas class="zoom" id="zoom" width="168" height="168"></canvas>
          <div class="kv" id="s3kv"></div></div></div>
      <div id="p-1.2"></div></div></div>

    <div class="card step"><div class="num">4</div><div>
      <h2>Merge each 2 × 2 block of patches into one token</h2>
      <p class="lead">The language model stores one token per <span class="v-F"></span> × <span class="v-F"></span> px square. These are the tokens VisRetain tracks. Hover to see a token's number, row, column and position.</p>
      <div class="math" id="s4math"></div>
      <div class="toggles"><label><input type="checkbox" id="t-num" checked> token numbers</label>
        <label><input type="checkbox" id="t-win"> encoder windows</label>
        <button class="run" style="padding:6px 12px;font-size:13px;font-weight:500" id="order">Show reading order</button></div>
      <div class="figrow"><canvas class="view" id="cv-tok"></canvas>
        <div class="side"><div class="kv" id="s4kv"></div></div></div>
      <div id="p-1.3"></div></div></div>

    <div class="card step"><div class="num">5</div><div>
      <h2>The vision encoder turns patches into tokens</h2>
      <p class="lead" id="s5lead"></p>
      <div class="math" id="s5math"></div><div id="p-1.8"></div></div></div>

    <div class="card step"><div class="num">6</div><div>
      <h2>The tokens join the prompt</h2>
      <p class="lead">What the language model actually reads: the chat template, the image as one block of tokens, then your question. Each image token gets a 3D position (time, row, column).</p>
      <div class="strip" id="strip"></div>
      <div class="math" id="s6math" style="margin-top:14px"></div>
      <div id="p-1.4"></div><div id="p-1.7"></div></div></div>

    <div class="card step"><div class="num">7</div><div>
      <h2>Proof: are our labels right?</h2>
      <p class="lead">Steps 3–6 labelled every square with the note the model keeps for it. Here we <b>check those labels</b>.
        <b>Click any square.</b> We flip that square's colours (it gets a <b>black outline</b>: “you changed this”) and run the model again.
        <b>Red shows which of the model's notes reacted.</b><br>
        <b>Left:</b> the model's notes are kept separate, so only one note should react: the one we labelled as your square. If any other note turned red, our labels would be wrong.<br>
        <b>Right:</b> the real model, where notes share information with their neighbours, so the red spreads around your square.</p>
      <div class="twin">
        <div><div class="twinhead">Notes kept separate <span class="note">· checks our labels</span></div><canvas class="view" id="cv-occ-off"></canvas>
          <div class="result" id="resA">Click a square to test it.</div>
          <div class="tallyrow"><span class="tally" id="tallyA">0 / 0</span><span class="note" id="tallyAsub">squares where only the predicted token changed</span></div></div>
        <div><div class="twinhead">The real model <span class="note">· notes share with neighbours</span></div><canvas class="view" id="cv-occ"></canvas>
          <div class="result" id="resB">&nbsp;</div>
          <div class="tallyrow"><span class="tally" id="tallyB" style="color:var(--ink)">0 / 0</span><span class="note" id="tallyBsub">squares where the changed square's own token still reacted most</span></div></div>
      </div>
      <div class="figrow" style="margin-top:16px">
        <div class="side" style="min-width:300px"><div class="kv" id="occkv"></div></div>
        <div class="side"><div style="font-weight:600;color:var(--ink)">How far the change spreads (real encoder)</div>
          <div class="note">Average response by distance from the changed square (1.00 = the square itself).</div>
          <div class="bars" id="bars"></div><div style="height:24px"></div>
          <span class="legend">no change <i></i> most change</span></div></div>
      <div id="p-1.5a"></div><div id="p-1.5b"></div></div></div>

    <div class="card"><h2 style="margin:0 0 4px;font-size:18px">Stage 1 checks</h2>
      <p class="note" style="margin:0 0 12px">Pass criteria were written down in STAGES.md before these checks were built.</p>
      <table id="checks"></table></div>
  </div>
  <div id="out2" class="hidden">
    <div class="card" id="banner2" style="display:flex;align-items:center;gap:18px;padding:16px 22px"></div>
    <p class="intro">While writing each word of its answer, the model <b>looks back</b> at everything in its memory: the photo's patches, your question, and the words it has written so far. How strongly it looks at each one is called <b>attention</b>. Later, VisRetain throws away the notes that get the least attention, so this has to be right. Stage 2 shows it, and proves our numbers are the model's own.</p>

    <div class="card step"><div class="num">1</div><div>
      <h2>The model answers your question</h2>
      <p class="lead" id="s2q"></p>
      <p class="note" style="margin:0 0 8px"><b>Click any word</b> to see where the model looked while choosing it. The grey first word was chosen while reading the prompt, before the step-by-step loop starts.</p>
      <div class="words" id="words"></div>
      <div style="margin-top:12px"><button class="run" style="padding:8px 14px;font-size:13px" id="play2">▶ Play through the answer</button></div></div></div>

    <div class="card step"><div class="num">2</div><div>
      <h2>Where it looked while choosing “<span id="curword"></span>”</h2>
      <p class="lead">Blue = how much attention each patch of the photo got. Darker = looked at more. <b>Click a patch</b> to follow it word by word (step 3).</p>
      <div class="figrow"><canvas class="view" id="cv-att"></canvas>
        <div class="side">
          <div class="big" id="attshare"></div><div id="attsharesub"></div>
          <div style="margin-top:16px;font-weight:600;color:var(--ink)">Layer <span id="layerval"></span> of <span id="nlayers"></span></div>
          <input type="range" class="layer" id="layer" min="0" step="1">
          <div class="note">The model has many layers, each looking in its own way. Early layers look broadly; middle layers tend to look at the objects being described.</div>
          <div style="margin-top:14px;font-weight:600;color:var(--ink)">Most-looked-at patches</div>
          <div class="toplist" id="toplist"></div>
          <div style="margin-top:12px"><span class="legend">less <i style="background:linear-gradient(90deg,rgba(37,99,235,0),rgba(37,99,235,1))"></i> more attention</span></div>
        </div></div></div></div>

    <div class="card step"><div class="num">3</div><div>
      <h2>One patch, word by word</h2>
      <p class="lead" id="patchlead">Click a patch in step 2 to see how much attention it got at every word of the answer.</p>
      <svg id="patchsvg" style="width:100%;height:210px"></svg></div></div>

    <div class="card step"><div class="num">4</div><div>
      <h2>Where each layer looks</h2>
      <p class="lead">For every layer, how the attention splits between the very first token, your question, the photo, and the answer written so far (averaged over the answer). Click a layer to show it in step 2.</p>
      <div style="display:flex;gap:16px;font-size:12px;color:var(--sub);margin-bottom:8px">
        <span><i style="display:inline-block;width:12px;height:12px;background:#9ca3af;vertical-align:-2px;margin-right:5px"></i>first token</span>
        <span><i style="display:inline-block;width:12px;height:12px;background:#a78bfa;vertical-align:-2px;margin-right:5px"></i>your question &amp; template</span>
        <span><i style="display:inline-block;width:12px;height:12px;background:#2563eb;vertical-align:-2px;margin-right:5px"></i>the photo</span>
        <span><i style="display:inline-block;width:12px;height:12px;background:#f59e0b;vertical-align:-2px;margin-right:5px"></i>answer so far</span></div>
      <div id="splits"></div></div></div>

    <div class="card step"><div class="num">5</div><div>
      <h2>How attention is computed, and the proof it's the model's own</h2>
      <p class="lead">We compute attention ourselves from the model's numbers, because later stages must recompute it after rounding. So we must prove our attention is exactly the model's.</p>
      <div class="math" id="attmath"></div>
      <div id="p2-2.1"></div><div id="p2-2.2"></div><div id="p2-2.3"></div><div id="p2-2.4"></div><div id="p2-2.5"></div><div id="p2-2.N"></div></div></div>

    <div class="card"><h2 style="margin:0 0 4px;font-size:18px">Stage 2 checks</h2>
      <p class="note" style="margin:0 0 12px">Pass criteria were written down in STAGES.md before these checks were built. 2.N runs the same checks on deliberately broken versions, which must fail: it proves the checks can catch an error.</p>
      <table id="checks2"></table></div>
  </div>
</main>

<script>
const DATA = __DATA__;
const LIVE = __LIVE__;
const $ = id => document.getElementById(id);
let S = null, img = null, dataURL = null, hoverP = -1, hoverT = -1, orderAnim = -1, occ = null, tested = {};
const HEAT = getComputedStyle(document.documentElement).getPropertyValue("--heat").trim();
const fmt = n => n.toLocaleString("en-US");

// ---------------------------------------------------------------- input
const setStatus=(t,err)=>{$("status").textContent=t||""; $("status").classList.toggle("err",!!err);};
function take(url){ dataURL=url; $("drop").querySelectorAll("img").forEach(x=>x.remove());
  const im=document.createElement("img"); im.src=url; $("drop").appendChild(im); $("droptext").style.display="none";
  $("run").disabled=!LIVE; setStatus(LIVE?`Ready. Press Run stage ${curStage}.`:""); }
if(LIVE){
  $("file").onchange=e=>{const f=e.target.files[0]; if(!f) return;
    if(!f.type.startsWith("image/")) return setStatus("That file is not an image.",true);
    const r=new FileReader(); r.onload=()=>take(r.result); r.readAsDataURL(f);};
  ["dragover","dragenter"].forEach(ev=>$("drop").addEventListener(ev,e=>{e.preventDefault();$("drop").classList.add("over");}));
  ["dragleave","drop"].forEach(ev=>$("drop").addEventListener(ev,e=>{e.preventDefault();$("drop").classList.remove("over");}));
  $("drop").addEventListener("drop",e=>{const f=e.dataTransfer.files[0]; if(f){const r=new FileReader(); r.onload=()=>take(r.result); r.readAsDataURL(f);}});
  if(DATA.samples && DATA.samples.length){
    $("samples").innerHTML="or try one: ";
    DATA.samples.forEach(s=>{const i=document.createElement("img"); i.src="data:image/jpeg;base64,"+s.img; i.title=s.label;
      i.onclick=()=>take(i.src); $("samples").appendChild(i);});
  }
  $("run").onclick=()=>curStage===2?runStage2():runStage1();
} else {
  $("uploadcard").classList.add("hidden"); $("fulllink").href="visretain_demo.html";
}

async function api(path, body){
  const r=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  const j=await r.json().catch(()=>({error:"bad response"}));
  if(!r.ok) throw new Error(j.error||("HTTP "+r.status)); return j;
}

async function runStage1(){
  if(!dataURL) return;
  $("run").disabled=true; setStatus("Running stage 1 on the model…");
  try{ S=await api("/api/stage1",{image:dataURL,question:$("question").value}); }
  catch(e){ setStatus(e.message,true); $("run").disabled=false; return; }
  render(S);
  setStatus(`Stage 1 ran. Now testing ${S.squares.length} random squares one by one…`);
  for(let i=0;i<S.squares.length;i++){
    const [r,c]=S.squares[i];
    try{ const res=await api("/api/stage1/occlude",{r,c,record:true}); onTrial(res,true); }
    catch(e){ setStatus("Causal test failed: "+e.message,true); break; }
  }
  setStatus("Done. Click any square in step 7 to test it yourself.");
  $("run").disabled=false;
}

// ---------------------------------------------------------------- drawing helpers
function fit(cv, w, h, maxW){
  const W=Math.min(maxW, w), H=Math.round(W*h/w), dpr=window.devicePixelRatio||1;
  cv.width=W*dpr; cv.height=H*dpr; cv.style.width=W+"px"; cv.style.height=H+"px";
  const g=cv.getContext("2d"); g.setTransform(dpr,0,0,dpr,0,0); return [g,W,H];
}
function grid(g,W,H,cols,rows,color,width){
  g.strokeStyle=color; g.lineWidth=width; g.beginPath();
  for(let c=1;c<cols;c++){const x=c*W/cols; g.moveTo(x,0); g.lineTo(x,H);}
  for(let r=1;r<rows;r++){const y=r*H/rows; g.moveTo(0,y); g.lineTo(W,y);} g.stroke();
}
function cellAt(cv,e,cols,rows){ const b=cv.getBoundingClientRect();
  const c=Math.floor((e.clientX-b.left)/b.width*cols), r=Math.floor((e.clientY-b.top)/b.height*rows);
  return (c<0||r<0||c>=cols||r>=rows)?null:[r,c]; }
const proof=(id)=>{ const c=S.checks.find(x=>x.id===id); const el=$("p-"+id); if(!c||!el) return;
  const cls=c.pass===null?"info":(c.pass?"ok":"bad"), mark=c.pass===null?"≈":(c.pass?"✓":"✗"),
        tag=c.pass===null?"Measurement":"Check";
  el.innerHTML=`<div class="proof ${cls}"><span class="tick">${mark}</span><div><b>${tag} ${c.id}: ${c.name}</b><small>${c.detail}</small></div></div>`; };

// ---------------------------------------------------------------- render
function render(d){
  $("out").classList.remove("hidden");
  const T=d.token, Pt=d.patch, E=d.encoder, B=d.budget, Ps=d.positions;
  document.querySelectorAll(".v-F").forEach(e=>e.textContent=T.px);
  document.querySelectorAll(".v-P").forEach(e=>e.textContent=Pt.px);
  img=new Image(); img.onload=()=>drawAll(); img.src="data:image/png;base64,"+d.resized.img;
  const orig=new Image(); orig.onload=()=>{const [g,W,H]=fit($("cv-orig"),d.original.w,d.original.h,520); g.drawImage(orig,0,0,W,H);};
  orig.src="data:image/jpeg;base64,"+d.original.img;

  $("s1lead").innerHTML=`${fmt(d.original.w)} × ${fmt(d.original.h)} px. Question used for the prompt: “${d.question}”`;
  $("s2math").innerHTML=d.resize_steps.map((s,i)=>`<span class="c">${i+1}. ${s.what}</span>\n<b>${s.math}</b>`).join("\n\n");
  $("s2size").textContent=`${d.resized.w} × ${d.resized.h} px`;
  $("s2sub").innerHTML=`= ${T.cols} × ${T.rows} squares of ${T.px} px<br>pixel budget: ${fmt(B.max_px)} px (at most ${fmt(B.cap_tokens)} tokens)`
     +(B.uncapped_tokens!==T.n?`<br><span style="color:var(--bad)">without the budget this image would be ${fmt(B.uncapped_tokens)} tokens</span>`:"");
  $("s3math").innerHTML=`<b>${Pt.cols} × ${Pt.rows} = ${fmt(Pt.n)} patches</b> of ${Pt.px} × ${Pt.px} px\n`+
     `each patch = ${Pt.px} × ${Pt.px} px × 3 colours × ${Pt.frames} frames = <b>${fmt(Pt.numbers)} numbers</b>\n`+
     `<span class="c">(a still image is fed as a ${Pt.frames}-frame clip by repeating it)</span>`;
  $("s4math").innerHTML=`<b>${Pt.cols} ÷ ${T.merge} = ${T.cols} columns,  ${Pt.rows} ÷ ${T.merge} = ${T.rows} rows  →  ${T.cols} × ${T.rows} = ${fmt(T.n)} tokens</b>\n`+
     `token k sits at  row = k ÷ ${T.cols},  column = k mod ${T.cols}   <span class="c">(read left to right, top to bottom)</span>`;
  $("s5lead").innerHTML=`${E.layers} transformer layers process the ${fmt(Pt.n)} patches. Most layers only let a patch look at others inside its own ${E.window_px} × ${E.window_px} px window (${E.window_tokens} × ${E.window_tokens} tokens; tick "encoder windows" in step 4 to see them). Layers ${E.full_layers.join(", ")} look across the whole image.`;
  $("s5math").innerHTML=`${fmt(Pt.n)} patches × ${E.hidden} numbers  →  ${E.layers} layers  →  merge each 2 × 2 block\n`+
     `<b>→ ${fmt(T.n)} tokens × ${fmt(T.dim)} numbers each</b>   <span class="c">(the vectors the language model stores in its KV cache)</span>`;
  // strip
  $("strip").innerHTML=d.strip.map(s=>s.kind==="image"?`<span class="im">${fmt(s.count)} image tokens</span>`
     :`<span class="${s.kind==="special"?"sp":""}">${s.text.replace(/\n/g,"↵").replace(/</g,"&lt;")}</span>`).join("");
  $("s6math").innerHTML=`image tokens: positions (time ${Ps.t}, row ${Ps.start}…${Ps.start+T.rows-1}, column ${Ps.start}…${Ps.start+T.cols-1})\n`+
     (Ps.after_image!=null?`next text token: position <b>${Ps.after_image} = ${Ps.start} + ${Ps.jump}</b> = start + max(${T.rows} rows, ${T.cols} columns), <span class="c">not start + ${fmt(T.n)}</span>`:"");
  d.checks.forEach(c=>proof(c.id));
  tested={}; occ=null; hoverP=-1; hoverT=-1;
  $("tallyA").textContent=`0 / ${d.squares.length}`; $("tallyB").textContent=`0 / ${d.squares.length}`;
  $("bars").innerHTML=""; $("occkv").innerHTML=""; $("p-1.5a").innerHTML=""; $("p-1.5b").innerHTML="";
  checksTable();
}

function drawAll(){ drawPatch(); drawTok(); drawOcc(); const d=S;
  if(hoverP<0) showPatch([Math.floor(d.patch.rows/3), Math.floor(d.patch.cols/2)]);   // a default, so nothing starts empty
  if(hoverT<0) showToken([Math.floor(d.token.rows/3), Math.floor(d.token.cols/2)]);
  const [g,W,H]=fit($("cv-resized"),d.resized.w,d.resized.h,520); g.drawImage(img,0,0,W,H); grid(g,W,H,d.token.cols,d.token.rows,"rgba(255,255,255,.55)",1); }

function drawPatch(){ const d=S, P=d.patch;
  const cv=$("cv-patch"), [g,W,H]=fit(cv,d.resized.w,d.resized.h,620); g.drawImage(img,0,0,W,H);
  grid(g,W,H,P.cols,P.rows,"rgba(255,255,255,.45)",.5);
  if(hoverP>=0 && hoverP<P.n){ const r=Math.floor(hoverP/P.cols), c=hoverP%P.cols;
    g.strokeStyle="#f59e0b"; g.lineWidth=2.5; g.strokeRect(c*W/P.cols,r*H/P.rows,W/P.cols,H/P.rows); }
}
function showPatch(rc){ const P=S.patch; hoverP=rc[0]*P.cols+rc[1]; drawPatch();
  const z=$("zoom").getContext("2d"); z.imageSmoothingEnabled=false; z.clearRect(0,0,168,168);
  z.drawImage(img, rc[1]*P.px, rc[0]*P.px, P.px, P.px, 0, 0, 168, 168);
  const tk=Math.floor(rc[0]/S.token.merge)*S.token.cols+Math.floor(rc[1]/S.token.merge);
  $("s3kv").innerHTML=`<span>patch</span><b>row ${rc[0]}, column ${rc[1]}</b><span>pixels</span><b>${P.px} × ${P.px} × 3 colours</b>`+
    `<span>numbers</span><b>${fmt(P.numbers)} (× ${P.frames} frames)</b><span>part of token</span><b>${tk}</b>`; }
$("cv-patch").addEventListener("mousemove",e=>{ if(!S) return; const P=S.patch, rc=cellAt($("cv-patch"),e,P.cols,P.rows);
  if(rc && rc[0]*P.cols+rc[1]!==hoverP) showPatch(rc); });

function drawTok(){ const d=S, T=d.token, E=d.encoder;
  const cv=$("cv-tok"), [g,W,H]=fit(cv,d.resized.w,d.resized.h,700); g.drawImage(img,0,0,W,H);
  g.fillStyle="rgba(255,255,255,.12)"; g.fillRect(0,0,W,H);
  grid(g,W,H,d.patch.cols,d.patch.rows,"rgba(255,255,255,.25)",.5);
  grid(g,W,H,T.cols,T.rows,"rgba(255,255,255,.9)",1.2);
  if($("t-win").checked){ const wt=E.window_tokens; g.strokeStyle="#2563eb"; g.lineWidth=2.5; g.beginPath();
    for(let c=wt;c<T.cols;c+=wt){const x=c*W/T.cols; g.moveTo(x,0); g.lineTo(x,H);}
    for(let r=wt;r<T.rows;r+=wt){const y=r*H/T.rows; g.moveTo(0,y); g.lineTo(W,y);} g.stroke(); }
  const cw=W/T.cols, ch=H/T.rows, upto = orderAnim>=0 ? orderAnim : ($("t-num").checked ? T.n : 0);
  g.font=`600 ${Math.max(8,Math.min(13,cw*0.36))}px -apple-system,Segoe UI,sans-serif`; g.textAlign="center"; g.textBaseline="middle";
  g.lineJoin="round"; g.lineWidth=3; g.strokeStyle="rgba(17,24,39,.75)"; g.fillStyle="#fff";
  for(let k=0;k<Math.min(upto,T.n);k++){ const r=Math.floor(k/T.cols), c=k%T.cols;
    g.strokeText(k,c*cw+cw/2,r*ch+ch/2); g.fillText(k,c*cw+cw/2,r*ch+ch/2); }
  if(hoverT>=0 && hoverT<T.n && orderAnim<0){ const r=Math.floor(hoverT/T.cols), c=hoverT%T.cols;
    g.strokeStyle="#f59e0b"; g.lineWidth=3; g.strokeRect(c*cw,r*ch,cw,ch); }
}
function showToken(rc){ const T=S.token, k=rc[0]*T.cols+rc[1]; hoverT=k; drawTok(); const Ps=S.positions;
  $("s4kv").innerHTML=`<span>token</span><b>${k}</b><span>row</span><b>${rc[0]} = ${k} ÷ ${T.cols}</b><span>column</span><b>${rc[1]} = ${k} mod ${T.cols}</b>`+
    `<span>pixels</span><b>x ${rc[1]*T.px}–${(rc[1]+1)*T.px-1}, y ${rc[0]*T.px}–${(rc[0]+1)*T.px-1}</b>`+
    `<span>made of patches</span><b>4 (2 × 2)</b><span>3D position</span><b>(${Ps.t}, ${Ps.start+rc[0]}, ${Ps.start+rc[1]})</b>`; }
$("cv-tok").addEventListener("mousemove",e=>{ if(!S) return; const T=S.token, rc=cellAt($("cv-tok"),e,T.cols,T.rows); if(rc) showToken(rc); });
["t-num","t-win"].forEach(id=>$(id).onchange=()=>drawTok());
$("order").onclick=()=>{ if(!S) return; orderAnim=0; const n=S.token.n, step=Math.max(1,Math.ceil(n/90));
  const iv=setInterval(()=>{ orderAnim+=step; if(orderAnim>=n){ clearInterval(iv); orderAnim=-1; } drawTok(); },22); };

// ---------------------------------------------------------------- causal test
function drawOccOne(cv, mapKey, outline){ const d=S, T=d.token;
  const [g,W,H]=fit(cv,d.resized.w,d.resized.h,560); const cw=W/T.cols, ch=H/T.rows;
  g.drawImage(img,0,0,W,H);
  if(occ){ // the changed square shown inverted, as the model saw it
    g.save(); g.globalCompositeOperation="difference"; g.fillStyle="#fff"; g.fillRect(occ.c*cw,occ.r*ch,cw,ch); g.restore();
    const m=occ[mapKey], mx=Math.max(...m);
    for(let k=0;k<m.length;k++){ const a=m[k]/mx; if(!(a>=0.02)) continue;
      g.fillStyle=`rgba(${HEAT},${(0.85*a).toFixed(3)})`; g.fillRect((k%T.cols)*cw,Math.floor(k/T.cols)*ch,cw,ch); }
  }
  grid(g,W,H,T.cols,T.rows,"rgba(255,255,255,.5)",.8);
  Object.values(tested).forEach(t=>{ g.strokeStyle=outline(t)?"#15803d":"#b91c1c"; g.lineWidth=2; g.strokeRect(t.c*cw+1,t.r*ch+1,cw-2,ch-2); });
  if(occ){ g.strokeStyle="#111827"; g.lineWidth=4; g.strokeRect(occ.c*cw-1,occ.r*ch-1,cw+2,ch+2);
    const lbl="you changed this", tx=Math.min(Math.max(occ.c*cw+cw/2,58),W-58), ty=occ.r*ch>22?occ.r*ch-8:occ.r*ch+ch+16;
    g.font="600 12px -apple-system,Segoe UI,sans-serif"; g.textAlign="center"; g.textBaseline="middle";
    const tw=g.measureText(lbl).width+12; g.fillStyle="#111827"; g.fillRect(tx-tw/2,ty-10,tw,20); g.fillStyle="#fff"; g.fillText(lbl,tx,ty); }
}
function drawOcc(){ if(!S||!img.complete) return;
  drawOccOne($("cv-occ-off"),"routing_map",t=>t.routing_ok);
  drawOccOne($("cv-occ"),"map",t=>t.correct); }
function reacted(m){ const mx=Math.max(...m); return m.filter(v=>v>=S.react_frac*mx).length; }
function onTrial(res,recorded){
  occ=res; if(recorded) tested[res.r+","+res.c]=res; drawOcc();
  const nA=reacted(res.routing_map), nB=reacted(res.map);
  $("resA").className="result "+(res.routing_ok?"ok":"bad");
  $("resA").textContent = res.routing_ok
    ? `${nA} note reacted: exactly the square you changed ✓ Our label for it is right.`
    : `✗ The wrong note reacted: our label for this square would be wrong.`;
  $("resB").className="result";
  $("resB").textContent = `${nB} note${nB===1?"":"s"} reacted: your square ${res.correct?"most":"and its neighbours (a neighbour even reacted more)"}`
    + (res.correct && nB>1 ? ", plus its neighbours a little. The notes share information." : ".");
  const T=Object.values(tested), n=T.length, a=T.filter(t=>t.routing_ok).length, b=T.filter(t=>t.correct).length;
  $("tallyA").textContent=`${a} / ${n}`; $("tallyA").style.color = a===n?"var(--ok)":"var(--bad)";
  $("tallyB").textContent=`${b} / ${n}`;
  $("tallyAsub").textContent=`random squares where only the predicted token changed (of ${S.squares.length})`;
  $("occkv").innerHTML=`<span>square changed</span><b>row ${res.r}, column ${res.c} → predicted token ${res.target}</b>`+
    `<span>mixing off</span><b style="color:${res.routing_ok?"var(--ok)":"var(--bad)"}">${res.routing_changed} token${res.routing_changed===1?"":"s"} changed`+
    `${res.routing_ok?" ✓ only the predicted one":" ✗"}; largest other change ${res.routing_other_max.toExponential(1)}</b>`+
    `<span>real encoder</span><b>token that reacted most: ${res.top} ${res.correct?"(the changed square's own)":"(a different square)"}`+
    `${res.margin!=null?`, ${res.margin.toFixed(2)}× the next-largest`:""}</b>`;
  if(res.final){ const f=res.final, ids=f.checks.map(c=>c.id);
    S.checks=S.checks.filter(c=>!ids.includes(c.id)).concat(f.checks);
    $("bars").innerHTML=f.ring.slice(0,8).map((v,i)=>`<div style="height:${Math.max(2,v*100)}%;opacity:${0.35+0.65*v}"><i>${v.toFixed(2)}</i><span>${i===0?"itself":i+" away"}</span></div>`).join("");
    ids.forEach(proof); checksTable(); }
}
["cv-occ","cv-occ-off"].forEach(id=>$(id).addEventListener("click",async e=>{ if(!S) return; const T=S.token, rc=cellAt($(id),e,T.cols,T.rows); if(!rc) return;
  if(LIVE){ try{ setStatus(`Testing row ${rc[0]}, column ${rc[1]}…`); onTrial(await api("/api/stage1/occlude",{r:rc[0],c:rc[1],record:false}),false); setStatus(""); }
           catch(err){ setStatus(err.message,true); } }
  else if(DATA.allmaps){ const k=rc[0]*T.cols+rc[1], m=DATA.allmaps[k]; if(m) onTrial(m,false); } }));

const idKey=id=>{const m=id.match(/^(\d+)\.(\d+)([a-z]?)$/); return m?[+m[1],+m[2],m[3]]:[0,0,id];};
const byId=(a,b)=>{const x=idKey(a),y=idKey(b); return x[0]-y[0]||x[1]-y[1]||(x[2]<y[2]?-1:x[2]>y[2]?1:0);};
function ids_all(){ return [...new Set(S.checks.map(c=>c.id).concat(S.pending_checks.map(c=>c.id)))].sort(byId); }
function checksTable(){
  // banner: computed from the checks, never written by hand. Measurements are not pass/fail.
  const judged=S.checks.filter(c=>c.pass!==null), ok=judged.filter(c=>c.pass).length;
  const pendingJudged=S.pending_checks.filter(p=>!S.checks.find(c=>c.id===p.id)).length;
  const total=judged.length+pendingJudged;
  $("banner").innerHTML = pendingJudged
    ? `<span class="pill wait" style="font-size:14px;padding:6px 12px">checking</span><div><div class="big2">${ok} of ${total} checks passed so far</div><div class="note">the causal test is still running</div></div><a href="#checks">all checks ↓</a>`
    : `<span class="pill ${ok===total?"ok":"bad"}" style="font-size:14px;padding:6px 12px">${ok===total?"VERIFIED":"FAILED"}</span><div><div class="big2">Stage 1: ${ok} of ${total} checks passed</div><div class="note">${ok===total?"We know exactly which square of your image each stored token stands for.":"At least one check failed: see the table below before trusting later stages."}</div></div><a href="#checks">all checks ↓</a>`;
  $("checks").innerHTML="<tr><th>#</th><th>Check</th><th>Result</th><th>Evidence</th></tr>"+ids_all().map(id=>{
    const c=S.checks.find(x=>x.id===id);
    if(!c){ const p=S.pending_checks.find(x=>x.id===id); return `<tr><td>${id}</td><td>${p.name}</td><td><span class="pill wait">running</span></td><td></td></tr>`; }
    const pill=c.pass===null?`<span class="pill info">MEASURED</span>`:`<span class="pill ${c.pass?"ok":"bad"}">${c.pass?"PASS":"FAIL"}</span>`;
    return `<tr><td>${c.id}</td><td>${c.name}</td><td>${pill}</td><td class="note">${c.detail}</td></tr>`; }).join("");
}
window.addEventListener("resize",()=>{ if(S&&img) drawAll(); });

// =============================================================== stage switching
let curStage=1;
const UP={1:["Stage 1 · Image → visual tokens","How Qwen cuts your image into the pieces it remembers, and the proof that we know exactly which piece is which."],
          2:["Stage 2 · Attention","Ask a question. The model answers it, and you can click any word to see which parts of the photo it looked at while choosing that word."]};
function setStage(n){ curStage=n;
  document.querySelectorAll("nav a[data-stage]").forEach(a=>a.classList.toggle("on",+a.dataset.stage===n));
  $("uptitle").textContent=UP[n][0]; $("upnote").textContent=UP[n][1]; $("run").textContent=`Run stage ${n}`;
  $("out").classList.toggle("hidden", !(n===1 && S)); $("out2").classList.toggle("hidden", !(n===2 && S2));
  if(n===1 && S) drawAll(); if(n===2 && S2) drawAtt(); }
document.querySelectorAll("nav a[data-stage]").forEach(a=>a.onclick=e=>{e.preventDefault(); setStage(+a.dataset.stage);});

// =============================================================== stage 2: attention
let S2=null, L2=null, LM={}, wi=1, selPatch=-1, img2=null, play2=null;
const ATT="37,99,235";
const escH=t=>t.replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
async function runStage2(){
  if(!dataURL) return;
  $("run").disabled=true; setStatus("The model is answering your question while we watch every layer (about 30–60 seconds)…");
  try{ S2=await api("/api/stage2",{image:dataURL,question:$("question").value}); }
  catch(e){ setStatus(e.message,true); $("run").disabled=false; return; }
  LM={}; await render2(S2, true); setStatus("Done. Click any word of the answer."); $("run").disabled=false;
}
async function getLayer(L){ if(LM[L]) return LM[L];
  if(LIVE){ const r=await fetch("/api/stage2/layer?L="+L), j=await r.json(); if(!r.ok) throw new Error(j.error); LM[L]=j; }
  else LM[L]=DATA.stage2_layers[L];
  return LM[L]; }
async function render2(d, show){
  img2=new Image(); await new Promise(r=>{img2.onload=r; img2.src="data:image/jpeg;base64,"+d.image;});
  $("s2q").innerHTML=`Your question: “${escH(d.question)}”`;
  $("words").innerHTML=d.tokens.map((t,i)=>`<span data-i="${i}" class="${i===0?"first":""}">${escH(t)}</span>`).join("");
  $("words").querySelectorAll("span").forEach(sp=>sp.onclick=()=>{const i=+sp.dataset.i; if(i>=1 && i<=d.n_maps) selectWord(i);});
  $("layer").max=d.layers.length-1; $("nlayers").textContent=d.layers.length;
  $("layer").value=Math.max(0,d.layers.indexOf(d.display_layer));
  wi=1; selPatch=-1; L2=d.layers[+$("layer").value];
  await getLayer(L2);
  if(show){ $("out2").classList.remove("hidden"); setStage(2); }
  selectWord(1); drawSplits(); math2(); proofs2();
}
function selectWord(i){ wi=i;
  $("words").querySelectorAll("span").forEach(sp=>sp.classList.toggle("on",+sp.dataset.i===i));
  $("curword").textContent=S2.tokens[i].trim()||"(space)";
  drawAtt(); drawPatchChart(); math2(); }
$("layer").oninput=async()=>{ L2=S2.layers[+$("layer").value]; await getLayer(L2); drawAtt(); drawPatchChart(); drawSplits(); };

function drawAtt(){ if(!S2||!img2||!LM[L2]) return; const [R,C]=S2.grid, M=LM[L2], j=wi-1;
  const [g,W,H]=fit($("cv-att"),img2.naturalWidth,img2.naturalHeight,640), cw=W/C, ch=H/R;
  g.drawImage(img2,0,0,W,H); g.fillStyle="rgba(255,255,255,.25)"; g.fillRect(0,0,W,H);
  const q=M.q[j];
  for(let k=0;k<q.length;k++){ const a=q[k]/255; if(a<0.03) continue;
    g.fillStyle=`rgba(${ATT},${(0.85*a).toFixed(3)})`; g.fillRect((k%C)*cw,Math.floor(k/C)*ch,cw,ch); }
  grid(g,W,H,C,R,"rgba(255,255,255,.35)",.6);
  if(selPatch>=0){ g.strokeStyle="#f59e0b"; g.lineWidth=3; g.strokeRect((selPatch%C)*cw,Math.floor(selPatch/C)*ch,cw,ch); }
  $("layerval").textContent=L2;
  const share=M.share[j];
  $("attshare").textContent=`${(share*100).toFixed(1)}% on the photo`;
  $("attsharesub").textContent=`While choosing “${S2.tokens[wi].trim()}”, this layer put ${(share*100).toFixed(1)}% of its attention on the photo's ${S2.n_image} patches; the rest went to your question, the template and the words written so far.`;
  const top=[...q.keys()].sort((a,b)=>q[b]-q[a]).slice(0,5);
  $("toplist").innerHTML=top.map(k=>`<span>row ${Math.floor(k/C)}, column ${k%C}</span><b>${(q[k]/255*M.mx[j]*100).toFixed(2)}% of this word's attention</b>`).join("");
}
$("cv-att").addEventListener("click",e=>{ if(!S2) return; const [R,C]=S2.grid, rc=cellAt($("cv-att"),e,C,R); if(!rc) return;
  selPatch=rc[0]*C+rc[1]; drawAtt(); drawPatchChart(); });

function drawPatchChart(){ const svg=$("patchsvg"); if(!S2||selPatch<0||!LM[L2]){ svg.innerHTML=""; return; }
  const [R,C]=S2.grid, M=LM[L2], n=S2.n_maps, W=svg.clientWidth||900, H=210, pad={l:52,r:12,t:12,b:34};
  const vals=[...Array(n).keys()].map(j=>M.q[j][selPatch]/255*M.mx[j]*100);
  const mx=Math.max(...vals,1e-9), X=j=>pad.l+(W-pad.l-pad.r)*(j+.5)/n, bw=Math.max(1,(W-pad.l-pad.r)/n-1), Y=v=>pad.t+(H-pad.t-pad.b)*(1-v/mx);
  let s="";
  for(let t=0;t<=4;t++){ const y=pad.t+(H-pad.t-pad.b)*(1-t/4);
    s+=`<line x1="${pad.l}" x2="${W-pad.r}" y1="${y}" y2="${y}" stroke="#eef0f3"/><text x="${pad.l-6}" y="${y+4}" font-size="11" text-anchor="end" fill="#9ca3af">${(mx*t/4).toFixed(2)}%</text>`; }
  vals.forEach((v,j)=>{ s+=`<rect x="${X(j)-bw/2}" y="${Y(v)}" width="${bw}" height="${H-pad.b-Y(v)}" fill="${j===wi-1?"#f59e0b":"rgba(37,99,235,.75)"}"><title>${escH(S2.tokens[j+1])}: ${v.toFixed(3)}%</title></rect>`; });
  s+=`<text x="${(W)/2}" y="${H-6}" font-size="11" text-anchor="middle" fill="#9ca3af">each bar = one word of the answer (orange = the word selected above); height = attention this patch got</text>`;
  svg.setAttribute("viewBox",`0 0 ${W} ${H}`); svg.innerHTML=s;
  $("patchlead").innerHTML=`Patch at <b>row ${Math.floor(selPatch/C)}, column ${selPatch%C}</b> (layer ${L2}): how much attention it got at every word of the answer. Hover a bar to see the word. This is the signal VisRetain's later stages use to decide what to keep.`;
}

function drawSplits(){ if(!S2) return;
  const cols=["#9ca3af","#a78bfa","#2563eb","#f59e0b"];
  $("splits").innerHTML=S2.layers.map(L=>{ const v=S2.split[L]||S2.split[String(L)];
    return `<div class="splitrow ${L===L2?"on":""}" data-l="${L}"><span>layer ${L}</span><div class="splitbar">${v.map((x,i)=>`<div style="width:${(x*100).toFixed(2)}%;background:${cols[i]}" title="${(x*100).toFixed(1)}%"></div>`).join("")}</div></div>`; }).join("");
  $("splits").querySelectorAll(".splitrow").forEach(r=>r.onclick=async()=>{ const L=+r.dataset.l; $("layer").value=S2.layers.indexOf(L); L2=L; await getLayer(L); drawAtt(); drawPatchChart(); drawSplits(); });
}

function math2(){ if(!S2) return; const h=S2.heads, S=S2.prompt_len+wi, sc=Math.pow(h.dim,-0.5);
  $("attmath").innerHTML=`<span class="c">For the word being chosen, the model has a query vector q (what it is looking for); every stored note i has a key k_i.</span>\n`+
   `<b>score_i = (q · k_i) × ${h.dim}^-½ = (q · k_i) ÷ ${(1/sc).toFixed(2)}</b>\n`+
   `<b>attention_i = e^score_i ÷ Σ_j e^score_j</b>   <span class="c">(softmax: all attention adds up to 100%)</span>\n\n`+
   `<span class="c">While choosing “${escH(S2.tokens[wi].trim())}”, it looks over</span> <b>${S} stored notes</b> = ${S2.prompt_len} from the prompt (incl. ${S2.n_image} photo patches) + ${wi} answer word${wi===1?"":"s"} so far.\n`+
   `<span class="c">Each layer has</span> <b>${h.q} query heads sharing ${h.kv} key heads</b> <span class="c">(${h.q/h.kv} queries per key head), ${h.dim} numbers per head. The maps above average the ${h.q} heads.</span>`;
}
function proofs2(){ S2.checks.forEach(c=>{ const el=$("p2-"+c.id); if(!el) return;
    el.innerHTML=`<div class="proof ${c.pass?"ok":"bad"}"><span class="tick">${c.pass?"✓":"✗"}</span><div><b>Check ${c.id}: ${c.name}</b><small>${c.detail}</small></div></div>`; });
  const ok=S2.checks.filter(c=>c.pass).length, n=S2.checks.length;
  $("banner2").innerHTML=`<span class="pill ${ok===n?"ok":"bad"}" style="font-size:14px;padding:6px 12px">${ok===n?"VERIFIED":"FAILED"}</span><div><div class="big2">Stage 2: ${ok} of ${n} checks passed</div><div class="note">${ok===n?"The attention shown here is exactly the attention the model computes.":"At least one check failed: see the table below."}</div></div><a href="#checks2">all checks ↓</a>`;
  $("checks2").innerHTML="<tr><th>#</th><th>Check</th><th>Result</th><th>Evidence</th></tr>"+S2.checks.map(c=>
    `<tr><td>${c.id}</td><td>${c.name}</td><td><span class="pill ${c.pass?"ok":"bad"}">${c.pass?"PASS":"FAIL"}</span></td><td class="note">${c.detail}</td></tr>`).join("");
}
$("play2").onclick=()=>{ if(!S2) return; if(play2){ clearInterval(play2); play2=null; $("play2").textContent="▶ Play through the answer"; return; }
  if(wi>=S2.n_maps) selectWord(1);
  $("play2").textContent="❚❚ Pause";
  play2=setInterval(()=>{ if(wi>=S2.n_maps){ clearInterval(play2); play2=null; $("play2").textContent="▶ Play through the answer"; return; } selectWord(wi+1); },260); };
window.addEventListener("resize",()=>{ if(S2&&curStage===2){ drawAtt(); drawPatchChart(); } });

// offline build: a precomputed run, replayed
if(!LIVE && DATA.stage2){ S2=DATA.stage2; render2(S2, false); }
if(!LIVE && DATA.stage1){ S=DATA.stage1; render(S);
  (async()=>{ for(const t of DATA.trials){ await new Promise(r=>setTimeout(r,250)); onTrial(t,true); } })(); }
</script>
</body>
</html>
"""

# Guard: element ids must be unique (a duplicate once sent stage 2's math into stage 1's box).
import re as _re
_ids = _re.findall(r'\bid="([^"$]+)"', PAGE)
assert len(_ids) == len(set(_ids)), sorted({i for i in _ids if _ids.count(i) > 1})
