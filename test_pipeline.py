"""End-to-end checks on the REAL model. Run before trusting any number or demo.

    python3 test_pipeline.py        # ~5 minutes on an M-series Mac, needs the GPU free

Each check targets a specific way the instrument could silently produce wrong results:

  T1 non-interference   our hooks must not change a single token of the model's answer
  T2 determinism        the same input must give byte-identical eviction timelines
  T3 live == batch      the live app's streaming must not alter what is computed
  T4 image scale cap    a huge photo must not explode the token count
  T5 record integrity   budgets respected, death steps in range, lengths consistent
  T6 live app over HTTP the real server: a full run, plus bad inputs and a busy GPU

Writes TEST_REPORT.md with every result, pass or fail.
"""

import base64, io, json, queue, threading, time, traceback, urllib.request, urllib.error
from http.server import ThreadingHTTPServer
from pathlib import Path

import torch
from PIL import Image
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.integrations.sdpa_attention import sdpa_attention_forward

import run_timeline as rt
import app

MAX_NEW = 80   # shorter than the config default only to keep the test fast
RESULTS = []


def check(name, ok, detail):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def gen_ids(model, proc, dev, img, q):
    """Plain greedy generation, returning token ids (no logging involved)."""
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q +
            " Explain your reasoning step by step, describing what you see in detail."}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=[img], return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False)
    return out[0, inp["input_ids"].shape[1]:].tolist()


def deaths(rec):
    return {L["layer"]: L["curves"] for L in rec["layers"]}


def main():
    model, proc, dev = rt.load_model()
    item = rt.build_items(1)[0]
    img, q = item["image"], item["question"]
    shim = ALL_ATTENTION_FUNCTIONS["visretain"]

    # ---- T1 non-interference --------------------------------------------------
    try:
        rec_a = rt.run_item(model, proc, dev, item, 0, MAX_NEW)      # hooks logging
        ALL_ATTENTION_FUNCTIONS["visretain"] = sdpa_attention_forward  # pure vanilla
        vanilla = gen_ids(model, proc, dev, img, q)
        ALL_ATTENTION_FUNCTIONS["visretain"] = shim
        # tokens are compared one by one, each decoded from the real generated ids
        vtoks = [proc.tokenizer.decode([t], skip_special_tokens=True) for t in vanilla]
        same = vtoks == rec_a["tokens"]
        check("T1 non-interference", same,
              f"answer generated with logging on is token-for-token identical to vanilla "
              f"attention ({len(vanilla)} tokens; identical={same})")
    except Exception as e:
        ALL_ATTENTION_FUNCTIONS["visretain"] = shim
        check("T1 non-interference", False, f"crashed: {e!r}"); traceback.print_exc()
        rec_a = None

    # ---- T2 determinism -------------------------------------------------------
    try:
        rec_b = rt.run_item(model, proc, dev, item, 0, MAX_NEW)
        same = rec_a is not None and deaths(rec_a) == deaths(rec_b) and rec_a["tokens"] == rec_b["tokens"]
        n_tok = sum(len(c) for L in deaths(rec_b).values() for c in L.values())
        check("T2 determinism", same,
              f"two runs, same input: identical eviction timelines across "
              f"{len(rec_b['layers'])} layers x 3 arms ({n_tok} token deaths compared)")
    except Exception as e:
        check("T2 determinism", False, f"crashed: {e!r}"); traceback.print_exc()
        rec_b = None

    # ---- T3 live streaming == batch ------------------------------------------
    try:
        qq = queue.Queue()
        rec_c = rt.run_item(model, proc, dev, item, 0, MAX_NEW,
                            streamer=app.LiveStreamer(qq, proc.tokenizer),
                            on_setup=lambda info: qq.put({"type": "setup"}))
        evs = []
        while not qq.empty():
            evs.append(qq.get())
        steps = [e for e in evs if e["type"] == "step"]
        same = rec_b is not None and deaths(rec_c) == deaths(rec_b) and rec_c["tokens"] == rec_b["tokens"]
        check("T3 live == batch", same,
              f"with the live streamer attached, record identical to batch run (identical={same})")
        # the streamed deaths must reproduce the final timeline at the display layer
        L = next(L for L in rec_c["layers"] if L["layer"] == app.SHOW_LAYER)
        rebuilt = {a: [-1] * len(L["curves"][a]) for a in rt.ARMS[1:]}
        for e in steps:
            for a, arr in e["died"].items():
                for i, s in arr:
                    rebuilt[a][i] = s
        check("T3b stream reproduces timeline", rebuilt == {a: L["curves"][a] for a in rebuilt},
              f"{len(steps)} streamed steps for {rec_c['n_steps']} generated tokens; "
              f"replaying the stream gives the final layer-{app.SHOW_LAYER} timeline")
        check("T3c one event per token", len(steps) == rec_c["n_steps"],
              f"{len(steps)} step events vs {rec_c['n_steps']} tokens")
    except Exception as e:
        check("T3 live == batch", False, f"crashed: {e!r}"); traceback.print_exc()

    # ---- T4 image scale cap --------------------------------------------------
    try:
        big = img.resize((4000, 3000))
        rec_d = rt.run_item(model, proc, dev, {"image": big, "question": q}, 0, 20)
        from cfg import CFG
        cap = CFG["image"]["max_pixels"] // (28 * 28)      # one visual token per 28x28 px
        check("T4 image scale cap", rec_d["n_visual"] <= cap,
              f"4000x3000 photo -> {rec_d['n_visual']} visual tokens "
              f"(cap ~{cap}; uncapped it would be ~15,000); grid {rec_d['grid']}")
        check("T4b short answer flagged", rec_d["too_short"] is True,
              f"20-token answer flagged too_short={rec_d['too_short']}")
    except Exception as e:
        check("T4 image scale cap", False, f"crashed: {e!r}"); traceback.print_exc()

    # ---- T5 record integrity -------------------------------------------------
    try:
        r = rec_b
        problems = []
        if len(r["tokens"]) != r["n_steps"]:
            problems.append("tokens != n_steps")
        for L in r["layers"]:
            for a, d in L["curves"].items():
                if any(not (x == -1 or 1 <= x <= r["n_steps"]) for x in d):
                    problems.append(f"L{L['layer']} {a}: death step out of range")
                alive = sum(1 for x in d if x == -1)
                if alive > r["budget"]:
                    problems.append(f"L{L['layer']} {a}: {alive} visual alive > budget {r['budget']}")
            if L["n_gun"] != len(L["gun_idx"]) or len(set(L["gun_idx"])) != L["n_gun"]:
                problems.append(f"L{L['layer']}: evidence set malformed")
        check("T5 record integrity", not problems,
              "budgets respected, death steps in range, evidence sets well-formed"
              if not problems else "; ".join(problems[:5]))
    except Exception as e:
        check("T5 record integrity", False, f"crashed: {e!r}"); traceback.print_exc()

    # ---- T6 the live app over real HTTP --------------------------------------
    try:
        app.STATE.update(model=model, proc=proc, dev=dev)
        app.PAGE_HTML = "<html>test</html>"
        srv = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
        port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{port}"

        def post(payload):
            req = urllib.request.Request(base + "/api/run", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return resp.status, json.loads(resp.read())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read())

        def b64(im, fmt="PNG"):
            buf = io.BytesIO(); im.save(buf, fmt)
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        code, body = post({"image": "data:image/png;base64,bm90IGFuIGltYWdl", "question": "x"})
        check("T6a rejects non-image", code == 400, f"HTTP {code}: {body.get('error')}")
        code, body = post({"image": b64(img.resize((20, 20))), "question": "x"})
        check("T6b rejects tiny image", code == 400, f"HTTP {code}: {body.get('error')}")

        app.MAX_NEW = MAX_NEW
        code, body = post({"image": b64(img), "question": ""})
        check("T6c starts a run", code == 200, f"HTTP {code}")
        code2, body2 = post({"image": b64(img), "question": "second"})
        check("T6d rejects a second concurrent run", code2 == 409, f"HTTP {code2}: {body2.get('error')}")

        evs, t0 = [], time.time()
        with urllib.request.urlopen(base + "/api/events", timeout=600) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if line.startswith("data: "):
                    ev = json.loads(line[6:]); evs.append(ev)
                    if ev["type"] in ("done", "error"):
                        break
        kinds = [e["type"] for e in evs]
        done = next((e for e in evs if e["type"] == "done"), None)
        ok = kinds[0] == "setup" and done is not None and done["item"] is not None
        check("T6e full live run over HTTP", ok,
              f"events: 1 setup, {kinds.count('step')} steps, "
              f"{'done' if done else 'NO done'} in {time.time()-t0:.0f}s; saved {done and done.get('saved')}")
        if done and done["item"]:
            check("T6f empty question gets a default", done["item"]["question"] == "Describe this image.",
                  f"question used: {done['item']['question']!r}")
        srv.shutdown()
    except Exception as e:
        check("T6 live app over HTTP", False, f"crashed: {e!r}"); traceback.print_exc()

    write_report()


def write_report():
    n_pass = sum(ok for _, ok, _ in RESULTS)
    lines = [f"# VisRetain test report", "",
             f"Run {time.strftime('%Y-%m-%d %H:%M')} on the real Qwen2.5-VL-3B-Instruct, "
             f"local (Apple M-series, MPS). Command: `python3 test_pipeline.py`.", "",
             f"**{n_pass} / {len(RESULTS)} checks passed.**", "",
             "| Check | Result | Detail |", "|---|---|---|"]
    for name, ok, detail in RESULTS:
        lines.append(f"| {name} | {'PASS' if ok else '**FAIL**'} | {detail} |")
    Path("TEST_REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"\n{n_pass}/{len(RESULTS)} passed -> TEST_REPORT.md")


if __name__ == "__main__":
    main()
