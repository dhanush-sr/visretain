"""VisRetain live app: drop in your own image, watch the model answer and evict, live.

    python -m visretain.app      # then open http://localhost:8765

Fully local and offline: the model runs on this machine, nothing is sent anywhere.
Standard library server, one run at a time (the GPU can hold one model run).

The experiment code is NOT duplicated here. Every run goes through
run_timeline.run_item, the same function the batch runs use, so a live result and a
batch result for the same image are the same computation (test_pipeline.py checks it).
The live stream only READS the eviction state as it changes; it never alters it.
"""

import base64, io, json, queue, sys, threading, time, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from transformers.generation.streamers import BaseStreamer

from . import run_timeline as rt
from . import build_demo as bd
from . import stages
from . import stages_page as sp

from .cfg import CFG, ROOT, RESULTS
PORT = CFG["server"]["port"]
SHOW_LAYER = CFG["display_layer"]   # the layer the live view draws
OUT = RESULTS / "live"                 # live uploads (git-ignored: may be personal photos)
MAX_UPLOAD = int(CFG["server"]["max_upload_mb"] * 1024 * 1024 * 4 / 3)  # base64 inflates by 4/3

# rt.CTX already holds the config's settings, identical to the batch runs.
MAX_NEW = CFG["generation"]["max_new_tokens"]

STATE = {"model": None, "proc": None, "dev": None, "busy": threading.Lock(),
         "q": None, "n": 0}


class LiveStreamer(BaseStreamer):
    """Called by model.generate once with the prompt, then once per new token.

    After token j has been produced, decode steps 1..j have run, so every eviction up
    to step j is final. We send the token plus the visual tokens newly evicted in each
    arm at the display layer. Read-only: it only looks at the logged state.
    """

    def __init__(self, q, tok):
        self.q, self.tok, self.first, self.j = q, tok, True, 0
        self.seen = {a: set() for a in rt.ARMS[1:]}

    def put(self, value):
        if self.first:                       # the prompt, not a generated token
            self.first = False
            return
        text = self.tok.decode(value.reshape(-1).tolist(), skip_special_tokens=True)
        died = {}
        st = rt.CTX["state"].get(SHOW_LAYER)
        if st is not None:
            vis = rt.CTX["vis_idx"]
            for a in self.seen:
                d = st[a]["death"][vis]
                now = set((d >= 0).nonzero(as_tuple=True)[0].tolist())
                new = sorted(now - self.seen[a])
                self.seen[a] |= now
                died[a] = [[i, int(d[i])] for i in new]   # local index, death step
        self.j += 1
        self.q.put({"type": "step", "step": self.j, "token": text, "died": died})

    def end(self):
        pass


def _run(img, question, q):
    try:
        STATE["n"] += 1
        i = STATE["n"]
        rec = rt.run_item(STATE["model"], STATE["proc"], STATE["dev"],
                          {"image": img, "question": question}, i, MAX_NEW,
                          streamer=LiveStreamer(q, STATE["proc"].tokenizer),
                          on_setup=lambda info: q.put({"type": "setup", **info}))
        OUT.mkdir(exist_ok=True)
        stamp = time.strftime("%m%d_%H%M%S")
        (OUT / f"item_{stamp}.json").write_text(json.dumps(rec))
        item = bd.to_item(rec)
        if item is not None:
            item["id"] = f"live-{i}"
            item["too_short"] = rec.get("too_short", False)
        q.put({"type": "done", "item": item, "saved": f"{OUT}/item_{stamp}.json",
               "secs": rec["secs"], "too_short": rec.get("too_short", False)})
    except Exception as e:
        traceback.print_exc()
        q.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
    finally:
        STATE["busy"].release()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):       # keep the terminal quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html", "/stages"):
            self._send(200, STAGES_HTML, "text/html; charset=utf-8")
        elif self.path == "/full":
            self._send(200, PAGE_HTML, "text/html; charset=utf-8")
        elif self.path.startswith("/api/events"):
            self._events()
        elif self.path.startswith("/api/stage2/layer"):
            self._stage2_layer()
        elif self.path == "/favicon.ico":
            self.send_response(204); self.end_headers()   # no icon; avoids a 404
        elif self.path == "/api/status":
            self._send(200, json.dumps({"busy": STATE["busy"].locked(),
                                        "ready": STATE["model"] is not None}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            raise ValueError("empty request")
        if n > MAX_UPLOAD:
            raise ValueError(f"image too large (max {CFG['server']['max_upload_mb']} MB)")
        return json.loads(self.rfile.read(n))

    @staticmethod
    def _image(body):
        """Decode and validate an uploaded image. One place for every endpoint."""
        from PIL import Image
        try:
            b64 = body["image"].split(",", 1)[-1]
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        except Exception as e:
            raise ValueError(f"could not read that image ({type(e).__name__})")
        if min(img.size) < 56:
            raise ValueError("image is too small (need at least 56 px per side)")
        return img

    def do_POST(self):
        try:
            body = self._body()
        except ValueError as e:
            return self._send(413, json.dumps({"error": str(e)}))
        except Exception:
            return self._send(400, json.dumps({"error": "bad request"}))
        if self.path == "/api/run":
            return self._full_run(body)
        if self.path in ("/api/stage1", "/api/stage1/occlude"):
            return self._stage1(body)
        if self.path == "/api/stage2":
            return self._stage2(body)
        self._send(404, json.dumps({"error": "not found"}))

    def _full_run(self, body):
        try:
            img = self._image(body)
        except ValueError as e:
            return self._send(400, json.dumps({"error": str(e)}))
        question = (body.get("question") or "").strip() or CFG["generation"]["default_question"]
        if not STATE["busy"].acquire(blocking=False):
            return self._send(409, json.dumps({"error": "a run is already in progress, wait for it to finish"}))
        STATE["q"] = queue.Queue()
        threading.Thread(target=_run, args=(img, question, STATE["q"]), daemon=True).start()
        self._send(200, json.dumps({"ok": True}))

    def _stage1(self, body):
        """Stage 1 calls are short; wait briefly for the GPU instead of failing."""
        if not STATE["busy"].acquire(timeout=30):
            return self._send(409, json.dumps({"error": "the model is busy with a full run, try again shortly"}))
        try:
            S = STATE["stage1"]
            if self.path == "/api/stage1":
                try:
                    img = self._image(body)
                except ValueError as e:
                    return self._send(400, json.dumps({"error": str(e)}))
                return self._send(200, json.dumps(S.run(img, (body.get("question") or "").strip() or None)))
            if getattr(S, "grid_rc", None) is None:
                return self._send(400, json.dumps({"error": "run stage 1 on an image first"}))
            R, C = S.grid_rc
            r, c = int(body.get("r", -1)), int(body.get("c", -1))
            if not (0 <= r < R and 0 <= c < C):
                return self._send(400, json.dumps({"error": f"square out of range (grid is {R} x {C})"}))
            return self._send(200, json.dumps(S.occlude(r, c, record=bool(body.get("record")))))
        except Exception as e:
            traceback.print_exc()
            return self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}))
        finally:
            STATE["busy"].release()

    def _stage2(self, body):
        """Answer the question with every attention layer observed (~30 s)."""
        try:
            img = self._image(body)
        except ValueError as e:
            return self._send(400, json.dumps({"error": str(e)}))
        if not STATE["busy"].acquire(timeout=30):
            return self._send(409, json.dumps({"error": "the model is busy, try again shortly"}))
        try:
            q = (body.get("question") or "").strip() or None
            return self._send(200, json.dumps(STATE["stage2"].run(img, q)))
        except Exception as e:
            traceback.print_exc()
            return self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}))
        finally:
            STATE["busy"].release()

    def _stage2_layer(self):
        from urllib.parse import urlparse, parse_qs
        S2 = STATE["stage2"]
        if getattr(S2, "maps", None) is None:
            return self._send(400, json.dumps({"error": "run stage 2 first"}))
        try:
            L = int(parse_qs(urlparse(self.path).query)["L"][0])
            return self._send(200, json.dumps(S2.layer_map(L)))
        except (KeyError, ValueError):
            return self._send(400, json.dumps({"error": f"layer must be one of {sorted(S2.maps)}"}))

    def _events(self):
        q = STATE["q"]
        if q is None:
            return self._send(404, json.dumps({"error": "no run started"}))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while True:
                try:
                    ev = q.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": keep-alive\n\n"); self.wfile.flush()
                    continue
                self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode()); self.wfile.flush()
                if ev["type"] in ("done", "error"):
                    break
        except (BrokenPipeError, ConnectionResetError):
            pass                             # viewer closed the tab; the run carries on


def main():
    global PAGE_HTML, STAGES_HTML
    # the live page = the static demo page, plus the pilot images, in live mode
    data = bd.page_data(RESULTS / "timeline", RESULTS / "ranking")
    PAGE_HTML = (bd.PAGE.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
                 .replace("__LIVE__", "true"))

    # stages page: sample buttons use the ORIGINAL pilot photos (cached locally, works offline),
    # so stage 1 shows the real resize rather than an already-resized image
    # The cached files' bytes are sent unchanged (no re-encoding), so a sample run on the
    # page sees exactly the pixels the batch tests saw.
    cache = ROOT / CFG["pilot"]["cache_dir"]
    samples = [{"img": base64.b64encode((cache / f"{it['id']}.jpg").read_bytes()).decode(),
                "label": it["question"]} for it in rt.build_items(len(CFG["pilot"]["items"]))]
    STAGES_HTML = (sp.PAGE.replace("__DATA__", json.dumps({"samples": samples}).replace("</", "<\\/"))
                   .replace("__LIVE__", "true"))

    print("Loading the model once (about 20 seconds)...", flush=True)
    STATE["model"], STATE["proc"], STATE["dev"] = rt.load_model()
    STATE["stage1"] = stages.Stage1(STATE["model"], STATE["proc"], STATE["dev"])
    STATE["stage2"] = stages.Stage2(STATE["model"], STATE["proc"], STATE["dev"])
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"\nReady. Open  http://localhost:{PORT}  in your browser.\n"
          f"  /      stages, verified one by one (start here)\n"
          f"  /full  the full pipeline on any image\n"
          f"Ctrl+C here to stop.", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
