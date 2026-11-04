"""Drive the live app in a real browser, like a user would. Needs app.py running.

    python -m visretain.app &       # in another terminal
    python -m tests.test_browser

Checks the parts test_pipeline.py cannot see: the page itself. Appends to reports/TEST_REPORT.md.
Uses the system Google Chrome through Playwright (no browser download needed).
"""

import sys, time, urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://localhost:8765/full"     # "/" is now the stages page
from visretain.cfg import DEMO, RESULTS as RESULTS_DIR, REPORTS   # RESULTS below = this file's checks
SHOTS = DEMO / "screens"; SHOTS.mkdir(parents=True, exist_ok=True)
IMG_URL = "http://images.cocodataset.org/val2017/000000000785.jpg"
QUESTION = "What is the person wearing, what are they doing, and what is around them?"
RESULTS = []


def check(name, ok, detail):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def main():
    img_path = SHOTS / "_test_input.jpg"
    if not img_path.exists():
        img_path.write_bytes(urllib.request.urlopen(IMG_URL, timeout=30).read())

    errors = []
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        page = b.new_page(viewport={"width": 1440, "height": 1300})
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.on("console", lambda m: m.type == "error" and errors.append(f"console: {m.text}"))

        page.goto(URL)
        check("B1 live card shown", page.is_visible("#livecard"), "upload card visible when served by app.py")
        check("B2 pilot images present", page.locator("#item option").count() >= 8,
              f"{page.locator('#item option').count()} images in the picker")

        # static playback works: Play advances the step
        page.click("#play"); time.sleep(1.5); page.click("#play")
        s = int(page.input_value("#step"))
        check("B3 playback", s > 0, f"Play advanced to step {s}")

        # a user picks a file, types a question, presses Run
        page.set_input_files("#file", str(img_path))
        page.wait_for_function("!document.getElementById('lrun').disabled", timeout=5000)
        page.fill("#lq", QUESTION)
        page.click("#lrun")
        page.wait_for_function("document.getElementById('lstatus').textContent.includes('step')",
                               timeout=120000)
        time.sleep(8)
        page.screenshot(path=str(SHOTS / "live_midrun.png"))
        mid = page.evaluate("""() => { const it = DATA.items[DATA.items.length-1];
            return {steps: it.steps, toks: it.tokens.length,
                    dead: Object.values(it.layers['15'].deaths).map(d => d.filter(x => x >= 0).length)}; }""")
        check("B4 live updates arrive", mid["steps"] > 0 and mid["toks"] == mid["steps"],
              f"mid-run: step {mid['steps']}, {mid['toks']} tokens shown, evicted so far per arm {mid['dead']}")
        check("B5 evidence withheld until the end",
              page.inner_text("#ev-fp16").strip() == "after the answer",
              f"evidence counter mid-run reads {page.inner_text('#ev-fp16')!r}")

        # switch to another image mid-run, then back: live data must stay on the live item
        page.select_option("#item", "0"); time.sleep(4)
        pilot_tokens = page.evaluate("DATA.items[0].tokens.length")
        live_idx = page.evaluate("DATA.items.length-1")
        page.select_option("#item", str(live_idx))
        after = page.evaluate("""() => { const it = DATA.items[DATA.items.length-1];
            return {steps: it.steps, toks: it.tokens.length}; }""")
        check("B6 switching images mid-run is safe",
              after["toks"] == after["steps"] and after["steps"] >= mid["steps"],
              f"live item kept updating while another image was shown "
              f"(step {mid['steps']} -> {after['steps']}); pilot image untouched ({pilot_tokens} tokens)")

        page.wait_for_function("document.getElementById('lstatus').textContent.startsWith('Done')",
                               timeout=300000)
        time.sleep(1)
        page.screenshot(path=str(SHOTS / "live_done.png"))
        ev = page.inner_text("#ev-fp16").strip()
        layers = page.locator("#layer option").count()
        check("B7 evidence revealed at the end", "/" in ev and layers == 8,
              f"FP16 evidence counter {ev!r}; layer picker has {layers} layers; status: "
              + page.inner_text("#lstatus")[:80])
        check("B8 charts drawn", page.locator("#survsvg path").count() >= 3,
              f"{page.locator('#survsvg path').count()} survival lines drawn")

        # results tab
        page.click("#t-res"); time.sleep(0.5)
        page.screenshot(path=str(SHOTS / "results.png"), full_page=True)
        from visretain import build_demo
        want = f"{build_demo.summarise_timeline(RESULTS_DIR / 'timeline')['fp16']*100:.1f}%"
        check("B9 results tab", want in page.inner_text("#res"),
              f"results tab shows the FP16 evidence-loss figure computed from the data ({want})")

        check("B10 no JavaScript errors", not errors, "none" if not errors else "; ".join(errors[:3]))
        b.close()

    lines = ["", "## Browser checks (live app, real Chrome)", "",
             f"**{sum(ok for _,ok,_ in RESULTS)} / {len(RESULTS)} passed.** "
             f"Screenshots in `demo/screens/`.", "", "| Check | Result | Detail |", "|---|---|---|"]
    lines += [f"| {n} | {'PASS' if ok else '**FAIL**'} | {d} |" for n, ok, d in RESULTS]
    rep = REPORTS / "TEST_REPORT.md"
    old = rep.read_text() if rep.exists() else ""
    old = old.split("\n## Browser checks")[0].rstrip("\n") + "\n"   # replace, don't append twice
    rep.write_text(old + "\n".join(lines) + "\n")
    print(f"\n{sum(ok for _,ok,_ in RESULTS)}/{len(RESULTS)} passed")
    sys.exit(0 if all(ok for _, ok, _ in RESULTS) else 1)


if __name__ == "__main__":
    main()
