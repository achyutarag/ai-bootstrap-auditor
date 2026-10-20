"""
Regenerates the paper's app screenshots (paper/figures/*.png) from a running app.

    streamlit run audit_toolkit_app.py          # in one terminal
    python scripts/make_figures.py [--url http://localhost:8501]

Uses Playwright with the locally installed Chrome (pip install playwright; no browser
download needed). Each screenshot opens one of the app's demo links, waits until the
results are rendered, crops the relevant region and adds numbered callouts that the
paper's walkthrough (Sec. "Demonstration Scenario") refers to.
"""
import argparse
import os

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "figures")
SCALE = 2          # device pixel ratio -> print-quality PNGs
ACCENT = (200, 30, 45)


def box(page, text, exact=False):
    """Bounding box (CSS px, page coordinates) of the first visible element with this text."""
    # Streamlit scrolls its main panel internally, not the window, so the viewport is made
    # tall enough (see __main__) that everything is on screen and viewport coords suffice.
    b = page.get_by_text(text, exact=exact).first.bounding_box()
    return {"x": b["x"], "y": b["y"], "w": b["width"], "h": b["height"]}


def capture(page, url, ready_text, segments, name, pad=12, gap=16, content_texts=(), width=None):
    """Screenshot one or more vertical segments of the page and stack them.

    segments: list of (top_text, bottom_text, [(callout_number, anchor_text), ...]).
    Each segment spans from its top anchor to the bottom of its bottom anchor; callouts
    are numbered circles in the left gutter, level with their anchor element.
    """
    page.goto(url)
    page.get_by_text(ready_text).first.wait_for(timeout=120_000)
    # tables first show same-height loading placeholders, so wait for real cell contents
    for text in content_texts:
        page.get_by_text(text, exact=True).first.wait_for(timeout=60_000)
    # Streamlit keeps re-laying-out for a moment after the last element appears: wait until
    # the last anchor has stopped moving for two consecutive checks.
    previous, stable = None, 0
    while stable < 2:
        page.wait_for_timeout(1000)
        current = box(page, segments[-1][1])["y"]
        stable = stable + 1 if current == previous else 0
        previous = current

    try:
        font = ImageFont.truetype("arialbd.ttf", 15 * SCALE)
    except OSError:
        font = ImageFont.load_default()
    r = 13 * SCALE
    pieces = []
    for i, (top_text, bottom_text, callouts) in enumerate(segments):
        top, bottom = box(page, top_text), box(page, bottom_text)
        y0, y1 = top["y"] - pad, bottom["y"] + bottom["h"] + pad
        raw = os.path.join(OUT, f"{name}_raw{i}.png")
        clip_w = width or page.viewport_size["width"]  # narrower crop keeps text legible at column width
        page.screenshot(path=raw, clip={"x": 0, "y": y0, "width": clip_w, "height": y1 - y0})
        img = Image.open(raw).convert("RGB")
        os.remove(raw)
        draw = ImageDraw.Draw(img)
        for number, anchor in callouts:
            b = box(page, anchor)
            cx = 40 * SCALE  # left gutter (content starts at x~80 css px)
            cy = (b["y"] - y0 + b["h"] / 2) * SCALE
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=ACCENT)
            draw.text((cx, cy), str(number), fill="white", font=font, anchor="mm")
        pieces.append(img)

    width = max(p.width for p in pieces)
    out_img = Image.new("RGB", (width, sum(p.height for p in pieces) + gap * SCALE * (len(pieces) - 1)), "white")
    y = 0
    for piece in pieces:
        out_img.paste(piece, (0, y))
        y += piece.height + gap * SCALE
    out = os.path.join(OUT, f"{name}.png")
    out_img.save(out, optimize=True)
    print(f"wrote {out} {out_img.size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8501")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1000, "height": 2600}, device_scale_factor=SCALE)
        capture(page, f"{args.url}/?demo=real_release&static=1", ready_text="Overall: NO-GO",
                segments=[("Core value metrics", "Overall: NO-GO",
                           [(1, "Core value metrics"), (2, "Guardrail / cost metrics"), (3, "Overall: NO-GO")])],
                name="fig_release", content_texts=("ARC-Challenge", "GSM8K answer length (words)"))
        capture(page, f"{args.url}/?demo=validation&static=1",
                ready_text="known-bad inputs correctly blocked",
                segments=[("[FWER STRESS TEST]", "empirical rates vary", [(4, "[FWER STRESS TEST]")]),
                          ("[DATA INTEGRITY GUARDRAIL]", "known-bad inputs correctly blocked",
                           [(5, "[DATA INTEGRITY GUARDRAIL]")])],
                name="fig_validation", pad=20, width=730)
        browser.close()
