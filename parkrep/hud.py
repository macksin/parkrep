"""Evangelion / Arena style HUD over a 720x1280 portrait canvas.

Everything shown at frame i is computable from frames <= i: a rep's numbers
appear only once its top has been reached, like a live readout would.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .reps import Analysis, Rep

W, H = 720, 1280
WHITE = (240, 244, 247, 255)
MUTED = (164, 178, 197, 255)
CYAN = (0, 207, 234, 255)
LIME = (165, 235, 25, 255)
ORANGE = (255, 132, 18, 255)
PURPLE = (179, 96, 243, 255)
GRAY = (105, 118, 132, 255)
GRID = (120, 143, 165, 40)

BODY = "/System/Library/Fonts/Avenir Next Condensed.ttc"
DISPLAY = "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf"


@lru_cache(maxsize=None)
def font(size: int, style: str = "body"):
    try:
        if style == "display":
            return ImageFont.truetype(DISPLAY, size)
        return ImageFont.truetype(BODY, size, index=7)  # Avenir Next Condensed Medium
    except OSError:
        return ImageFont.load_default(size)


def text(d, xy, value, size=14, color=WHITE, style="body"):
    d.text(xy, str(value), font=font(size, style), fill=color, anchor="lt")


def fit(d, xy, value, size, maxw, color=WHITE, style="body"):
    while size > 9 and d.textlength(value, font=font(size, style)) > maxw:
        size -= 1
    text(d, xy, value, size, color, style)


def panel(d, box, accent=PURPLE):
    x, y, r, b = box
    k = 8
    pts = [(x + k, y), (r - k, y), (r, y + k), (r, b - k), (r - k, b), (x + k, b), (x, b - k), (x, y + k)]
    d.polygon(pts, fill=(15, 24, 33, 158))
    d.line(pts + [pts[0]], fill=(141, 156, 195, 150), width=1)
    d.line((x + 14, y + 12, x + 14, y + 28), fill=accent, width=2)


def fmt(x, suffix="", digits=1, signed=False):
    if x is None or not np.isfinite(x):
        return "--"
    return f"{x:+.{digits}f}{suffix}" if signed else f"{x:.{digits}f}{suffix}"


class Frame:
    """Maps upright source pixels onto the portrait canvas (cover or letterbox)."""

    def __init__(self, width: int, height: int):
        self.src = (width, height)
        portrait = height >= width
        self.scale = max(W / width, H / height) if portrait else W / width
        self.ox = (W - width * self.scale) / 2
        self.oy = (H - height * self.scale) / 2 if portrait else 250 + (H - 250 - 520 - height * self.scale) / 2
        self.portrait = portrait

    def place(self, rgb: np.ndarray) -> Image.Image:
        img = Image.fromarray(rgb)
        size = (round(self.src[0] * self.scale), round(self.src[1] * self.scale))
        fg = img.resize(size, Image.BILINEAR)
        if self.portrait:
            canvas = Image.new("RGB", (W, H))
        else:  # landscape: blurred fill behind a full-width strip
            bg = img.resize((round(img.width * H / img.height), H), Image.BILINEAR)
            canvas = bg.crop(((bg.width - W) // 2, 0, (bg.width - W) // 2 + W, H)).filter(ImageFilter.GaussianBlur(24))
            canvas = Image.blend(canvas, Image.new("RGB", (W, H)), 0.45)
        canvas.paste(fg, (round(self.ox), round(self.oy)))
        return canvas

    def map(self, x, y):
        return self.ox + x * self.scale, self.oy + y * self.scale


class HUD:
    def __init__(self, a: Analysis, width: int, height: int, title: str, subtitle: str):
        self.a = a
        self.frame = Frame(width, height)
        self.title = title.upper()
        self.subtitle = subtitle
        self.t = a.t
        self.reps = a.reps
        self.refs = [r for r in a.reps if r.reference]
        self.reference = self._reference_curve()
        self.gradient = Image.new("RGBA", (W, H))
        d = ImageDraw.Draw(self.gradient)
        for y in range(200):
            d.line((0, y, W, y), fill=(5, 10, 18, int(150 * (1 - y / 200))))
        for y in range(1215, H):
            d.line((0, y, W, y), fill=(5, 10, 18, 120))
        finite = a.s[np.isfinite(a.s)]
        self.s_lo, self.s_hi = (float(finite.min()), float(finite.max())) if len(finite) else (0.0, 1.0)

    # ---- data at frame i -------------------------------------------------
    def _reference_curve(self):
        if len(self.refs) < 2:
            return None
        limit = min(r.concentric_s for r in self.refs)
        grid = np.linspace(0, limit, 60)
        curves = []
        for r in self.refs:
            tt = self.t[r.start:r.peak + 1] - self.t[r.start]
            vv = self.a.v[r.start:r.peak + 1]
            ok = np.isfinite(vv)
            if ok.sum() < 3:
                return None
            curves.append(np.interp(grid, tt[ok], vv[ok]))
        c = np.array(curves)
        return grid, np.median(c, axis=0), c.min(axis=0), c.max(axis=0)

    def done(self, i) -> list[Rep]:
        return [r for r in self.reps if r.peak <= i]

    def ref_ready(self, i) -> bool:
        return bool(self.refs) and all(r.peak <= i for r in self.refs)

    def loss(self, r: Rep, i: int):
        """Velocity loss is only defined once the whole reference has been seen."""
        return r.velocity_loss_pct if self.ref_ready(i) else None

    def rom(self, r: Rep, i: int):
        return r.rom_pct if self.ref_ready(i) else None

    def current(self, i) -> Rep | None:
        for r in self.reps:
            if r.start <= i <= r.end:
                return r
        return None

    # ---- widgets -----------------------------------------------------------
    def axes(self, d, box, xmax, ymin, ymax, xticks=True):
        x0, y0, x1, y1 = box
        for k in range(5):
            x = x0 + (x1 - x0) * k / 4
            d.line((x, y0, x, y1), fill=GRID)
            if xticks and k % 2 == 0:
                text(d, (x - 8, y1 + 5), f"{xmax * k / 4:.1f}", 13, MUTED)
        for k in range(3):
            y = y1 - (y1 - y0) * k / 2
            d.line((x0, y, x1, y), fill=GRID)
            text(d, (x0 - 34, y - 7), f"{ymin + (ymax - ymin) * k / 2:.0f}", 13, MUTED)
        d.line((x0, y0, x0, y1, x1, y1), fill=(166, 105, 218, 190))
        return lambda t, v: (x0 + (x1 - x0) * np.clip(t / xmax, 0, 1),
                             y1 - (y1 - y0) * np.clip((v - ymin) / (ymax - ymin), 0, 1))

    @staticmethod
    def polyline(d, tt, vv, m, color, width=2):
        tt, vv = np.asarray(tt, float), np.asarray(vv, float)
        ok = np.isfinite(vv)
        for a, b in zip(*[np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))[k::2] for k in (0, 1)]):
            if b - a > 1:
                xx, yy = m(tt[a:b], vv[a:b])
                d.line(list(zip(xx, yy)), fill=color, width=width)

    def header(self, d, i):
        d.line((14, 31, 14, 21, 24, 21), fill=PURPLE, width=2)
        d.line((695, 21, 706, 21, 706, 31), fill=PURPLE, width=2)
        text(d, (28, 28), "PARKREP", 38, WHITE, "display")
        x = 28 + d.textlength("PARKREP", font=font(38, "display")) + 12
        d.line((x, 40, x, 57), fill=LIME)
        fit(d, (x + 18, 35), self.title, 22, 540 - x - 18, MUTED, "display")
        d.line((28, 69, 538, 69), fill=(111, 165, 189, 160))
        fit(d, (28, 78), self.subtitle, 13, 500, MUTED)
        done = self.done(i)
        text(d, (596, 27), f"{len(done):02d}", 38, ORANGE, "display")
        text(d, (648, 36), "REPS", 22, WHITE, "display")
        cur = self.current(i)
        shown = done + ([cur] if cur and cur not in done and cur.start <= i else [])
        spacing = min(18, 136 / max(1, len(shown) - 1)) if shown else 18
        radius = min(5, spacing / 3)
        for j, r in enumerate(shown):
            x = 556 + j * spacing
            col = CYAN if r.peak <= i else ORANGE
            d.ellipse((x - radius, 69 - radius, x + radius, 69 + radius),
                      fill=col if r.peak <= i else None, outline=col)

    def stats(self, d, i):
        d.rectangle((20, 94, 704, 196), fill=(5, 12, 21, 113))
        labels = [(28, "PHASE"), (211, "LAST CONCENTRIC"), (380, "VELOCITY LOSS"), (560, "ROM")]
        for x, label in labels:
            text(d, (x, 104), label, 16, MUTED)
        for x in (191, 362, 545):
            d.line((x, 103, x, 166), fill=(99, 157, 181, 155))
        d.line((27, 103, 27, 164), fill=PURPLE, width=2)
        phase = self.a.phase[i]
        col = {"CONCENTRIC": LIME, "ECCENTRIC": CYAN, "TRACKING GAP": ORANGE}.get(phase, WHITE)
        fit(d, (39, 126), phase, 48, 145, col, "display")
        done = self.done(i)
        last = done[-1] if done else None
        text(d, (211, 125), fmt(last.concentric_s if last else None, " s", 2), 48, WHITE, "display")
        loss = self.loss(last, i) if last else None
        loss_col = WHITE if loss is None else LIME if loss < 10 else ORANGE if loss < 30 else (255, 70, 70, 255)
        text(d, (380, 125), fmt(loss, "%", 1, signed=True), 45, loss_col, "display")
        text(d, (560, 125), fmt(self.rom(last, i) if last else None, "%", 0), 45, WHITE, "display")
        ref_ready = self.ref_ready(i)
        sub = f"REP {last.number} @ {last.peak_s:.1f}s" if last else "AWAITING FIRST REP"
        text(d, (211, 176), sub, 13, MUTED)
        if not self.refs:
            cap = "NEEDS 2+ REPS"
        elif ref_ready:
            cap = "VS REPS " + "/".join(str(r.number) for r in self.refs)
        else:
            cap = "BUILDING REFERENCE"
        text(d, (380, 176), cap, 13, CYAN)
        text(d, (560, 176), "OF REFERENCE ROM" if ref_ready else cap, 13, CYAN)

    def ruler(self, d, i):
        panel(d, (645, 214, 698, 560))
        text(d, (657, 227), "px", 14, MUTED)
        lo, hi = self.s_lo - 10, self.s_hi + 10
        m = lambda v: 249 + 288 * (hi - v) / (hi - lo)
        d.line((659, 249, 659, 537), fill=(113, 157, 180, 170))
        for v in np.linspace(lo, hi, 5):
            y = m(v)
            d.line((655, y, 665, y), fill=LIME)
            text(d, (669, y - 6), f"{v - self.s_lo:.0f}", 12, MUTED)
        if np.isfinite(self.a.s[i]):
            y = m(self.a.s[i])
            d.ellipse((653, y - 6, 665, y + 6), outline=ORANGE, width=2)
        else:
            text(d, (652, 390), "--", 14, ORANGE, "display")

    def marker(self, d, i):
        xy = self.a.xy
        a = max(0, i - int(self.a.fps * 0.8))
        trail = [self.frame.map(*p) for p in xy[a:i + 1] if np.isfinite(p).all()]
        for k in range(1, len(trail)):
            alpha = int(200 * k / len(trail))
            d.line((trail[k - 1], trail[k]), fill=(165, 235, 25, alpha), width=3)
        if np.isfinite(xy[i]).all():
            x, y = self.frame.map(*xy[i])
            col = ORANGE if self.a.inferred[i] else CYAN
            d.ellipse((x - 10, y - 10, x + 10, y + 10), outline=col, width=2)
            d.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(233, 250, 251, 220))
            d.line((x + 12, y, x + 30, y), fill=col)
            text(d, (x + 34, y - 8), f"{self.a.v[i]:+.0f} px/s" if np.isfinite(self.a.v[i]) else "", 14, col)

    def curve_panel(self, d, i):
        panel(d, (14, 780, 438, 1025))
        text(d, (34, 793), "CONCENTRIC VS REFERENCE", 25, WHITE, "display")
        d.line((27, 822, 420, 822), fill=(121, 157, 184, 145))
        cur = self.current(i)
        done = self.done(i)
        rep = cur if cur and cur.start <= i else (done[-1] if done else None)
        if rep is None:
            text(d, (196, 897), "--", 32, MUTED, "display")
            return
        a = rep.start
        b = min(i, rep.peak) + 1
        tt = self.t[a:b] - self.t[a]
        vv = self.a.v[a:b]
        ref = self.reference if self.ref_ready(i) else None
        xmax = max(1.0, rep.concentric_s if rep.peak <= i else float(tt[-1]) if len(tt) else 1.0,
                   float(ref[0][-1]) if ref else 0.0)
        xmax = math.ceil(xmax * 2) / 2
        vals = list(vv[np.isfinite(vv)]) + (list(ref[3]) if ref else []) + [rep.peak_speed if rep.peak <= i else 0]
        ymax = max(50.0, math.ceil(max(vals, default=50) / 50) * 50)
        m = self.axes(d, (97, 847, 421, 989), xmax, 0, ymax)
        text(d, (26, 895), "Velocity", 16, MUTED)
        text(d, (26, 916), "(px/s)", 16, MUTED)
        text(d, (203, 1007), "Time (s)", 14, MUTED)
        if ref:
            grid, med, lo, hi = ref
            poly = [m(t, v) for t, v in zip(grid, hi)] + [m(t, v) for t, v in zip(grid[::-1], lo[::-1])]
            d.polygon([(float(x), float(y)) for x, y in poly], fill=(181, 191, 201, 51))
            self.polyline(d, grid, med, m, (180, 189, 200, 200))
        self.polyline(d, tt, vv, m, LIME)
        if len(tt) and np.isfinite(vv[-1]):
            x, y = m(tt[-1], vv[-1])
            d.line((x, 847, x, 989), fill=(255, 132, 18, 150))
            d.ellipse((x - 3, y - 3, x + 3, y + 3), fill=LIME)
        text(d, (300, 799), f"REP {rep.number:02d}", 17, LIME, "display")
        text(d, (366, 801), "REF " + ("/".join(str(r.number) for r in self.refs) if ref else "--"), 13, MUTED)

    def mini_panels(self, d, i):
        cur = self.current(i)
        done = self.done(i)
        rep = cur if cur and cur.start <= i else (done[-1] if done else None)
        for j, label in enumerate(("X POSITION", "Y POSITION")):
            top, bottom = (780, 901) if j == 0 else (909, 1025)
            panel(d, (448, top, 706, bottom))
            text(d, (477, top + 11), label, 24, WHITE, "display")
            text(d, (622, top + 15), "REL px" + (" / +UP" if j else ""), 11, MUTED)
            if rep is None:
                text(d, (560, top + 52), "--", 24, MUTED, "display")
                continue
            a, b = rep.start, min(i, rep.end) + 1
            tt = self.t[a:b] - self.t[a]
            vals = (self.a.xy[a:b, j] - self.a.xy[a, j]) * (1 if j == 0 else -1)
            fin = vals[np.isfinite(vals)]
            if not len(fin):
                text(d, (560, top + 52), "--", 24, MUTED, "display")
                continue
            lo, hi = min(float(fin.min()), 0), max(float(fin.max()), 0)
            pad = max(5.0, 0.12 * (hi - lo))
            xmax = max(1.0, math.ceil((self.t[rep.end] - self.t[a]) * 2) / 2)
            m = self.axes(d, (493, top + 39, 690, bottom - 28), xmax, lo - pad, hi + pad)
            self.polyline(d, tt, vals, m, LIME)
            if np.isfinite(vals[-1]):
                x, y = m(tt[-1], vals[-1])
                d.line((x, top + 39, x, bottom - 28), fill=(255, 132, 18, 180))
                d.ellipse((x - 3, y - 3, x + 3, y + 3), outline=ORANGE, width=2)

    def histogram(self, d, i):
        panel(d, (14, 1036, 706, 1212))
        text(d, (34, 1048), "VELOCITY LOSS BY REP", 25, WHITE, "display")
        d.line((28, 1077, 691, 1077), fill=(121, 157, 184, 145))
        reps = self.reps
        done = self.done(i)
        vals = [self.loss(r, i) for r in done if self.loss(r, i) is not None]
        lo = min(-20.0, math.floor(min(vals, default=0) / 20) * 20)
        hi = max(40.0, math.ceil(max(vals, default=0) / 20) * 20)
        y0, y1 = 1093, 1170
        ymap = lambda v: y1 - (y1 - y0) * (v - lo) / (hi - lo)
        zero = ymap(0)
        for v in (lo, 0, hi):
            d.line((64, ymap(v), 690, ymap(v)), fill=(130, 145, 171, 55))
            text(d, (24, ymap(v) - 8), f"{v:.0f}%", 14, MUTED)
        d.line((64, zero, 690, zero), fill=(164, 107, 215, 150))
        slots = max(len(reps), 8)
        width = 626 / slots
        for j, r in enumerate(reps):
            x = 64 + (j + 0.5) * width
            bw = min(43, width * 0.66)
            active = r.start <= i < r.peak
            if r.peak <= i and self.loss(r, i) is not None:
                col = PURPLE if r.reference else CYAN
                y = ymap(r.velocity_loss_pct)
                d.rectangle((x - bw / 2, min(y, zero), x + bw / 2, max(y, zero) + 1), fill=col)
                text(d, (x - 13, max(1079, min(y, zero) - 21)), f"{r.velocity_loss_pct:.0f}%", 18, col, "display")
            elif active:
                d.rectangle((x - bw / 2, zero - 8, x + bw / 2, zero), outline=ORANGE)
                text(d, (x - 7, zero - 21), "...", 16, ORANGE, "display")
            if r.start <= i:
                text(d, (x - 9, 1179), f"{r.number:02d}", 22, ORANGE if active else CYAN, "display")

    def comparison(self, d, i):
        panel(d, (14, 684, 706, 772))
        done = self.done(i)
        losses = [r.velocity_loss_pct for r in done if r.velocity_loss_pct is not None]
        if len(losses) >= 2 and self.ref_ready(i):
            last2 = float(np.median(losses[-2:]))
            nums = "/".join(str(r.number) for r in done[-2:])
            fit(d, (32, 694), f"LAST 2 REPS {nums}  |  VELOCITY LOSS {last2:+.1f}%  |  VS REPS "
                + "/".join(str(r.number) for r in self.refs), 23, 660, CYAN, "display")
            verdict = ("FRESH - SPEED HOLDING" if last2 < 10 else "FATIGUE BUILDING" if last2 < 25
                       else "GRINDING - NEAR THE END OF THE SET" if last2 < 40 else "HEAVY SLOWDOWN")
            fit(d, (32, 723), verdict, 22, 660, ORANGE if last2 >= 25 else LIME, "display")
        else:
            fit(d, (32, 694), "SPEED COMPARISON / BUILDING REFERENCE FROM FIRST REPS", 23, 660, MUTED, "display")
            fit(d, (32, 723), "EACH REP UPDATES WHEN ITS TOP IS REACHED", 21, 660, MUTED, "display")
        fit(d, (32, 750), "IMAGE-PLANE MOTION / RELATIVE TO THIS SET ONLY", 15, 660, GRAY)

    def footer(self, d, i):
        ref = "/".join(str(r.number) for r in self.refs) or "--"
        fit(d, (27, 1225), f"Reference: reps {ref} (median)  |  Point: auto-selected, LocoTrack-S", 16, 510, MUTED)
        text(d, (560, 1225), f"SOURCE {self.t[i]:.2f} s", 17, MUTED)
        fit(d, (27, 1246), "Local inference on-device  |  No upload  |  Velocity in px/s, not calibrated", 13, 669, MUTED)
        d.line((14, 1258, 14, 1270, 26, 1270), fill=PURPLE, width=2)
        d.line((693, 1270, 706, 1270, 706, 1258), fill=PURPLE, width=2)

    def draw(self, rgb: np.ndarray, i: int) -> Image.Image:
        base = self.frame.place(rgb).convert("RGBA")
        layer = self.gradient.copy()
        d = ImageDraw.Draw(layer)
        self.marker(d, i)
        self.header(d, i)
        self.stats(d, i)
        self.ruler(d, i)
        self.comparison(d, i)
        self.curve_panel(d, i)
        self.mini_panels(d, i)
        self.histogram(d, i)
        self.footer(d, i)
        return Image.alpha_composite(base, layer).convert("RGB")

    # ---- end card ----------------------------------------------------------
    def summary_card(self, background: Image.Image) -> Image.Image:
        s = self.a.summary()
        layer = Image.new("RGBA", (W, H))
        d = ImageDraw.Draw(layer)
        panel(d, (24, 229, 696, 768))
        d.polygon([(32, 237), (688, 237), (688, 760), (32, 760)], fill=(8, 15, 25, 240))
        d.line((48, 255, 48, 294), fill=PURPLE, width=3)
        text(d, (66, 254), "SET COMPLETE", 42, WHITE, "display")
        fit(d, (66, 301), f"{self.title}  /  {s['reps']} REPS  /  {self.t[-1]:.0f} S CLIP", 19, 600, MUTED)
        d.line((48, 333, 672, 333), fill=(138, 154, 184, 140))
        text(d, (49, 350), "VELOCITY LOSS, LAST 2 REPS", 25, WHITE, "display")
        last = s["last_reps_velocity_loss_pct"]
        text(d, (49, 388), fmt(last, "%", 1, signed=True), 78, LIME if last is not None and last < 25 else ORANGE, "display")
        text(d, (400, 355), "REPS", 25, MUTED, "display")
        text(d, (400, 388), f"{s['reps']:02d}", 78, CYAN, "display")
        reps = self.reps
        roms = [r.rom_pct for r in reps[-2:] if r.rom_pct is not None]
        conc = [r.concentric_s for r in reps]
        fastest = min(reps, key=lambda r: r.concentric_s) if reps else None
        rows = [
            ("REFERENCE", "REPS " + ("/".join(str(r.number) for r in self.refs) or "--"), PURPLE),
            ("MAX VELOCITY LOSS", fmt(s["max_velocity_loss_pct"], "%", 1, signed=True), ORANGE),
            ("ROM RETAINED, LAST 2", fmt(float(np.median(roms)) if roms else None, "%", 0), CYAN),
            ("AVG CONCENTRIC", fmt(float(np.mean(conc)) if conc else None, " s", 2), WHITE),
            ("FASTEST REP", f"REP {fastest.number} / {fastest.concentric_s:.2f} s" if fastest else "--", LIME),
            ("POINT TRACKED", f"{s['tracked_share'] * 100:.0f}% OF FRAMES", WHITE),
        ]
        for k, (label, value, col) in enumerate(rows):
            y = 486 + k * 36
            text(d, (49, y), label, 22, MUTED, "display")
            text(d, (400, y - 2), value, 26, col, "display")
        d.line((48, 708, 672, 708), fill=(138, 154, 184, 140))
        fit(d, (49, 720), "+ slower / - faster than the first reps  |  image motion, not a diagnosis", 17, 621, MUTED)
        return Image.alpha_composite(background.convert("RGBA"), layer).convert("RGB")
