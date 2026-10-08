"""NewsGuard AI - Fake News Detection & Source Credibility Analyzer.

Streamlit front end for the TF-IDF + Logistic Regression content classifier,
extended with:
  * word-level explanation (which words pushed the score, and which way)
  * a rule-based source credibility check from an optional article URL
  * a fused credibility score (70% content, 30% source)
  * charts: score gauge, probability donut, score build-up bar, diverging word
    chart, source-score waterfall, writing-signal tiles, session trend line and
    a confusion-matrix heatmap (all drawn as inline SVG/HTML, no extra packages)
"""

import html
import math
import re
from urllib.parse import urlparse

import joblib
import numpy as np
import streamlit as st

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NewsGuard AI",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ============================================================
# MODEL
# ============================================================


@st.cache_resource
def load_artifacts():
    return joblib.load("model.pkl"), joblib.load("tfidf.pkl")


model, tfidf = load_artifacts()

# Same default token pattern the TfidfVectorizer was trained with.
TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")

# Held-out test results (20% split, 8,938 articles) from training.
CONFUSION = {"fake_fake": 4645, "fake_real": 51, "real_fake": 43, "real_real": 4199}

# Fusion weights: content classifier vs. source check.
W_CONTENT, W_SOURCE = 0.7, 0.3

RED, GOLD, TEAL, INK = "#B42318", "#B8901A", "#17745A", "#101B33"
TONE_COLOUR = {"fake": RED, "mixed": GOLD, "real": TEAL}

# ============================================================
# SAMPLE ARTICLES (invented for demonstration)
# ============================================================

SAMPLES = {
    "wire": {
        "label": "Wire-style report",
        "headline": "Senate passes funding bill to avert government shutdown",
        "article": (
            "WASHINGTON (Reuters) - The U.S. Senate on Tuesday passed a bill to "
            "keep the government funded through December, sending the measure to "
            "the House of Representatives, officials said. The vote was 72-26. "
            "Senate leaders told reporters they expected the House to act before "
            "the Friday deadline, according to a statement from the majority "
            "leader's office."
        ),
        "url": "https://www.reuters.com/world/us/senate-passes-funding-bill",
    },
    "clickbait": {
        "label": "Clickbait post",
        "headline": "BREAKING VIDEO: You won't believe what they just did",
        "article": (
            "Watch this before it gets deleted! This is the video the mainstream "
            "media doesn't want you to see. Share it with everyone you know. "
            "Featured image via Twitter. READ MORE and tell us what you think "
            "in the comments below!!!"
        ),
        "url": "http://patriot-truth-exposed24.xyz/breaking-video",
    },
    "plain": {
        "label": "Plain science item",
        "headline": "Scientists announce a major breakthrough in battery technology",
        "article": (
            "Researchers at a university lab have developed a new battery design "
            "that could double the driving range of electric vehicles. The team "
            "published its findings in a peer-reviewed journal and plans larger "
            "trials next year."
        ),
        "url": "",
    },
}

# ============================================================
# SOURCE CREDIBILITY (rule-based)
# ============================================================

REPUTABLE_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "npr.org", "pbs.org",
    "thehindu.com", "nature.com", "science.org", "who.int", "un.org",
    "pib.gov.in", "economist.com", "theguardian.com",
}
OFFICIAL_SUFFIXES = (".gov", ".edu", ".gov.in", ".nic.in", ".ac.in", ".edu.in")
RISKY_TLDS = (".xyz", ".top", ".click", ".buzz", ".icu", ".cf", ".tk", ".gq")
RISKY_WORDS = ("viral", "exposed", "truth", "patriot", "leaks", "shocking")


def score_source(url: str):
    """Return (score 0-100, factors, host) for a URL, or None if no usable URL."""
    url = (url or "").strip()
    if not url:
        return None
    parsed = urlparse(url if "//" in url else "https://" + url)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    if not host or "." not in host:
        return None

    score, factors = 50, [("Starting point", 0, "Neutral baseline for an unknown site")]

    def add(label, delta, note):
        nonlocal score
        score += delta
        factors.append((label, delta, note))

    if host in REPUTABLE_DOMAINS or any(host.endswith("." + d) for d in REPUTABLE_DOMAINS):
        add("Established outlet", +30, "Domain is on the reviewed list of established outlets")
    if host.endswith(OFFICIAL_SUFFIXES):
        add("Official or academic domain", +10, "Government, university or institute domain")
    if parsed.scheme == "https" or not parsed.scheme:
        add("Secure connection", +5, "Uses HTTPS")
    elif parsed.scheme == "http":
        add("No secure connection", -15, "Uses plain HTTP")
    if host.endswith(RISKY_TLDS):
        add("Low-cost domain ending", -20, "Often used by short-lived sites")
    name = host.rsplit(".", 1)[0]
    if sum(ch.isdigit() for ch in name) >= 2:
        add("Numbers in the name", -10, "Typical of auto-generated sites")
    if name.count("-") >= 2:
        add("Many hyphens", -10, "Typical of look-alike or throwaway sites")
    if len(name) > 25:
        add("Very long name", -10, "Long names are rare for established outlets")
    if any(w in name for w in RISKY_WORDS):
        add("Sensational wording in name", -10, "Name uses emotive or conspiratorial words")

    return max(0, min(100, score)), factors, host


# ============================================================
# EXPLANATION
# ============================================================


def explain(content: str):
    """Per-word contribution to the 'real' score (positive = real, negative = fake)."""
    vec = tfidf.transform([content]).tocsr()
    names = tfidf.get_feature_names_out()
    coef = model.coef_[0]
    contrib = {}
    for idx, val in zip(vec.indices, vec.data):
        contrib[names[idx]] = float(val * coef[idx])
    return vec, contrib


def annotate(text: str, contrib: dict, top_n: int = 14) -> str:
    """Return the article as HTML with the strongest words highlighted."""
    ranked = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    marks = dict(ranked)
    peak = max((abs(v) for v in marks.values()), default=1) or 1
    out, last = [], 0
    for m in TOKEN_RE.finditer(text):
        word = m.group(0)
        out.append(html.escape(text[last:m.start()]))
        val = marks.get(word.lower())
        if val is None:
            out.append(html.escape(word))
        else:
            cls = "real" if val > 0 else "fake"
            alpha = 0.14 + 0.34 * abs(val) / peak
            tip = f"{'Pushes toward real' if val > 0 else 'Pushes toward fake'} ({val:+.2f})"
            out.append(
                f'<mark class="{cls}" style="--a:{alpha:.2f}" title="{tip}">'
                f"{html.escape(word)}</mark>"
            )
        last = m.end()
    out.append(html.escape(text[last:]))
    return "".join(out).replace("\n", "<br>")


def h(s: str) -> str:
    """Flatten HTML so Markdown never treats indented lines as code blocks."""
    return " ".join(line.strip() for line in s.splitlines() if line.strip())


def band(score: float) -> str:
    return "real" if score >= 70 else "mixed" if score >= 40 else "fake"


# ============================================================
# CHARTS (inline SVG / HTML)
# ============================================================


def gauge_svg(score: float, tone: str) -> str:
    """Semicircle gauge with three score zones and an animated needle."""
    cx, cy, r = 200, 190, 140

    def pt(s, rad=r):
        th = math.pi * (1 - s / 100)
        return cx + rad * math.cos(th), cy - rad * math.sin(th)

    def arc(s1, s2, colour):
        x1, y1 = pt(s1)
        x2, y2 = pt(s2)
        return (f'<path d="M{x1:.1f} {y1:.1f} A{r} {r} 0 0 1 {x2:.1f} {y2:.1f}" '
                f'fill="none" stroke="{colour}" stroke-width="26"/>')

    ticks = ""
    for s in (0, 40, 70, 100):
        x, y = pt(s, r + 30)
        ticks += (f'<text x="{x:.1f}" y="{y + 5:.1f}" text-anchor="middle" '
                  f'font-size="15" fill="#5A647D">{s}</text>')
    rot = score * 1.8
    colour = {"fake": RED, "mixed": "#7A5C0A", "real": TEAL}[tone]
    return (
        f'<svg viewBox="0 0 400 290" role="img" aria-label="Credibility gauge showing {score:.0f} out of 100">'
        f'{arc(0, 40, "#E7B7B2")}{arc(40, 70, "#EAD9A4")}{arc(70, 100, "#A9D3C3")}{ticks}'
        f'<g class="needle" style="--rot:{rot:.1f}deg">'
        f'<line x1="{cx}" y1="{cy}" x2="{cx - (r - 34)}" y2="{cy}" stroke="{INK}" '
        f'stroke-width="5" stroke-linecap="round"/><circle cx="{cx}" cy="{cy}" r="11" fill="{INK}"/></g>'
        f'<text x="{cx}" y="{cy + 62}" text-anchor="middle" font-size="58" font-weight="700" '
        f'class="serif" fill="{colour}">{score:.0f}</text>'
        f'<text x="{cx}" y="{cy + 86}" text-anchor="middle" font-size="16" fill="#5A647D">credibility out of 100</text>'
        f"</svg>"
    )


def donut_svg(real_p: float) -> str:
    r = 54
    circ = 2 * math.pi * r
    real_len = circ * real_p / 100
    return (
        f'<svg viewBox="0 0 140 140" role="img" aria-label="Writing probability: {real_p:.0f} percent real">'
        f'<circle cx="70" cy="70" r="{r}" fill="none" stroke="{RED}" stroke-width="18"/>'
        f'<circle class="arc" cx="70" cy="70" r="{r}" fill="none" stroke="{TEAL}" stroke-width="18" '
        f'stroke-dasharray="{real_len:.2f} {circ:.2f}" transform="rotate(-90 70 70)" '
        f'style="--c:{circ:.2f}"/>'
        f'<text x="70" y="76" text-anchor="middle" font-size="30" font-weight="700" class="serif" '
        f'fill="{INK}">{real_p:.0f}%</text>'
        f'<text x="70" y="96" text-anchor="middle" font-size="12" fill="#5A647D">real</text></svg>'
    )


def stack_bar(real_p: float, src_score) -> str:
    """Show how many of the 100 points come from writing and from the source."""
    if src_score is None:
        w_pts, s_pts = real_p, 0.0
    else:
        w_pts, s_pts = W_CONTENT * real_p, W_SOURCE * src_score
    total = w_pts + s_pts
    segs = f'<i class="sw" style="width:{w_pts:.2f}%"></i>'
    if s_pts:
        segs += f'<i class="ss" style="width:{s_pts:.2f}%"></i>'
    legend = f'<span><i class="k sw"></i>Writing {w_pts:.1f} points</span>'
    if src_score is not None:
        legend += f'<span><i class="k ss"></i>Source {s_pts:.1f} points</span>'
    note = (
        f"Writing is worth up to {int(W_CONTENT * 100)} points and the source up to {int(W_SOURCE * 100)}."
        if src_score is not None
        else "No link was added, so the writing score is the whole score."
    )
    return (
        f'<div class="stack" role="img" aria-label="Score build-up: {total:.0f} of 100 points">{segs}</div>'
        f'<div class="stack-axis"><span>0</span><span>40</span><span>70</span><span>100</span></div>'
        f'<div class="legend2">{legend}<span class="tot">Total ≈ {total:.0f}</span></div>'
        f'<p class="mini">{note}</p>'
    )


def diverging_words(contrib: dict, per_side: int = 7) -> str:
    neg = sorted([kv for kv in contrib.items() if kv[1] < 0], key=lambda kv: kv[1])[:per_side]
    pos = sorted([kv for kv in contrib.items() if kv[1] > 0], key=lambda kv: -kv[1])[:per_side]
    rows = neg + sorted(pos, key=lambda kv: kv[1])
    if not rows:
        return '<div class="empty">No scoring words found in this text.</div>'
    peak = max(abs(v) for _, v in rows) or 1
    out = ['<div class="dvhead"><span>← pushes toward fake</span><span>pushes toward real →</span></div>']
    for w, v in rows:
        width = abs(v) / peak * 50
        side = f"right:50%;width:{width:.1f}%" if v < 0 else f"left:50%;width:{width:.1f}%"
        cls = "neg" if v < 0 else "pos"
        out.append(
            f'<div class="dv"><span class="w">{html.escape(w)}</span>'
            f'<div class="trk"><i class="{cls}" style="{side}"></i></div>'
            f'<span class="v {cls}">{v:+.2f}</span></div>'
        )
    return "".join(out)


def source_waterfall(factors, final_score) -> str:
    clamp = lambda x: max(0.0, min(100.0, x))
    rows = [
        '<div class="wf"><div class="lab"><b>Start</b><small>Neutral baseline for an unknown site</small></div>'
        '<div class="trk"><i class="base" style="left:0;width:50%"></i></div><span class="pts">50</span></div>'
    ]
    running = 50
    for label, delta, note in factors[1:]:
        before, after = running, running + delta
        lo, hi = clamp(min(before, after)), clamp(max(before, after))
        cls = "pos" if delta > 0 else "neg"
        rows.append(
            f'<div class="wf"><div class="lab"><b>{html.escape(label)}</b><small>{html.escape(note)}</small></div>'
            f'<div class="trk"><i class="{cls}" style="left:{lo:.1f}%;width:{max(hi - lo, 0.8):.1f}%"></i></div>'
            f'<span class="pts {cls}">{delta:+d}</span></div>'
        )
        running = after
    rows.append(
        '<div class="wf fin"><div class="lab"><b>Source score</b><small>Limited to 0 to 100</small></div>'
        f'<div class="trk"><i class="fin" style="left:0;width:{final_score:.1f}%"></i></div>'
        f'<span class="pts">{final_score}</span></div>'
    )
    axis = '<div class="wf axis"><div></div><div class="ax"><span>0</span><span>50</span><span>100</span></div><span></span></div>'
    return "".join(rows) + axis


def signal_tiles(content: str, words_n: int, sent_n: int) -> str:
    caps = [w for w in re.findall(r"\b[A-Za-z]{3,}\b", content) if w.isupper()]
    caps_pct = len(caps) / max(words_n, 1) * 100
    excl, ques = content.count("!"), content.count("?")
    avg_len = words_n / max(sent_n, 1)
    tiles = [
        ("ALL-CAPS words", f"{caps_pct:.1f}%", min(caps_pct / 20, 1)),
        ("Exclamation marks", f"{excl}", min(excl / 5, 1)),
        ("Question marks", f"{ques}", min(ques / 5, 1)),
        ("Words per sentence", f"{avg_len:.0f}", min(avg_len / 40, 1)),
    ]
    return "".join(
        f'<div class="tile"><b>{v}</b><span>{html.escape(k)}</span>'
        f'<div class="mb"><i style="width:{frac * 100:.0f}%"></i></div></div>'
        for k, v, frac in tiles
    )


def history_svg(items) -> str:
    """Line chart of credibility scores across checks in this session."""
    W, H, left, right, top, bottom = 760, 214, 56, 24, 14, 38
    plot_h = H - top - bottom
    y = lambda s: top + plot_h * (1 - s / 100)
    x0, x1 = left, W - right
    bands = (
        (0, 40, "#E7B7B2"), (40, 70, "#EAD9A4"), (70, 100, "#A9D3C3"),
    )
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Credibility scores of your checks this session">']
    for lo, hi, col in bands:
        parts.append(f'<rect x="{x0}" y="{y(hi):.1f}" width="{x1 - x0}" height="{y(lo) - y(hi):.1f}" fill="{col}" opacity="0.45"/>')
    for s in (0, 40, 70, 100):
        parts.append(f'<text x="{left - 10}" y="{y(s) + 5:.1f}" text-anchor="end" font-size="14" fill="#5A647D">{s}</text>')
    n = len(items)
    xs = [(x0 + x1) / 2] if n == 1 else [x0 + (x1 - x0) * i / (n - 1) for i in range(n)]
    pts = " ".join(f"{px:.1f},{y(s):.1f}" for px, (_, s, _) in zip(xs, items))
    if n > 1:
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{INK}" stroke-width="2.5" stroke-linejoin="round"/>')
    for px, (k, s, tone) in zip(xs, items):
        parts.append(f'<circle cx="{px:.1f}" cy="{y(s):.1f}" r="8" fill="{TONE_COLOUR[tone]}" stroke="#fff" stroke-width="2.5"/>')
        parts.append(f'<text x="{px:.1f}" y="{y(s) - 14:.1f}" text-anchor="middle" font-size="14" font-weight="600" fill="{INK}">{s:.0f}</text>')
        parts.append(f'<text x="{px:.1f}" y="{H - 12}" text-anchor="middle" font-size="14" fill="#5A647D">#{k}</text>')
    parts.append("</svg>")
    return "".join(parts)


def confusion_heatmap(cm) -> str:
    fake_n = cm["fake_fake"] + cm["fake_real"]
    real_n = cm["real_fake"] + cm["real_real"]
    cells = [
        (cm["fake_fake"], cm["fake_fake"] / fake_n, True),
        (cm["fake_real"], cm["fake_real"] / fake_n, False),
        (cm["real_fake"], cm["real_fake"] / real_n, False),
        (cm["real_real"], cm["real_real"] / real_n, True),
    ]
    body = ""
    for i, (n, frac, ok) in enumerate(cells):
        if i == 0:
            body += '<div class="rh">Actual fake</div>'
        if i == 2:
            body += '<div class="rh">Actual real</div>'
        colour = "23,116,90" if ok else "180,35,24"
        alpha = 0.10 + 0.55 * frac if ok else 0.18
        body += (f'<div class="c" style="background:rgba({colour},{alpha:.2f})"><b>{n:,}</b>'
                 f'<span>{frac * 100:.1f}% of row</span><em>{"correct" if ok else "wrong"}</em></div>')
    return ('<div class="cm"><div></div><div class="ch">Predicted fake</div><div class="ch">Predicted real</div>'
            f"{body}</div>")


def metric_bars(rows) -> str:
    out = []
    for label, v in rows:
        frac = max(0.0, min(1.0, (v - 95) / 5))
        out.append(
            f'<div class="mrow"><span>{html.escape(label)}</span>'
            f'<div class="mb"><i style="width:{frac * 100:.1f}%"></i></div><b>{v:.2f}%</b></div>'
        )
    return "".join(out)


# ============================================================
# STYLES
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;0,6..72,700;1,6..72,400&family=Instrument+Sans:wght@400;500;600;700&display=swap');

:root {
  --ink: #101B33; --ink-2: #4B5670; --ink-3: #5A647D;
  --paper: #F4F6FA; --white: #FFFFFF; --line: #D6DBE6;
  --gold: #9A7712; --red: #B42318; --teal: #17745A;
}
html, body, .stApp, [data-testid="stAppViewContainer"] {
  background: var(--paper) !important;
  color: var(--ink);
  font-family: 'Instrument Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { max-width: 920px; padding-top: 2.2rem; padding-bottom: 4rem; }
svg text { font-family: 'Instrument Sans', system-ui, sans-serif; }
svg text.serif { font-family: 'Newsreader', Georgia, serif; }

/* masthead */
.mast { display: flex; align-items: baseline; justify-content: space-between;
        border-bottom: 2px solid var(--ink); padding-bottom: 10px; margin-bottom: 34px; }
.mast .name { font-family: 'Newsreader', Georgia, serif; font-size: 26px; font-weight: 700; letter-spacing: -0.01em; }
.mast .name span { color: var(--gold); }
.mast .tag { color: var(--ink-3); font-size: 13px; }

h1.lead { font-family: 'Newsreader', Georgia, serif; font-weight: 600; font-size: 46px;
          line-height: 1.08; letter-spacing: -0.02em; margin: 0 0 14px; max-width: 22ch; text-wrap: balance; }
p.sub { color: var(--ink-2); font-size: 17px; line-height: 1.55; max-width: 58ch; margin: 0 0 22px; }
.chips { display: flex; flex-wrap: wrap; gap: 10px; margin: 0 0 28px; }
.chips span { background: var(--white); border: 1px solid var(--line); border-radius: 999px;
              padding: 7px 14px; font-size: 14px; color: var(--ink-2); }
.chips b { color: var(--ink); font-weight: 700; margin-right: 4px; }

/* inputs */
label, [data-testid="stWidgetLabel"] p { color: var(--ink) !important; font-weight: 600 !important; font-size: 14px !important; }
.stTextInput input, .stTextArea textarea {
  background: var(--white) !important; color: var(--ink) !important;
  border: 1px solid var(--line) !important; border-radius: 6px !important;
  font-family: 'Instrument Sans', system-ui, sans-serif !important; font-size: 16px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
  border-color: var(--ink) !important; box-shadow: 0 0 0 3px rgba(16,27,51,0.12) !important;
}

/* buttons */
.stButton > button {
  border-radius: 6px; font-weight: 600; font-size: 15px; padding: 0.55rem 1.1rem;
  border: 1px solid var(--line); background: var(--white); color: var(--ink);
  transition: border-color .15s ease, background .15s ease;
}
.stButton > button:hover { border-color: var(--ink); color: var(--ink); background: var(--white); }
.stButton > button:focus-visible { outline: 3px solid var(--gold); outline-offset: 2px; }
.stButton > button[kind="primary"] { background: var(--ink); color: #fff; border-color: var(--ink); }
.stButton > button[kind="primary"]:hover { background: #1d2c52; color: #fff; }

/* verdict */
.verdict { margin-top: 36px; border-top: 2px solid var(--ink); padding-top: 22px; }
.vgrid { display: grid; grid-template-columns: 1fr 380px; gap: 28px; align-items: center; }
.verdict .kicker { color: var(--ink-3); font-size: 14px; }
.verdict .call { font-family: 'Newsreader', Georgia, serif; font-size: 54px; font-weight: 700;
                 letter-spacing: -0.02em; line-height: 1.05; margin: 4px 0 8px; }
.call.fake { color: var(--red); } .call.real { color: var(--teal); } .call.mixed { color: #7A5C0A; }
.verdict p { color: var(--ink-2); font-size: 16px; line-height: 1.55; max-width: 60ch; margin: 0 0 12px; }
.numrow { display: flex; gap: 36px; flex-wrap: wrap; margin-top: 14px; }
.numrow div b { display: block; font-family: 'Newsreader', Georgia, serif; font-size: 32px; font-weight: 600; }
.numrow div span { color: var(--ink-3); font-size: 13px; }
.gwrap svg { width: 100%; height: auto; display: block; }
.needle { transform-origin: 200px 190px; transform: rotate(var(--rot)); animation: sweep 1.2s cubic-bezier(.2,.8,.2,1) both; }
@keyframes sweep { from { transform: rotate(0deg); } }

/* cards + chart blocks */
.viz { display: grid; grid-template-columns: 1fr 1.5fr; gap: 20px; margin-top: 30px; }
.card { background: var(--white); border: 1px solid var(--line); border-radius: 12px; padding: 20px 22px; }
.card h4 { font-size: 13px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: var(--ink-3); margin: 0 0 14px; }
.dn { display: flex; align-items: center; gap: 18px; }
.dn svg { width: 132px; height: 132px; flex: none; }
.dn svg .arc { animation: grow 1.1s ease-out both; }
@keyframes grow { from { stroke-dasharray: 0 var(--c); } }
.dl { font-size: 15px; line-height: 1.9; color: var(--ink-2); }
.dl .k, .legend2 .k { display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 8px; vertical-align: -1px; }
.k.fk { background: var(--red); } .k.rk { background: var(--teal); }
.stack { display: flex; height: 30px; border-radius: 8px; overflow: hidden; background: #E4E8F0; }
.stack i { display: block; height: 100%; animation: widen 1s ease-out both; transform-origin: left; }
@keyframes widen { from { transform: scaleX(0); } }
.sw { background: var(--ink); } .ss { background: var(--gold); }
.stack-axis { display: flex; justify-content: space-between; color: var(--ink-3); font-size: 12px; margin-top: 6px; }
.legend2 { display: flex; gap: 18px; flex-wrap: wrap; font-size: 14px; color: var(--ink-2); margin-top: 10px; }
.legend2 .tot { margin-left: auto; font-weight: 700; color: var(--ink); }
p.mini { color: var(--ink-3); font-size: 13px; line-height: 1.5; margin: 10px 0 0; }

/* sections */
.sec { margin-top: 44px; }
.sec h2 { font-family: 'Newsreader', Georgia, serif; font-size: 26px; font-weight: 600; margin: 0 0 6px; letter-spacing: -0.01em; }
.sec p.note { color: var(--ink-2); font-size: 15px; line-height: 1.55; margin: 0 0 16px; max-width: 64ch; }

/* the marked-up article */
.proof { background: var(--white); border: 1px solid var(--line); border-left: 4px solid var(--ink);
         border-radius: 4px; padding: 22px 26px; max-height: 340px; overflow: auto;
         font-family: 'Newsreader', Georgia, serif; font-size: 18px; line-height: 1.7; }
.proof h3 { font-size: 22px; line-height: 1.25; margin: 0 0 10px; font-weight: 700; }
.proof mark { padding: 1px 3px; border-radius: 3px; color: var(--ink); }
.proof mark.real { background: rgba(23,116,90, var(--a)); border-bottom: 2px solid var(--teal); }
.proof mark.fake { background: rgba(180,35,24, var(--a)); border-bottom: 2px solid var(--red); }
.legend { display: flex; gap: 22px; margin-top: 12px; color: var(--ink-2); font-size: 13px; }
.legend span::before { content: ""; display: inline-block; width: 14px; height: 10px; border-radius: 2px; margin-right: 8px; vertical-align: -1px; }
.legend .lf::before { background: rgba(180,35,24,.35); border-bottom: 2px solid var(--red); }
.legend .lr::before { background: rgba(23,116,90,.35); border-bottom: 2px solid var(--teal); }

/* diverging word chart */
.dvhead { display: grid; grid-template-columns: 110px 1fr 56px; font-size: 13px; color: var(--ink-3); margin-bottom: 6px; }
.dvhead span:first-child { grid-column: 2; justify-self: start; } .dvhead span:last-child { grid-column: 2; justify-self: end; margin-top: -18px; }
.dv { display: grid; grid-template-columns: 110px 1fr 56px; align-items: center; gap: 10px; font-size: 14px; margin-bottom: 7px; }
.dv .w { text-align: right; font-weight: 600; }
.dv .trk { position: relative; height: 14px; background: #EDF0F6; border-radius: 7px; }
.dv .trk::after { content: ""; position: absolute; left: 50%; top: -3px; bottom: -3px; width: 1px; background: var(--ink-3); }
.dv .trk i { position: absolute; top: 0; bottom: 0; border-radius: 7px; animation: widen .9s ease-out both; }
.dv .trk i.neg { background: var(--red); transform-origin: right; } .dv .trk i.pos { background: var(--teal); transform-origin: left; }
.dv .v { font-size: 12px; color: var(--ink-3); } .dv .v.neg { color: var(--red); } .dv .v.pos { color: var(--teal); }
.empty { color: var(--ink-3); font-size: 14px; }

/* source waterfall */
.wf { display: grid; grid-template-columns: 250px 1fr 48px; align-items: center; gap: 14px; padding: 7px 0; border-bottom: 1px solid #E4E8F0; }
.wf .lab b { display: block; font-size: 14px; } .wf .lab small { display: block; color: var(--ink-3); font-size: 12px; line-height: 1.35; }
.wf .trk { position: relative; height: 16px; background: #EDF0F6; border-radius: 8px;
           background-image: linear-gradient(to right, transparent calc(50% - .5px), #C3CADB calc(50% - .5px), #C3CADB calc(50% + .5px), transparent calc(50% + .5px)); }
.wf .trk i { position: absolute; top: 0; bottom: 0; border-radius: 8px; animation: widen .9s ease-out both; transform-origin: left; }
.wf .trk i.base { background: #8A94AD; } .wf .trk i.pos { background: var(--teal); } .wf .trk i.neg { background: var(--red); } .wf .trk i.fin { background: var(--ink); }
.wf .pts { font-variant-numeric: tabular-nums; font-weight: 700; text-align: right; font-size: 15px; }
.wf .pts.pos { color: var(--teal); } .wf .pts.neg { color: var(--red); }
.wf.fin { border-bottom: none; }
.wf.axis { border: none; padding: 2px 0 0; } .wf .ax { display: flex; justify-content: space-between; font-size: 12px; color: var(--ink-3); }

/* signal tiles */
.tiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
.tile { background: var(--white); border: 1px solid var(--line); border-radius: 12px; padding: 16px 18px; }
.tile b { display: block; font-family: 'Newsreader', Georgia, serif; font-size: 32px; font-weight: 600; }
.tile span { display: block; color: var(--ink-3); font-size: 13px; margin: 2px 0 10px; }
.mb { height: 6px; background: #E4E8F0; border-radius: 3px; overflow: hidden; }
.mb i { display: block; height: 100%; background: var(--gold); border-radius: 3px; animation: widen .9s ease-out both; transform-origin: left; }

/* stats strip + tables */
.numrow.glance div b { font-size: 30px; }
table.t { width: 100%; border-collapse: collapse; font-size: 15px; }
table.t th { text-align: left; font-weight: 600; color: var(--ink-2); font-size: 13px; border-bottom: 1px solid var(--ink); padding: 8px 10px 8px 0; }
table.t td { border-bottom: 1px solid var(--line); padding: 10px 10px 10px 0; vertical-align: top; }
table.t td.n { font-variant-numeric: tabular-nums; white-space: nowrap; font-weight: 600; }

/* model panel */
.mgrid { display: grid; grid-template-columns: 1fr 1fr; gap: 26px; margin: 14px 0; }
.cm { display: grid; grid-template-columns: 92px 1fr 1fr; gap: 6px; }
.cm .ch { font-size: 12px; font-weight: 600; color: var(--ink-3); text-align: center; align-self: end; }
.cm .rh { font-size: 12px; font-weight: 600; color: var(--ink-3); align-self: center; }
.cm .c { border-radius: 8px; padding: 14px 8px; text-align: center; }
.cm .c b { display: block; font-family: 'Newsreader', Georgia, serif; font-size: 26px; }
.cm .c span { display: block; font-size: 12px; color: var(--ink-2); }
.cm .c em { display: block; font-size: 11px; font-style: normal; font-weight: 700; letter-spacing: .05em; text-transform: uppercase; color: var(--ink-2); margin-top: 2px; }
.mrow { display: grid; grid-template-columns: 120px 1fr 62px; align-items: center; gap: 10px; font-size: 14px; margin-bottom: 10px; }
.mrow b { text-align: right; font-variant-numeric: tabular-nums; }
.mrow .mb { height: 10px; border-radius: 5px; } .mrow .mb i { background: var(--ink); border-radius: 5px; }
.axisnote { font-size: 12px; color: var(--ink-3); margin: -2px 0 0; }

.caution { border-left: 4px solid var(--gold); background: #FBF6E6; padding: 14px 18px; font-size: 14px; line-height: 1.6; color: #4a3b0a; border-radius: 0 4px 4px 0; margin-top: 16px; }
.hist svg { width: 100%; height: auto; display: block; background: var(--white); border: 1px solid var(--line); border-radius: 12px; }
.foot { margin-top: 56px; padding-top: 16px; border-top: 1px solid var(--line); color: var(--ink-3); font-size: 13px; line-height: 1.6; }
details summary { cursor: pointer; font-weight: 600; }

@media (max-width: 760px) {
  .vgrid, .viz, .mgrid { grid-template-columns: 1fr; }
  .tiles { grid-template-columns: 1fr 1fr; }
  h1.lead { font-size: 36px; } .verdict .call { font-size: 42px; }
  .wf { grid-template-columns: 1fr 48px; } .wf .trk { grid-column: 1 / -1; grid-row: 2; }
  .wf .lab { grid-column: 1; } .wf .pts { grid-column: 2; grid-row: 1; }
  .dv, .dvhead { grid-template-columns: 84px 1fr 48px; }
}
@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; animation: none !important; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    h(
        """
<div class="mast">
  <div class="name">NewsGuard<span> AI</span></div>
  <div class="tag">Fake news detection and source credibility</div>
</div>
<h1 class="lead">Check an article before you share it.</h1>
<p class="sub">Paste a headline and the article text. NewsGuard AI scores the writing, shows
which words moved the score, and can check the website it came from.</p>
<div class="chips"><span><b>98.95%</b> test accuracy</span><span><b>44,689</b> articles in the dataset</span>
<span><b>5,000</b> word features</span><span><b>2</b> signals: writing and source</span></div>
"""
    ),
    unsafe_allow_html=True,
)

# ============================================================
# INPUT
# ============================================================

for key in ("headline", "article", "source_url"):
    st.session_state.setdefault(key, "")
st.session_state.setdefault("history", [])
st.session_state.setdefault("check_no", 0)


def load_sample(name: str):
    s = SAMPLES[name]
    st.session_state.headline = s["headline"]
    st.session_state.article = s["article"]
    st.session_state.source_url = s["url"]


st.caption("No article handy? Load an example:")
c1, c2, c3 = st.columns(3)
for col, key in zip((c1, c2, c3), SAMPLES):
    col.button(SAMPLES[key]["label"], key=f"s_{key}", on_click=load_sample, args=(key,), use_container_width=True)

headline = st.text_input("Headline", key="headline", placeholder="e.g. Scientists announce a major breakthrough")
article = st.text_area("Article text", key="article", height=200, placeholder="Paste the full article here")
source_url = st.text_input(
    "Article link (optional)",
    key="source_url",
    placeholder="https://example.com/news/story",
    help="Add the link to also check the website's credibility.",
)

analyze = st.button("Check article", type="primary")

# ============================================================
# ANALYSIS
# ============================================================

if analyze:
    if not headline.strip() and not article.strip():
        st.warning("Enter a headline or article text first.")
    else:
        content = f"{headline} {article}".strip()
        vec = tfidf.transform([content])
        probs = model.predict_proba(vec)[0]
        fake_p, real_p = float(probs[0]) * 100, float(probs[1]) * 100
        predicted_real = int(model.predict(vec)[0]) == 1
        confidence = max(fake_p, real_p)

        source = score_source(source_url)
        if source:
            src_score, src_factors, host = source
            combined = W_CONTENT * real_p + W_SOURCE * src_score
        else:
            src_score = None
            combined = real_p

        tone = band(combined)
        call = {"real": "Likely credible", "mixed": "Mixed signals", "fake": "Likely unreliable"}[tone]

        st.session_state.check_no += 1
        st.session_state.history.append((st.session_state.check_no, combined, tone))
        st.session_state.history = st.session_state.history[-10:]

        reason = (
            f"The writing matches patterns the model learned from {'real' if predicted_real else 'fake'} "
            f"news ({confidence:.1f}% model confidence)."
        )
        if source:
            reason += f" The source check for {html.escape(host)} scored {src_score}/100."
        else:
            reason += " No link was added, so this score uses the writing only."

        nums = (
            f'<div><b>{real_p:.0f}</b><span>Writing score (real)</span></div>'
            + (f'<div><b>{src_score}</b><span>Source score</span></div>' if source else "")
            + f'<div><b>{combined:.0f}</b><span>Credibility score</span></div>'
        )

        st.markdown(
            h(
                f"""
<div class="verdict" role="status">
  <div class="vgrid">
    <div>
      <div class="kicker">Result</div>
      <div class="call {tone}">{call}</div>
      <p>{reason}</p>
      <div class="numrow">{nums}</div>
    </div>
    <div class="gwrap">{gauge_svg(combined, tone)}</div>
  </div>
</div>
"""
            ),
            unsafe_allow_html=True,
        )

        # ---------- donut + score build-up ----------
        st.markdown(
            h(
                f"""
<div class="viz">
  <div class="card"><h4>Writing score</h4>
    <div class="dn">{donut_svg(real_p)}
      <div class="dl"><span class="k fk"></span>Fake {fake_p:.1f}%<br><span class="k rk"></span>Real {real_p:.1f}%</div>
    </div>
  </div>
  <div class="card"><h4>How the score is built</h4>{stack_bar(real_p, src_score)}</div>
</div>
"""
            ),
            unsafe_allow_html=True,
        )

        # ---------- marked-up article ----------
        _, contrib = explain(content)
        head_html = annotate(headline, contrib) if headline.strip() else ""
        body_html = annotate(article, contrib) if article.strip() else ""
        st.markdown(
            h(
                f"""
<div class="sec">
  <h2>What the model read</h2>
  <p class="note">The words below had the biggest effect on the writing score. Hover a word to see its weight.</p>
  <div class="proof">{f'<h3>{head_html}</h3>' if head_html else ''}{body_html}</div>
  <div class="legend"><span class="lf">Pushes toward fake</span><span class="lr">Pushes toward real</span></div>
</div>
"""
            ),
            unsafe_allow_html=True,
        )

        # ---------- diverging word chart ----------
        st.markdown(
            h(
                f"""
<div class="sec">
  <h2>Strongest words</h2>
  <p class="note">Each bar is the word's weight in the classifier, scaled by how prominent it is in this text.
  Red bars pull toward fake, green bars toward real.</p>
  <div class="card">{diverging_words(contrib)}</div>
</div>
"""
            ),
            unsafe_allow_html=True,
        )

        # ---------- source waterfall ----------
        if source:
            st.markdown(
                h(
                    f"""
<div class="sec">
  <h2>Source check: {html.escape(host)}</h2>
  <p class="note">A rule-based check on the web address only. It does not read the site itself,
  so treat it as a first filter. Each bar moves the score up or down from the starting 50.
  Final score is {int(W_CONTENT*100)}% writing and {int(W_SOURCE*100)}% source.</p>
  <div class="card">{source_waterfall(src_factors, src_score)}</div>
</div>
"""
                ),
                unsafe_allow_html=True,
            )

        # ---------- writing signals ----------
        words = TOKEN_RE.findall(content)
        sentences = [s for s in re.split(r"[.!?]+", content) if s.strip()]
        st.markdown(
            h(
                f"""
<div class="sec">
  <h2>Writing signals</h2>
  <p class="note">Simple counts that often stand out in sensational writing. They are shown for context only;
  the model does not use them.</p>
  <div class="tiles">{signal_tiles(content, len(words), len(sentences))}</div>
  <div class="numrow glance" style="margin-top:22px">
    <div><b>{len(words)}</b><span>Words</span></div>
    <div><b>{len(sentences)}</b><span>Sentences</span></div>
    <div><b>{len(content)}</b><span>Characters</span></div>
  </div>
</div>
"""
            ),
            unsafe_allow_html=True,
        )

        # ---------- session trend ----------
        hist = st.session_state.history
        if len(hist) >= 2:
            st.markdown(
                h(
                    f"""
<div class="sec hist">
  <h2>Your checks this session</h2>
  <p class="note">Each dot is one article you checked. Dot colour and the shaded zones show the verdict band.</p>
  {history_svg(hist)}
</div>
"""
                ),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="sec"><p class="note">Check a second article to see your results side by side as a trend line.</p></div>',
                unsafe_allow_html=True,
            )

# ============================================================
# ABOUT THE MODEL
# ============================================================

cm = CONFUSION
total = sum(cm.values())
acc = (cm["fake_fake"] + cm["real_real"]) / total * 100
p_real = cm["real_real"] / (cm["real_real"] + cm["fake_real"]) * 100
r_real = cm["real_real"] / (cm["real_real"] + cm["real_fake"]) * 100
p_fake = cm["fake_fake"] / (cm["fake_fake"] + cm["real_fake"]) * 100
r_fake = cm["fake_fake"] / (cm["fake_fake"] + cm["fake_real"]) * 100

st.markdown('<div class="sec"></div>', unsafe_allow_html=True)
with st.expander("About this model and its limits"):
    st.markdown(
        h(
            f"""
<p class="note">TF-IDF turns the text into 5,000 word features. A Logistic Regression classifier
trained on 44,689 labelled articles (80% train, 20% test) predicts fake or real.
Overall accuracy is <b>{acc:.2f}%</b> on {total:,} held-out articles.</p>
<div class="mgrid">
  <div><h4 style="font-size:13px;letter-spacing:.06em;text-transform:uppercase;color:#5A647D;margin:0 0 10px">Confusion matrix</h4>{confusion_heatmap(cm)}</div>
  <div><h4 style="font-size:13px;letter-spacing:.06em;text-transform:uppercase;color:#5A647D;margin:0 0 10px">Precision and recall</h4>
    {metric_bars([("Fake precision", p_fake), ("Fake recall", r_fake), ("Real precision", p_real), ("Real recall", r_real)])}
    <p class="axisnote">Bars run from 95% to 100% so small differences are visible.</p></div>
</div>
<div class="caution"><b>Read the score as a style check, not a fact check.</b>
The model learns the wording of its training data. Wire-service phrases such as
"Reuters" and "said" push strongly toward real, while words like "video", "via" and "watch"
push toward fake. A true story written casually can score low, and a false story written in
news-agency style can score high. Verify important claims with a trusted source.</div>
"""
        ),
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="foot">NewsGuard AI · B.Tech final year project, VIT · Built with Python, scikit-learn, TF-IDF, '
    "Logistic Regression and Streamlit</div>",
    unsafe_allow_html=True,
)