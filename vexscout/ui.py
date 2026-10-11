"""HTML/CSS building blocks for the Streamlit UI."""
from html import escape

ACCENT, YOU, RED, BLUE = "#5B8DEF", "#E0703A", "#E5484D", "#3E8EF7"
TIER_STYLE = {
    "Elite": ("#9DBDF8", "#0D1B33"), "Strong": ("#3E7BEA", "#FFFFFF"), "Solid": ("#24479A", "#E8EEFF"),
    "Developing": ("#18284D", "#B9C8EE"), "Building": ("#2A2C33", "#B4B8C2"), "Unrated": ("#202228", "#7D828D"),
}

CSS = """
<style>
.block-container {padding-top: 4.5rem; max-width: 1400px;}
header[data-testid="stHeader"] {background: rgba(13,14,18,.85); backdrop-filter: blur(6px);}
h1, h2, h3 {letter-spacing: -0.02em;}
.vs-title {font-size: 2.1rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.15; margin: 0;}
.vs-sub {color: #9AA0AB; font-size: 1.02rem; margin: .35rem 0 1.3rem;}
.vs-cards {display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: .6rem;}
.vs-card {background: #16181D; border: 1px solid #2A2D35; border-radius: 16px; padding: 18px 26px 16px; min-width: 150px;}
.vs-card .v {font-size: 1.9rem; font-weight: 700; line-height: 1.1;}
.vs-card .l {font-size: .74rem; letter-spacing: .12em; text-transform: uppercase; color: #B4B8C2; margin-top: 6px;}
.vs-chip {display: inline-block; padding: 3px 12px; border-radius: 999px; font-size: .82rem; font-weight: 600;
          background: #23262D; color: #D6D9DF; margin: 2px 6px 2px 0; white-space: nowrap;}
.vs-info {background: #142038; border: 1px solid #1F3157; border-radius: 16px; padding: 14px 20px; color: #D3DBEA;
          font-size: .92rem; line-height: 1.55; margin: .3rem 0 1rem;}
.vs-panel {background: #16181D; border: 1px solid #2A2D35; border-radius: 18px; padding: 20px 22px; margin-bottom: 14px;}
.vs-ring-wrap {display: flex; gap: 22px; align-items: center;}
.vs-ring-wrap .name {font-size: 1.6rem; font-weight: 800; letter-spacing: -0.02em;}
.vs-ring-wrap .org {color: #9AA0AB; margin: 2px 0 10px;}
.vs-row {display: grid; grid-template-columns: 130px 44px 1fr 64px; gap: 12px; align-items: center; padding: 9px 0;}
.vs-row .lab {font-size: .98rem;} .vs-row .w {color: #8B909B; font-size: .85rem; text-align: right;}
.vs-row .c {text-align: right; font-weight: 700; line-height: 1.15;} .vs-row .c small {display:block; color:#8B909B; font-weight:500; font-size:.75rem;}
.vs-track {position: relative; height: 14px; background: #262930; border-radius: 999px;}
.vs-fill {position: absolute; left: 0; top: 0; bottom: 0; border-radius: 999px; background: #5B8DEF;}
.vs-mid {position: absolute; left: 50%; top: -4px; bottom: -4px; width: 2px; background: #5B606B;}
.vs-you {position: absolute; top: -5px; bottom: -5px; width: 4px; border-radius: 2px; background: #E0703A; margin-left: -2px;}
.vs-legend {display: flex; gap: 18px; flex-wrap: wrap; color: #B4B8C2; font-size: .85rem; margin-top: 10px;}
.vs-legend i {display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; vertical-align: -1px;}
.vs-sw h4 {font-size: .85rem; letter-spacing: .12em; text-transform: uppercase; margin: 0 0 2px;}
.vs-sw .src {font-size: .72rem; letter-spacing: .12em; color: #8B909B; text-transform: uppercase; margin-bottom: 8px;}
.vs-sw li {margin: 4px 0; list-style: none;} .vs-sw ul {padding-left: 0; margin: 0;}
.vs-pick {display: grid; grid-template-columns: 84px 1fr auto; gap: 14px; align-items: center;}
.vs-pick .n {font-size: 1.3rem; font-weight: 800; color: #8B909B;}
.vs-pick .t {font-size: 1.1rem; font-weight: 700;} .vs-pick .s {color: #9AA0AB; font-size: .88rem;}
.vs-pick .big {font-size: 1.6rem; font-weight: 800; text-align: right;} .vs-pick .big small {display:block; font-size:.72rem; color:#8B909B; font-weight:500; letter-spacing:.08em;}
.vs-vs {display: grid; grid-template-columns: 1fr auto 1fr; gap: 16px; align-items: center; text-align: center;}
.vs-vs .sc {font-size: 3rem; font-weight: 800; line-height: 1;}
.vs-vs .tm {color: #B4B8C2; margin-top: 6px;}
.vs-split {display: flex; height: 16px; border-radius: 999px; overflow: hidden; margin: 18px 0 6px;}
.vs-split-l {display: flex; justify-content: space-between; font-weight: 600; font-size: .92rem;}
.vs-ai {background: linear-gradient(180deg,#171B26,#15171D); border: 1px solid #2E3A57; border-radius: 18px; padding: 18px 22px; margin-bottom: 14px;}
.vs-ai h4 {margin: 0 0 6px; font-size: 1.05rem;} .vs-ai .tag {font-size:.72rem; letter-spacing:.12em; color:#9DBDF8; text-transform:uppercase;}
.vs-ai ul {margin: .3rem 0 .6rem; padding-left: 1.1rem;} .vs-ai li {margin: 3px 0;}
.vs-ai .ev {color: #9AA0AB; font-size: .86rem;}
.vs-scout-h {font-size: 1.1rem; font-weight: 700; margin: 4px 0 2px;} .vs-scout-h span {color:#9AA0AB; font-weight:400; font-size:.92rem; margin-left:8px;}
.vs-lbl {font-size: .78rem; letter-spacing: .12em; text-transform: uppercase; color: #9AA0AB; font-weight: 600; margin: 12px 0 2px;}
/* compact header */
.vs-head {display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; margin-bottom: 2px;}
.vs-head .vs-title {font-size: 1.75rem;}
.vs-badge {display: inline-block; padding: 3px 10px; border-radius: 8px; font-size: .8rem; font-weight: 600;
           background: #3A2A10; color: #F5B95A; border: 1px solid #F5B95A44; white-space: nowrap;}
.vs-stats {display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0 6px;}
.vs-stat {background: #16181D; border: 1px solid #2A2D35; border-radius: 10px; padding: 6px 12px; font-size: .88rem; color: #B4B8C2;}
.vs-stat b {color: #F2F3F5; font-size: 1rem; margin-right: 4px;}
/* sticky navigation */
.stTabs [role="tablist"] {position: sticky; top: 3.6rem; z-index: 50; box-shadow: 0 6px 16px #0D0E12;}
/* evidence chips linking claims to matches */
.vs-ref {display: inline-block; font-size: .74rem; padding: 1px 7px; margin: 2px 4px 0 0; border-radius: 6px;
         background: #1F2B45; color: #9DBDF8; border: 1px solid #2E3F66; cursor: help; white-space: nowrap;}
/* three-card summary at the top of AI reports */
.vs-top3 {display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin: 10px 0 12px;}
.vs-top3 > div {background: #11141B; border: 1px solid #2A2D35; border-radius: 12px; padding: 12px 14px;}
.vs-top3 .k {font-size: .7rem; letter-spacing: .12em; text-transform: uppercase; font-weight: 700; margin-bottom: 4px;}
.vs-top3 .v {font-weight: 600; line-height: 1.35;}
.vs-top3 .e {color: #9AA0AB; font-size: .84rem; margin-top: 4px;}
/* prediction uncertainty, as prominent as the scores */
.vs-predict {display: grid; grid-template-columns: 1.4fr 1fr; gap: 12px;}
.vs-unc {border-radius: 18px; padding: 18px 20px;}
.vs-unc .lvl {font-size: 1.6rem; font-weight: 800; line-height: 1.1;}
.vs-unc .pm {font-size: 1.15rem; font-weight: 700; margin: 6px 0;}
.vs-unc ul {margin: 6px 0 0; padding-left: 1.1rem;} .vs-unc li {margin: 2px 0;}
.vs-blend {margin-top: 10px; font-size: .88rem; color: #B4B8C2; line-height: 1.5;}
.vs-blend b {color: #F2F3F5; font-variant-numeric: tabular-nums;}
.vs-blend span {display: block; font-size: .78rem; color: #8B909B;}
/* team cards for phones */
.vs-tcard {display: grid; grid-template-columns: 1fr auto; gap: 4px 10px; align-items: center;}
.vs-tcard .num {font-size: 1.15rem; font-weight: 800;} .vs-tcard .nm {color: #9AA0AB; font-size: .88rem;}
.vs-tcard .sc {font-size: 1.9rem; font-weight: 800; text-align: right; line-height: 1;}
.vs-tcard .meta {grid-column: 1 / -1; font-size: .84rem;}
.st-key-mobile_cards {display: none;}
.st-key-mobile_nav, .vs-mtitle {display: none;}
@media (max-width: 700px) {
  .block-container {padding: 3.6rem 0.9rem 2rem;}
  .st-key-desk_table, [class*="st-key-desk_only"] {display: none !important;}
  /* phone menu: sections live in the slide-out sidebar instead of a swipeable tab bar */
  .stTabs [role="tablist"] {display: none !important;}
  .st-key-mobile_nav {display: block;}
  .st-key-mobile_nav button {justify-content: flex-start !important; padding: 10px 14px; min-height: 46px;}
  .st-key-mobile_nav button > div, .st-key-mobile_nav button span {justify-content: flex-start !important; text-align: left;}
  .st-key-mobile_nav button [data-testid="stIconMaterial"] {margin-right: 10px;}
  .st-key-mobile_nav button p {font-size: 1rem; font-weight: 600;}
  .vs-mtitle {display: block; font-family: inherit; font-size: 1.15rem; font-weight: 700; margin: 2px 0 2px;
              padding-left: 10px; border-left: 3px solid #5B8DEF;}
  [data-testid="stExpandSidebarButton"] {width: auto !important; height: 38px; padding: 0 12px !important;
      border: 1px solid #3A3F4B !important; border-radius: 10px !important; background: #1A1C22 !important;}
  [data-testid="stExpandSidebarButton"] > * {display: none !important;}
  [data-testid="stExpandSidebarButton"]::before {content: "☰  Menu"; white-space: pre; color: #E8EAED;
      font-weight: 600; font-size: .95rem;}
  .vs-stats {flex-wrap: nowrap; overflow-x: auto; scrollbar-width: none; margin: 2px 0;}
  .vs-stat {white-space: nowrap; flex: none;}
  .vs-stat:last-child {display: none;}
  .vs-head {gap: 4px 8px;}
  .vs-badge {font-size: .72rem; padding: 2px 8px;}
  .st-key-mobile_cards {display: block;}
  .vs-head .vs-title {font-size: 1.25rem;}
  .vs-sub {font-size: .8rem; margin: .1rem 0 .4rem;}
  .vs-stat {padding: 4px 9px; font-size: .78rem;} .vs-stat b {font-size: .88rem;}
  .vs-predict {grid-template-columns: 1fr;}
  .vs-vs .sc {font-size: 2.2rem;}
  .stTabs [role="tablist"] {top: 3.2rem; padding: 4px;}
  .stTabs [role="tab"] {padding: 7px 11px;} .stTabs [role="tab"] p {font-size: .86rem;}
  .vs-row {grid-template-columns: 96px 34px 1fr 52px; gap: 8px;}
  /* let long text wrap instead of running off the screen */
  .stButton button p, .stButton button div {white-space: normal; text-align: center; line-height: 1.3;}
  .vs-chip {white-space: normal;}
  .vs-ring-wrap {flex-wrap: wrap; gap: 12px;}
  .vs-ring-wrap > div:first-child {width: 104px !important; height: 104px !important;}
  .vs-ring-wrap > div:first-child > div {width: 88px !important; height: 88px !important;}
  .vs-ring-wrap > div:first-child > div > div:first-child {font-size: 30px !important;}
  .vs-ring-wrap > div:last-child {min-width: 0; flex: 1 1 200px;}
  .vs-ring-wrap .name {font-size: 1.25rem; overflow-wrap: anywhere;}
  .vs-ring-wrap .org {font-size: .85rem;}
  .vs-panel {padding: 14px 14px;}
  .vs-pick {grid-template-columns: 72px 1fr;}
  .vs-pick > div:last-child {grid-column: 1 / -1; justify-content: space-around;}
  .vs-sw, .vs-top3 {min-width: 0;}
  .vs-top3 {grid-template-columns: 1fr;}
  [data-testid="stDialog"] .vs-panel[style*="grid-template-columns:1fr 1fr"] {grid-template-columns: 1fr !important;}
}
/* pill-style tabs */
.stTabs [role="tablist"] {background: #16181D; border: 1px solid #2A2D35; border-radius: 16px; padding: 6px; gap: 4px;
                          width: fit-content; max-width: 100%; overflow-x: auto;}
.stTabs [role="tab"] {border-radius: 11px; padding: 9px 20px; margin: 0; border: none;}
.stTabs [role="tab"] p {font-size: 1rem; font-weight: 600; color: #B4B8C2;}
.stTabs [role="tab"][aria-selected="true"] {background: #3A3F4B;}
.stTabs [role="tab"][aria-selected="true"] p {color: #F2F3F5;}
.stTabs [role="tab"]:hover p {color: #F2F3F5;}
</style>
"""


def cards(items):
    return '<div class="vs-cards">' + "".join(
        f'<div class="vs-card"><div class="v">{escape(str(v))}</div><div class="l">{escape(l)}</div></div>' for v, l in items) + "</div>"


def chip(text, bg=None, fg=None):
    style = f' style="background:{bg};color:{fg}"' if bg else ""
    return f'<span class="vs-chip"{style}>{escape(str(text))}</span>'


def tier_chip(t):
    bg, fg = TIER_STYLE.get(t, TIER_STYLE["Unrated"])
    return chip(t, bg, fg)


def ring(score, size=150, caption="OVERALL", color="#9DBDF8", inner="#16181D"):
    """Progress ring in plain CSS (conic-gradient). st.html strips SVG, so no <svg> here.
    `inner` must match the background the ring sits on."""
    frac = max(0, min(1, (score or 0) / 100))
    label = "–" if score is None else str(score)
    thick = max(6, round(size * 0.075))
    num = round(size * (0.27 if caption else 0.3))
    cap = (f'<div style="font-size:{max(9, round(size * 0.085))}px;letter-spacing:.12em;color:#B4B8C2;'
           f'margin-top:{round(size * 0.02)}px">{escape(caption)}</div>') if caption else ""
    return (f'<div style="width:{size}px;height:{size}px;border-radius:50%;flex:none;display:grid;place-items:center;'
            f'background:conic-gradient({color} {frac * 360:.1f}deg, #262930 0)">'
            f'<div style="width:{size - 2 * thick}px;height:{size - 2 * thick}px;border-radius:50%;background:{inner};'
            f'display:flex;flex-direction:column;align-items:center;justify-content:center;line-height:1">'
            f'<div style="font-size:{num}px;font-weight:800;color:#F2F3F5">{label}</div>{cap}</div></div>')


def score_color(score):
    """Blue scale for 0-100 scores: brighter = better."""
    return "#9DBDF8" if score >= 75 else "#5B8DEF" if score >= 60 else "#3E6BC4" if score >= 45 else "#5B606B"


def trait_chip(label, good, source=""):
    bg, fg, mark = ("#12301C", "#6BD58A", "▲") if good else ("#3A1517", "#FF8A8D", "▼")
    return (f'<span class="vs-chip" title="{escape(source)}" style="background:{bg};color:{fg};'
            f'border:1px solid {fg}33">{mark} {escape(label)}</span>')


def bar_row(label, weight, pct, contrib, you_pct=None):
    if pct is None:
        track = '<div class="vs-track"><div class="vs-mid"></div></div>'
        top = f"+{contrib:.0f}" if contrib is not None else "–"
        right = f'<div class="c" style="color:#6B707B">{top}<small>no data · avg</small></div>'
    else:
        you = f'<div class="vs-you" style="left:{you_pct:.0f}%"></div>' if you_pct is not None else ""
        track = (f'<div class="vs-track"><div class="vs-fill" style="width:{max(pct, 2):.0f}%"></div>'
                 f'<div class="vs-mid"></div>{you}</div>')
        top = f"+{contrib:.0f}" if contrib is not None else f"{pct:.0f}"
        right = f'<div class="c">{top}<small>pct {pct:.0f}</small></div>'
    return f'<div class="vs-row"><div class="lab">{escape(label)}</div><div class="w">{escape(weight)}</div>{track}{right}</div>'


def split_bar(p_red):
    return (f'<div class="vs-split"><div style="width:{p_red * 100:.1f}%;background:{RED}"></div>'
            f'<div style="flex:1;background:{BLUE}"></div></div>'
            f'<div class="vs-split-l"><span style="color:{RED}">Red {p_red:.0%}</span>'
            f'<span style="color:{BLUE}">Blue {1 - p_red:.0%}</span></div>')
