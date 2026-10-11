"""VEX Scout AI - Streamlit UI.  Run: streamlit run app.py"""
import json
import re
from datetime import datetime

import pandas as pd
import streamlit as st

from vexscout import analysis, db, feedback, mic, ml, navjs, predictor, ratings, scouting, traits, ui
from vexscout.config import MY_TEAM_ID, MY_TEAM_NUMBER, ROBOTEVENTS_TOKEN, SEASON_ID

st.set_page_config(page_title="VEX Scout AI", page_icon="assets/icon.png", layout="wide")
st.logo("assets/logo.svg", icon_image="assets/icon.svg", size="large")
st.html(ui.CSS)


@st.cache_resource
def get_conn():
    return db.connect()


conn = get_conn()
if "data_version" not in st.session_state:
    st.session_state.data_version = 0


def bump():
    st.session_state.data_version += 1


@st.cache_resource(max_entries=8)
def cached_model(event_id, _version, _day):  # _day: "today only" feedback expires at midnight
    m = predictor.build_model(conn, event_id)
    return m, ratings.rate_event(conn, event_id, m), ml.train(conn, event_id)


# RobotEvents responses persist in the api_cache table (6 h), so these survive restarts too
@st.cache_data(ttl=600, show_spinner=False)
def my_season_events():
    from vexscout.robotevents import RobotEvents, _event_row
    return [_event_row(e) for e in RobotEvents(ttl=6 * 3600).team_events(MY_TEAM_ID, SEASON_ID)]


@st.cache_data(ttl=600, show_spinner="Loading this season's competitions…")
def all_season_events(season_id):
    from vexscout.robotevents import season_events
    return season_events(season_id)


@st.cache_data(ttl=3600, show_spinner=False)
def my_team_info():
    from vexscout.robotevents import team_info
    return team_info(MY_TEAM_ID)


def fmt_date(iso):
    try:
        return datetime.fromisoformat(iso).strftime("%b %-d, %Y")
    except (TypeError, ValueError):
        return ""


def import_event(sku, hist=True):
    from vexscout.robotevents import ingest_event
    log = st.empty()
    try:
        with st.spinner(f"Pulling {sku} from RobotEvents…"):
            eid = ingest_event(sku.strip(), with_history=hist, log=lambda s: log.caption(s))
        st.session_state.pending_event = eid  # applied before the selectbox is built
        bump(); st.rerun()
    except Exception as e:
        st.error(f"Import failed: {e}")


# --- sidebar: your team, event selection, data ---------------------------------
TABS = ["Teams", "Live updates", "Scouting notes", "Alliance picks", "Match predictor", "Strategy copilot"]
TAB_ICONS = {"Teams": ":material/groups:", "Live updates": ":material/campaign:", "Scouting notes": ":material/edit_note:",
             "Alliance picks": ":material/handshake:", "Match predictor": ":material/sports_score:",
             "Strategy copilot": ":material/forum:"}


def _go(section):
    st.session_state.goto_tab = section
    st.session_state.close_nav = st.session_state.get("close_nav", 0) + 1  # phones: close the menu afterwards


with st.sidebar:
    # Phone menu (shown only at phone width, replacing the swipeable tab bar)
    with st.container(key="mobile_nav"):
        current = st.session_state.get("goto_tab") or st.session_state.get("main_tab") or TABS[0]
        for name in TABS:
            st.button(name, key=f"nav_{name}", icon=TAB_ICONS[name], width="stretch", on_click=_go, args=(name,),
                      type="primary" if name == current else "tertiary")
        st.divider()
    if n := st.session_state.get("close_nav"):
        if st.session_state.get("closed_nav") != n:
            st.session_state.closed_nav = n
            navjs.close_sidebar(key=f"close_nav_{n}")
    if MY_TEAM_NUMBER:
        st.html(f'<div style="color:#9AA0AB;margin:-6px 0 8px">Scouting for {ui.chip(MY_TEAM_NUMBER + " (you)", ui.YOU, "#fff")}</div>')
    events = [e for e in db.list_events(conn) if e["source"] != "synthetic"
              and db.one(conn, "SELECT 1 FROM event_teams WHERE event_id=?", (e["id"],))]
    ids = [e["id"] for e in events]
    if st.session_state.get("pending_event") in ids:
        st.session_state.event_id = st.session_state.pop("pending_event")
    if events and st.session_state.get("event_id") not in ids:
        mine = [e["id"] for e in events if MY_TEAM_ID and db.one(
            conn, "SELECT 1 FROM event_teams WHERE event_id=? AND team_id=?", (e["id"], MY_TEAM_ID))]
        st.session_state.event_id = (mine or ids)[0]
    event = None
    if events:
        eid = st.selectbox("Event", ids, key="event_id",
                           format_func=lambda i: next(f"{e['name'][:60]} ({e['sku']})" for e in events if e["id"] == i))
        event = next(e for e in events if e["id"] == eid)

    if ROBOTEVENTS_TOKEN:
        from vexscout.robotevents import miles
        st.markdown("**Find a competition**")
        modes = (["Signed up"] if MY_TEAM_ID else []) + ["Near me", "Search"]
        mode = st.segmented_control("Find", modes, default=modes[0], key="find_mode", label_visibility="collapsed") or modes[0]
        season = SEASON_ID or (event or {}).get("season_id")
        today = datetime.now().date().isoformat()
        loaded = {e["sku"]: e["id"] for e in events}
        results, home = [], None
        try:
            if mode == "Signed up":
                results = sorted(my_season_events(), key=lambda e: e["start"])
                if not results:
                    st.caption(f"{MY_TEAM_NUMBER} isn't registered for any events this season yet.")
            else:
                if not season:
                    st.caption("Set SEASON_ID in .env to search competitions.")
                else:
                    pool = all_season_events(season)
                    upcoming = st.toggle("Upcoming only", value=True, key="find_upcoming")
                    if upcoming:
                        pool = [e for e in pool if e["start"] >= today]
                    if mode == "Near me":
                        home = my_team_info() if MY_TEAM_ID else None
                        if not home or home["lat"] is None:
                            st.caption("Set MY_TEAM_ID in .env so we know where your team is.")
                        else:
                            radius = st.select_slider("Within", [10, 25, 50, 100, 250, 500], value=50, key="find_radius",
                                                      format_func=lambda m: f"{m} miles of {home['city']}")
                            for e in pool:
                                e["dist"] = miles(home["lat"], home["lon"], e["lat"], e["lon"])
                            results = sorted((e for e in pool if e["dist"] is not None and e["dist"] <= radius),
                                             key=lambda e: (e["start"], e["dist"]))
                            if not results:
                                st.caption("Nothing in range - try a bigger distance.")
                    else:
                        q = st.text_input("Search", key="find_q", placeholder="Name, city, state or SKU",
                                          label_visibility="collapsed").strip().lower()
                        if q:
                            results = sorted((e for e in pool if q in " ".join(
                                str(e[k] or "") for k in ("name", "city", "region", "sku", "venue")).lower()),
                                key=lambda e: e["start"])
                            if not results:
                                st.caption("No matches.")
        except Exception as e:
            st.caption(f"Couldn't reach RobotEvents: {e}")

        for e in results[:12]:
            where = ", ".join(x for x in (e["city"], e["region"]) if x)
            dist = f" · {e['dist']:.0f} mi" if e.get("dist") is not None else ""
            with st.container(border=True):
                st.html(f'<div style="font-size:.88rem;font-weight:600;line-height:1.3">{ui.escape(e["name"][:80])}</div>'
                        f'<div style="font-size:.78rem;color:#9AA0AB;margin-top:2px">{fmt_date(e["start"])} · '
                        f'{ui.escape(where)}{dist}</div>')
                if e["sku"] in loaded:
                    if loaded[e["sku"]] == (event or {}).get("id"):
                        st.caption("✓ Showing now")
                    elif st.button("Open", key=f"open_{e['sku']}", width="stretch"):
                        st.session_state.pending_event = loaded[e["sku"]]; st.rerun()
                elif st.button("Load", key=f"load_{e['sku']}", width="stretch", type="primary"):
                    import_event(e["sku"], st.session_state.get("find_hist", True))
        if len(results) > 12:
            st.caption(f"Showing 12 of {len(results)} - narrow the search to see more.")

    with st.expander("More options", expanded=not events):
        if ROBOTEVENTS_TOKEN:
            st.checkbox("Include each team's season history", value=True, key="find_hist",
                        help="One API call per team when loading an event; slower but much better ratings.")
        else:
            st.caption("Set ROBOTEVENTS_TOKEN in .env to load real competitions.")

if not event:
    st.html('<div class="vs-title">Welcome to VEX Scout AI</div>'
            '<div class="vs-sub">Find a competition in the sidebar and press Load.</div>')
    st.stop()

EID = event["id"]
with st.spinner("Crunching ratings from this season's matches…"):
    model, R, ML = cached_model(EID, st.session_state.data_version, datetime.now().date().isoformat())
teams = db.event_teams(conn, EID)
numbers = [t["number"] for t in teams]
by_number = {t["number"]: t for t in teams}
me = R.get(MY_TEAM_ID) if MY_TEAM_ID in R else None
notes = {r["team_id"]: r["n"] for r in db.rows(
    conn, "SELECT team_id, count(*) n FROM scouting_observations WHERE event_id=? AND status!='superseded' GROUP BY team_id", (EID,))}

T = {tid: traits.team_traits(conn, r, EID) for tid, r in R.items()}
N_RATED = sum(1 for r in R.values() if r.overall is not None)
CONF_HELP = ("Data confidence = how much verified match data backs the numbers "
             "(high: 8+ season matches, medium: 3–7, low: fewer).")
AI_CERT_HELP = ("AI certainty = how sure Claude is of its read, after weighing sample size, how recent the data is, "
                "and any conflicting scouting notes. It can be lower than data confidence.")


def split_title(name):
    """'X (Moved to SAT 10/17): VEX V5 Robotics Competition' -> ('X', ['Moved to SAT 10/17'])."""
    badges = re.findall(r"\(([^)]*)\)", name)
    base = re.sub(r"\s*\([^)]*\)", "", name)
    head, _, tail = base.partition(":")
    if tail and re.search(r"vex|robotics competition", tail, re.I):
        base = head
    return base.strip(), [b.strip() for b in badges]


SHORT_NAME, NAME_BADGES = split_title(event["name"])


# --- header --------------------------------------------------------------------
badges = "".join(f'<span class="vs-badge">📅 {ui.escape(b)}</span>' for b in NAME_BADGES)
sub = " · ".join(x for x in (fmt_date(event["start"]), event.get("venue"), event.get("location"), event["sku"]) if x)
st.html(f'<div class="vs-head"><div class="vs-title" title="{ui.escape(event["name"])}">{ui.escape(SHORT_NAME)}</div>{badges}</div>'
        f'<div class="vs-sub">{ui.escape(sub)}</div>')

quals = db.rows(conn, "SELECT scored FROM matches WHERE event_id=? AND round=2", (EID,))
played = sum(q["scored"] for q in quals)
qual_label = "not started" if not played else "complete" if played == len(quals) else f"{played}/{len(quals)} played"
stats = [(len(teams), f"teams · {N_RATED} rated"), ("Qualifiers", qual_label),
         (f"{model.mean_alliance:.0f}" if model.n_train else "–", "avg alliance score"),
         (sum(notes.values()), "scouting notes")]
live = event["source"] == "robotevents" and ROBOTEVENTS_TOKEN


def refresh():
    from vexscout.robotevents import refresh_event_matches
    refresh_event_matches(EID); bump()


c1, c2, c3 = st.columns([6, 1.3, 1.6], vertical_alignment="center")
as_of = (" · data as of " + datetime.fromisoformat(event["fetched_at"]).astimezone().strftime("%b %-d, %-I:%M %p")
         if event.get("fetched_at") else "")
c1.html('<div class="vs-stats">' + "".join(f'<div class="vs-stat"><b>{ui.escape(str(v))}</b>{ui.escape(l)}</div>'
                                           for v, l in stats)
        + f'<div class="vs-stat" style="border:none;background:none;padding-left:2px">{ui.escape(as_of)}</div></div>')
if live and c2.button("↻ Refresh", help="Pull the latest match results from RobotEvents", width="stretch"):
    try:
        with st.spinner("Getting the latest results from RobotEvents…"):
            refresh()
        st.rerun()
    except Exception as e:
        st.error(f"Refresh failed (showing saved data): {e}")
with c3.container(key="desk_only_auto"):
    auto = st.toggle("Auto-refresh 15 min", key="auto", disabled=not live)


@st.fragment(run_every="15m")
def auto_refresh():
    if st.session_state.get("auto") and live:
        try:
            refresh(); st.rerun()
        except Exception:
            pass


auto_refresh()


# --- team profile dialog -------------------------------------------------------
def recent_matches(team_id):
    ms = db.scored_matches(conn, team_id=team_id)[::-1][:12]
    nums = db.team_numbers(conn, {x for m in ms for x in (m["red1"], m["red2"], m["blue1"], m["blue2"]) if x})
    out = []
    for i, m in enumerate(ms, 1):
        c = "red" if team_id in (m["red1"], m["red2"]) else "blue"
        o = "blue" if c == "red" else "red"
        out.append({"Ref": f"M{i}", "Event": (m["event_name"] or "")[:40], "Match": m["name"], "Alliance": c.title(),
                    "Partner": ", ".join(nums.get(x, "?") for x in (m[f"{c}1"], m[f"{c}2"]) if x and x != team_id),
                    "Opponents": ", ".join(nums.get(x, "?") for x in (m[f"{o}1"], m[f"{o}2"]) if x),
                    "Score": f"{m[c + '_score']}–{m[o + '_score']}",
                    "Result": "W" if m[c + "_score"] > m[o + "_score"] else "L" if m[c + "_score"] < m[o + "_score"] else "T"})
    return out


def ref_index(ratings_list, prefixes):
    """{ref: match facts} for the matches Claude was shown, so cited refs can be displayed."""
    out = {}
    for rt in ratings_list:
        for m in analysis.team_facts(conn, EID, rt, 10, ref_prefix=prefixes.get(rt.team_id, rt.number + "-"))["recent_matches"]:
            out[m["ref"]] = dict(m, team=rt.number)
    return out


def ref_chips(ref_list, refs):
    chips = []
    for ref in ref_list:
        m = refs.get(ref)
        if not m:
            continue  # never show a citation we can't back up
        names = lambda xs: ", ".join(str(x or "?") for x in xs or [])
        tip = (f'{m["team"]} · {m["event"] or "event"} · {m["match"] or ""} · {m["score"]} · '
               f'partner {names(m["partner"]) or "–"} · vs {names(m["opponents"]) or "–"}')
        chips.append(f'<span class="vs-ref" title="{ui.escape(tip)}">{ui.escape(ref)} · '
                     f'{ui.escape(str(m["match"] or ""))} {ui.escape(str(m["score"]))}</span>')
    return "".join(chips)


def top3(*items):
    return '<div class="vs-top3">' + "".join(
        f'<div><div class="k" style="color:{color}">{ui.escape(k)}</div><div class="v">{ui.escape(v)}</div>'
        + (f'<div class="e">{ui.escape(e)}</div>' if e else "") + (f'<div>{chips}</div>' if chips else "") + "</div>"
        for k, color, v, e, chips in items) + "</div>"


def claims(items, refs):
    return "".join(f'<li>{ui.escape(c.point)}<div class="ev">{ui.escape(c.evidence)}</div>{ref_chips(c.match_refs, refs)}</li>'
                   for c in items) or "<li>–</li>"


def bullets(items):
    return "".join(f"<li>{ui.escape(x)}</li>" for x in items) or "<li>–</li>"


AGENT_LABEL = {"claude": "Claude", "chatgpt": "ChatGPT", "gemini": "Gemini"}
AGENT_COLOR = {"claude": "#E0703A", "chatgpt": "#6BD58A", "gemini": "#7AA7FF"}


def reviewers(disc):
    if disc and disc.get("version") in (3, 4) and disc.get("positions"):
        return "Panel: " + " + ".join(AGENT_LABEL[a] for a in disc["positions"])
    return "Claude + ChatGPT" if disc and disc.get("review") else "Claude"


def agent_chip(a, text=None):
    c = AGENT_COLOR.get(a, "#B4B8C2")
    return (f'<span class="vs-chip" style="background:{c}22;color:{c};border:1px solid {c}55">'
            f'{ui.escape(text or AGENT_LABEL.get(a, a))}</span>')


def collective_view(disc, match=False):
    """Three-agent panel: who took part, how they voted, and the full negotiation."""
    ag = disc.get("agents", {})
    status_txt = {"ok": "✓", "failed": "✗ failed", "no key": "no key", "dropped in round 2": "✗ dropped",
                  "dropped in round 3": "✗ dropped"}
    chips = "".join(agent_chip(a, f'{AGENT_LABEL[a]} {status_txt.get(v.get("status"), v.get("status"))}')
                    for a, v in ag.items())
    if disc.get("skipped") or disc.get("error"):
        note = disc.get("skipped") or "Panel stopped early (" + disc["error"][:140] + ") - showing Claude's draft."
        st.html(f'<div class="vs-panel" style="padding:12px 16px">{chips}<div class="ev" style="margin-top:6px;color:#9AA0AB">'
                f'{ui.escape(note)}'
                + "".join(f'<br>{AGENT_LABEL[a]}: {ui.escape(v["error"][:140])}' for a, v in ag.items() if v.get("error"))
                + '</div></div>')
        return
    probs = disc.get("ai_probs") or {}
    prob_row = ("".join(f'<div><div class="k" style="color:{AGENT_COLOR[a]}">{AGENT_LABEL[a]}</div>'
                        f'<div class="v">red {p:.0%}</div></div>' for a, p in probs.items())
                + f'<div><div class="k" style="color:#F2F3F5">Panel (median)</div><div class="v">red {disc["ai_median"]:.0%}</div></div>'
                if match and probs else "")
    votes = "".join(f'<li>{ui.escape(v["point"])}<div style="margin-top:3px">'
                    + "".join(agent_chip(a, "✓ " + AGENT_LABEL.get(a, a)) for a in v.get("agree", []) if a in AGENT_LABEL)
                    + "".join(agent_chip(a, "✗ " + AGENT_LABEL.get(a, a)) for a in v.get("disagree", []) if a in AGENT_LABEL)
                    + "</div></li>" for v in disc.get("votes", []))
    disputes = "".join(f'<li><b>{ui.escape(d["issue"])}</b><div class="ev">{ui.escape(d["sides"])}</div></li>'
                       for d in disc.get("still_disagree", []))
    decided = "voted, no moderator" if disc.get("version") == 4 else "negotiated"
    st.html(f'<div class="vs-panel" style="padding:14px 18px"><div class="vs-lbl" style="margin-top:0">Collective intelligence · '
            f'{len(disc.get("positions", {}))} agents {decided}</div>{chips}'
            + (f'<div class="vs-top3">{prob_row}</div>' if prob_row else "")
            + f'<div style="margin-top:8px"><b>How the panel voted</b><ul style="margin:.3rem 0 0">{votes or "<li>–</li>"}</ul></div>'
            + (f'<div style="margin-top:8px"><b style="color:#F5B95A">Still disagree</b><ul style="margin:.3rem 0 0">{disputes}</ul></div>'
               if disputes else '<div class="ev" style="margin-top:6px">No open disagreements.</div>')
            + f'<div class="ev" style="margin-top:6px">{ui.escape(disc.get("rationale", ""))}</div></div>')
    with st.expander("Full negotiation: independent drafts → challenges → resolution"):
        r1 = []
        for a, d in disc.get("drafts", {}).items():
            if match:
                head = f'{d["predicted_winner"].title()} · red {d["red_win_prob"]:.0%} - {d["decisive_factor"]}'
            else:
                head = f'{d["strongest_capability"]["point"]} · Action: {d["recommended_action"]}'
            r1.append(f'<li>{agent_chip(a)} {ui.escape(head)}</li>')
        r2 = []
        for a, p in disc.get("positions", {}).items():
            ch = "".join(f'<div class="ev">→ challenges {ui.escape(AGENT_LABEL.get(c["to"], c["to"]))}: '
                         f'<b>{ui.escape(c["claim"])}</b> - {ui.escape(c["why"])}</div>' for c in p.get("challenges", []))
            co = "".join(f'<div class="ev">↺ concedes: {ui.escape(x)}</div>' for x in p.get("concessions", []))
            rv = (f' · now red {p["red_win_prob"]:.0%}' if match and "red_win_prob" in p else "")
            r2.append(f'<li>{agent_chip(a)} “{ui.escape(p["bottom_line"])}”{rv}{ch}{co}</li>')
        dissent = "".join(f'<li>{agent_chip(a)} {ui.escape(x)}</li>' for a, xs in (disc.get("dissent") or {}).items() for x in xs)
        how = ("There is no moderator: every agent voted on every part of every revised analysis, a part is kept only "
               "with a majority, and the win chance is the median of the agents' final estimates."
               if disc.get("version") == 4 else
               "The final report keeps only claims backed by the data that survived challenge; the win chance is the "
               "median of the agents' revised estimates.")
        st.html(f'<div class="vs-ai" style="margin:0"><h4>① Independent first takes</h4><ul>{"".join(r1)}</ul>'
                f'<h4 style="margin-top:10px">② Negotiation</h4><ul>{"".join(r2)}</ul>'
                f'<h4 style="margin-top:10px">③ Vote · all agents agree on</h4><ul>{bullets(disc.get("consensus", []))}</ul>'
                + (f'<h4 style="margin-top:10px">Dissent</h4><ul>{dissent}</ul>' if dissent else "")
                + f'<div class="ev">Agents first wrote without seeing each other, so no one anchored on the first answer. {how} '
                  f'One outlier can\'t swing the result.</div></div>')


def discussion_view(disc, match=False):
    """Show how the AI agents got to the final version."""
    if not disc:
        return
    if disc.get("version") in (3, 4):
        collective_view(disc, match)
        return
    if disc.get("skipped") or disc.get("error"):
        st.caption("ChatGPT review: " + (disc.get("skipped") or f"unavailable this time ({disc['error'][:120]})")
                   + " Showing Claude's analysis only.")
        return
    rv = disc.get("review") or {}
    verdict_color = {"accepted": "#6BD58A", "partly accepted": "#F5B95A", "rejected": "#FF8A8D"}
    with st.expander(f"How Claude and ChatGPT discussed this · {len(rv.get('pushback', []))} pushback(s), "
                     f"{len(disc.get('still_disagree', []))} open disagreement(s)"):
        if match:
            d = disc["draft"]
            st.html('<div class="vs-top3">'
                    f'<div><div class="k" style="color:#D9A66B">Claude draft</div><div class="v">{d["predicted_winner"].title()} · '
                    f'red {d["red_win_prob"]:.0%}</div></div>'
                    f'<div><div class="k" style="color:#74C69D">ChatGPT</div><div class="v">{rv.get("predicted_winner", "?").title()} · '
                    f'red {rv.get("red_win_prob", 0):.0%}</div></div></div>')
        pb = "".join(f'<li><b>{ui.escape(x["claim"])}</b><div class="ev">{ui.escape(x["why"])}</div></li>'
                     for x in rv.get("pushback", [])) or "<li>No pushback - the draft held up.</li>"
        st.html(f'<div class="vs-ai" style="margin:0">'
                f'<h4 style="color:#74C69D">① ChatGPT reviewed Claude\'s draft</h4>'
                f'<div class="ev" style="margin-bottom:6px">“{ui.escape(rv.get("bottom_line", ""))}”</div>'
                f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px">'
                f'<div><b>Agrees</b><ul>{bullets(rv.get("agrees", []))}</ul></div>'
                f'<div><b>Pushes back</b><ul>{pb}</ul></div>'
                f'<div><b>Says Claude missed</b><ul>{bullets(rv.get("missed", []))}</ul></div></div>'
                f'<h4 style="color:#D9A66B;margin-top:10px">② Claude responded</h4><ul>'
                + "".join(f'<li><span style="color:{verdict_color.get(x["verdict"], "#B4B8C2")};font-weight:700">'
                          f'{x["verdict"].title()}</span> - {ui.escape(x["point"])}<div class="ev">{ui.escape(x["reason"])}</div></li>'
                          for x in disc.get("replies", [])) + '</ul>'
                f'<h4 style="margin-top:10px">③ Where they landed</h4>'
                f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px">'
                f'<div><b style="color:#6BD58A">Agree on</b><ul>{bullets(disc.get("consensus", []))}</ul></div>'
                f'<div><b style="color:#F5B95A">Still disagree</b><ul>{bullets(disc.get("still_disagree", [])) if disc.get("still_disagree") else "<li>Nothing - full agreement.</li>"}</ul></div></div>'
                f'<div class="ev" style="margin-top:8px">Models: {ui.escape(disc.get("claude_model", ""))} (Claude) · '
                f'{ui.escape(disc.get("chatgpt_model") or "")} (ChatGPT). The report above is Claude\'s final version after this exchange.</div></div>')


@st.dialog("Team profile", width="medium", position="right")
def team_dialog(number):
    t = by_number[number]
    r = R[t["id"]]
    chips = ui.tier_chip(r.tier)
    chips += ui.chip(analysis.rank_text(r, N_RATED, len(teams)) if r.rank else f"Not rated yet · {len(teams)} registered")
    chips += ui.chip(r.profile)
    chips += (f'<span class="vs-chip" title="{CONF_HELP}">Data confidence: {r.confidence} '
              f'({r.stats["matches"]} matches) ⓘ</span>')
    if t["id"] == MY_TEAM_ID:
        chips += ui.chip("you", ui.YOU, "#fff")
    org = " · ".join(x for x in (t["organization"], t["location"]) if x)
    if r.live:
        (st.success if r.live["good"] else st.error)(
            f'**Live: {r.live["label"]}** — "{r.live["text"]}"  ·  manage it in the Live updates tab', icon="📣")
    badges = "".join(ui.trait_chip(*x) for x in T[t["id"]])
    auton = traits.auton_summary(conn, t["id"], EID)
    auton_line = " · ".join(f"{side} side {w}/{n}" for side, (w, n) in auton.items())
    st.html(f'<div class="vs-panel"><div class="vs-ring-wrap">{ui.ring(r.overall)}<div>'
            f'<div class="name">{ui.escape(t["number"])} · {ui.escape(t["name"] or "")}</div>'
            f'<div class="org">{ui.escape(org)}</div>{chips}</div></div>'
            + (f'<div style="margin-top:14px">{badges}</div>' if badges else "")
            + (f'<div class="org" style="margin-top:6px">Scouted auton: {ui.escape(auton_line)}</div>' if auton_line else "")
            + "</div>")

    show_you = me is not None and t["id"] != MY_TEAM_ID
    rows = "".join(ui.bar_row(label, f"{w:.0%}", r.pct.get(k), r.contrib.get(k),
                              me.pct.get(k) if show_you else None)
                   for k, label, w, _ in ratings.COMPONENTS)
    def note_row(label, text, value, small):
        return (f'<div class="vs-row"><div class="lab">{label}</div><div class="w">–</div>'
                f'<div style="color:#9AA0AB;font-size:.85rem">{text}</div>'
                f'<div class="c">{value}<small>{small}</small></div></div>')

    base = r.overall_base if r.overall_base is not None else r.overall
    if base is not None and r.overall_raw is not None and round(base - r.overall_raw):
        rows += note_row("Track record", f'{r.stats["events"]} competition(s), {r.stats["matches"]} matches: keeps '
                         f'{r.reliability:.0%} of its distance from 50. One good event counts less than many.',
                         f"{base - r.overall_raw:+.0f}", "adjust")
    if r.live and r.overall_base is not None:
        rows += note_row("Live update", ui.escape(f'{r.live["label"]}: "{r.live["text"][:70]}"'),
                         ("out" if r.live["effect"] == "out" else f'{r.overall - r.overall_base:+.0f}'),
                         f'×{r.live["mult"]}')
    rows += ui.bar_row("Momentum", "–", r.pct.get("momentum"), None, me.pct.get("momentum") if show_you else None)
    legend = '<span><i style="background:#5B8DEF"></i>Percentile vs this event (bar)</span>'
    if show_you:
        legend += f'<span><i style="background:{ui.YOU}"></i>{ui.escape(MY_TEAM_NUMBER)} (you)</span>'
    legend += "<span>Center line = middle of the field</span>"
    total = (f'<div style="display:flex;justify-content:space-between;align-items:center;border-top:1px solid #2A2D35;'
             f'margin-top:10px;padding-top:14px"><b style="font-size:1.15rem">Overall score</b>'
             f'<span style="font-size:1.15rem;font-weight:700">{r.overall if r.overall is not None else "–"} / 100 '
             f'{ui.tier_chip(r.tier)}</span></div>')
    st.html(f'<div class="vs-panel"><div style="font-size:1.1rem;font-weight:700;margin-bottom:6px">How the overall score is built '
            f'<span style="color:#9AA0AB;font-weight:400;font-size:.92rem">verified match data, compared with this event\'s teams</span></div>'
            f'{rows}{total}<div class="vs-legend">{legend}</div></div>')

    def sw(title, color, arrow, items, empty):
        li = "".join(f'<li><span style="color:{color}">{arrow}</span> {ui.escape(x)}</li>' for x in items) or \
             f'<li style="color:#9AA0AB">{empty}</li>'
        return f'<div class="vs-sw"><h4><span style="color:{color}">{arrow}</span> {title}</h4><div class="src">From match data</div><ul>{li}</ul></div>'

    st.html('<div class="vs-panel" style="display:grid;grid-template-columns:1fr 1fr;gap:24px">'
            + sw("Strengths", "#46A758", "▲", r.strengths, "Nothing stands out yet.")
            + sw("Weaknesses", "#E5484D", "▼", r.weaknesses, "No major weak spots relative to the field.") + "</div>")

    s = r.stats
    k = st.columns(5)
    k[0].metric("OPR", f"{s['opr']:.1f}" if s["opr"] is not None else "–")
    k[1].metric("Elo", s["elo"])
    k[2].metric("CCWM", f"{s['ccwm']:+.1f}" if s["ccwm"] is not None else "–")
    k[3].metric("Season matches", s["matches"])
    k[4].metric("Event rank", t["rank"] or "–", help="Official rank at this event, once qualifiers start")

    ai_slot = st.container()  # filled at the end, so the rest of the panel shows while Claude writes


    def ai_section():
        saved, diff = analysis.saved_team_report(conn, EID, r, me)
        rep = saved["report"] if saved else None
        disc = saved["discussion"] if saved else None
        when = saved and datetime.fromisoformat(saved["created_at"]).astimezone().strftime("%b %-d, %-I:%M %p")
        fail_key = ("ai_fail", EID, number, st.session_state.data_version)
        forced = st.session_state.pop(f"ai_force_{number}", False)
        if (saved is None or diff or forced) and fail_key not in st.session_state:
            msg = ("Stats changed a lot since the saved report (" + "; ".join(diff) + ") - updating it…" if diff
                   else "Writing a scouting report on this team…")
            with st.status(msg, expanded=False) as stat:
                try:
                    rep, disc = analysis.new_team_report(conn, EID, r, len(teams), me, N_RATED,
                                                         on_stage=lambda m: stat.update(label=m))
                    when, diff = "just now", []
                    stat.update(label="Report ready", state="complete")
                except Exception as e:
                    st.session_state[fail_key] = f"{type(e).__name__}: {e}"
                    stat.update(label="AI report unavailable", state="error")
        if fail_key in st.session_state:
            st.error(f"AI report unavailable ({st.session_state[fail_key]})" + (" - showing the last saved one." if rep else ""))
        if rep:
            c1, c2 = st.columns([5, 1.2], vertical_alignment="center")
            c1.caption(f"AI report from {when} · updates automatically when the stats change a lot")
            if c2.button("↻ Regenerate", key=f"ai_{number}", type="tertiary"):
                st.session_state[f"ai_force_{number}"] = True
                st.session_state.pop(fail_key, None)
                st.rerun(scope="fragment")
        if rep:
            refs = ref_index([r] + ([me] if me is not None and me.team_id != r.team_id else []),
                             {r.team_id: "", **({me.team_id: me.number + "-"} if me is not None and me.team_id != r.team_id else {})})
            st.html(f'<div class="vs-ai"><div class="tag">AI scouting report · {reviewers(disc)} · '
                    f'<span title="{AI_CERT_HELP}">AI certainty: {rep.certainty} ⓘ</span></div>'
                    + top3(("Strongest capability", "#6BD58A", rep.strongest_capability.point,
                            rep.strongest_capability.evidence, ref_chips(rep.strongest_capability.match_refs, refs)),
                           ("Biggest uncertainty", "#F5B95A", rep.biggest_uncertainty.point,
                            rep.biggest_uncertainty.evidence, ref_chips(rep.biggest_uncertainty.match_refs, refs)),
                           ("Recommended action", "#9DBDF8", rep.recommended_action, "", ""))
                    + f'<p style="margin:0">{ui.escape(rep.summary)}</p>'
                    f'<div class="ev" style="margin-top:8px">{CONF_HELP} {AI_CERT_HELP}</div></div>')
            discussion_view(disc)
            with st.expander("Full report: strengths, weaknesses, tactics"):
                st.html(f'<div class="vs-ai" style="margin:0"><div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:18px">'
                        f'<div><h4 style="color:#46A758">▲ Strengths</h4><ul>{claims(rep.strengths, refs)}</ul></div>'
                        f'<div><h4 style="color:#E5484D">▼ Weaknesses</h4><ul>{claims(rep.weaknesses, refs)}</ul></div>'
                        f'<div><h4>⚔️ How to beat them</h4><ul>{bullets(rep.how_to_beat)}</ul></div>'
                        f'<div><h4>🤝 As a partner</h4><ul>{bullets(rep.as_partner)}</ul></div></div>'
                        f'<h4>🔎 Scout next</h4><ul>{bullets(rep.scout_next)}</ul>'
                        f'<div class="ev">Match chips (M1, M2…) point to rows in the Recent matches table below; hover for details.</div></div>')

    hist = recent_matches(t["id"])
    if hist:
        st.markdown("**Recent matches (verified)**")
        st.dataframe(pd.DataFrame(hist), hide_index=True, width="stretch")

    st.html('<div class="vs-scout-h">Scout this team<span>tap what you saw, then save</span></div>')
    nonce = st.session_state.setdefault(f"scout_n_{number}", 0)  # bumped after saving to clear the inputs
    k = f"{number}_{nonce}"
    flash = st.session_state.pop(f"scout_msg_{number}", None)
    with st.container(border=True):
        st.html('<div class="vs-lbl">Strengths you saw</div>')
        pos = st.pills("Strengths you saw", scouting.POS_TAGS, selection_mode="multi", label_visibility="collapsed", key=f"pos_{k}")
        st.html('<div class="vs-lbl">Weaknesses you saw</div>')
        neg = st.pills("Weaknesses you saw", scouting.NEG_TAGS, selection_mode="multi", label_visibility="collapsed", key=f"neg_{k}")
        st.html('<div class="vs-lbl">Autonomous attempts</div>')
        attempts = {}
        for col, side in zip(st.columns(2), ("left", "right")):
            with col.container(border=True):
                a, b, c, d = st.columns([1.1, 1, 0.7, 1], vertical_alignment="center")
                a.markdown(f"**{side.title()} side**")
                w = b.number_input(f"{side} worked", 0, 50, 0, label_visibility="collapsed", key=f"aw_{side}_{k}")
                c.caption("worked of")
                n = d.number_input(f"{side} tried", 0, 50, 0, label_visibility="collapsed", key=f"an_{side}_{k}")
                attempts[side] = [int(w), int(n)]
        note_slot = st.container()  # the text box sits above the mic, but is drawn after it so dictation lands in it
        said = mic.mic(key=f"mic_{k}")
        if said:
            cur = st.session_state.get(f"note_{k}", "").strip()
            st.session_state[f"note_{k}"] = f"{cur} {said}".strip()
        text = note_slot.text_area("Anything else?", label_visibility="collapsed", key=f"note_{k}",
                                  placeholder='Anything else? Type, or tap the mic below - plain English works, e.g. '
                                              '"fast robot and driver, auton works 9/10 on left side"')
        author = st.text_input("Scout name (optional)", placeholder="Scout name (optional)",
                               label_visibility="collapsed", key=f"scout_name_{number}")
        if flash:
            st.success(flash)
        if st.button("Save", type="primary", width="stretch", key=f"save_{k}"):
            tapped = bool(pos or neg or any(n_ for _, n_ in attempts.values()))
            bad = [side for side, (w_, n_) in attempts.items() if w_ > n_]
            if bad:
                st.error(f"Auton {', '.join(bad)} side: 'worked' can't be more than attempts.")
            elif not (tapped or text.strip()):
                st.warning("Tap something you saw, write a note, or use the mic first.")
            else:
                try:
                    msgs = []
                    if tapped:
                        res = scouting.record_tags(conn, EID, number, pos, neg, attempts, author or None)
                        msgs.append(f"Saved: {res['text']}")
                        if res["conflicts"]:
                            msgs.append("Disagrees with earlier notes, so both are kept and marked conflicting.")
                    if text.strip():
                        with st.spinner("Reading your note…"):
                            for s_ in scouting.record(conn, text, EID, number, author or None):
                                msgs.append(f"{s_['team']}: {s_['error']}" if "error" in s_
                                            else f"Note saved ({s_['category']}, read by {s_['extracted_by']}).")
                    bump()
                    st.session_state[f"scout_n_{number}"] = nonce + 1
                    st.session_state[f"scout_msg_{number}"] = " ".join(msgs)
                    st.rerun(scope="fragment")
                except Exception as e:
                    st.error(str(e))

    st.markdown("**Scouting history (subjective)**")
    for o in db.observations_for(conn, t["id"], EID, include_inactive=True):
        badge = {"active": "🟢", "conflicting": "🟠", "superseded": "⚪"}[o["status"]]
        with st.container(border=True):
            st.markdown(f"{badge} **{o['category'] or 'note'}** · conf {o['confidence']} · {o['created_at'][:16].replace('T', ' ')}"
                        + (f" · {o['match_ref']}" if o["match_ref"] else "") + (f" · {o['author']}" if o["author"] else ""))
            st.write(o["raw_text"])
            b1, b2, _ = st.columns([1.3, 1, 3])
            if o["status"] != "superseded" and b1.button("Mark outdated", key=f"sup{o['id']}"):
                db.set_observation_status(conn, o["id"], "superseded"); bump(); st.rerun(scope="fragment")
            if b2.button("🗑 Delete", key=f"del{o['id']}"):
                st.session_state.notes_undo = scouting.delete_notes(conn, [o["id"]], EID)
                bump(); st.rerun(scope="fragment")

    with ai_slot:
        ai_section()


if st.session_state.get("goto_tab") in TABS:  # e.g. "Plan this matchup" jumps to the copilot
    st.session_state["main_tab"] = st.session_state.pop("goto_tab")
sec = st.session_state.get("main_tab") or TABS[0]
st.html(f'<div class="vs-mtitle">{ui.escape(sec)}</div>')  # phones: which section the menu opened
tab_teams, tab_live, tab_notes, tab_picks, tab_pred, tab_copilot = st.tabs(TABS, key="main_tab", on_change="rerun")


# --- teams ---------------------------------------------------------------------
def fmt(v, spec):
    return None if v is None else float(format(v, spec))


with tab_teams:
    c1, c2 = st.columns([2, 2], vertical_alignment="center")
    q = c1.text_input("Filter", placeholder="Filter by number, name, or school", label_visibility="collapsed")
    open_no = c2.container(key="desk_only_picker").selectbox("Open team profile", numbers, index=None, key="open_team", label_visibility="collapsed",
                           placeholder="Open a team profile…",
                           format_func=lambda n: f"{n} · {by_number[n]['name'] or ''}")
    if open_no and open_no != st.session_state.get("opened_pick"):
        st.session_state.opened_pick = open_no
        team_dialog(open_no)
    elif not open_no:
        st.session_state.opened_pick = None
    with st.expander("How scores work"):
        st.html('<div style="line-height:1.6"><b>Overall (0–100)</b> is a weighted average of each team\'s percentile ranks '
                'against the teams at <i>this</i> event: scoring 20%, experience 20%, match impact 15%, then autonomous, '
                'driving, consistency and win rate 10% each, defense 5%. 50 is the middle of this field.<br>'
                '<b>Track record:</b> the score is pulled toward 50 until a team has proven itself over several '
                'competitions - doing well at one event counts far less than doing well at ten.<br>'
                '<b>Rank</b> in a team\'s profile counts only rated teams (those with match or skills data); the '
                'official event rank appears once qualifiers start.<br>'
                f'<b>{CONF_HELP.split(" = ")[0]}</b> = {CONF_HELP.split(" = ")[1]}<br>'
                f'<b>{AI_CERT_HELP.split(" = ")[0]}</b> = {AI_CERT_HELP.split(" = ")[1]}<br>'
                '<b>Live updates</b> (e.g. a robot broken today) scale the score until removed or expired. '
                'Missing data counts as average.<div style="margin-top:8px">'
                + "".join(ui.tier_chip(t) for t in ("Elite", "Strong", "Solid", "Developing", "Building")) + "</div></div>")
    waiting = len(teams) - N_RATED
    st.caption((f"⏳ {waiting} of {len(teams)} teams have no matches yet this season - Unrated until results come in. "
                if waiting else "") + "Tap a team for its profile and AI report.")
    data = []
    for t in teams:
        r = R[t["id"]]
        s = r.stats
        data.append({
            "Rank": f"#{t['rank']}" if t["rank"] else "–", "Team": t["number"] + (" ★" if t["id"] == MY_TEAM_ID else ""), "Name": t["name"] or "",
            "Overall": r.overall, "Tier": r.tier, "Profile": r.profile,
            "Strengths": "  ".join("▲ " + label for label, good, _ in T[t["id"]] if good),
            "Weaknesses": "  ".join("▼ " + label for label, good, _ in T[t["id"]] if not good),
            "OPR": fmt(s["opr"], ".1f"), "Elo": s["elo"], "CCWM": fmt(s["ccwm"], ".1f"),
            "Auton": s["auton"], "Driver": s["driver"], "Swing ±": fmt(s["swing"], ".1f"),
            "Win %": None if s["winrate"] is None else round(s["winrate"] * 100),
            "Matches": s["matches"], "Events": s["events"], "Notes": notes.get(t["id"], 0),
            "_school": t["organization"] or "",
        })
    df = pd.DataFrame(data).sort_values(["Overall", "Team"], ascending=[False, True], na_position="last")
    for col in ("Overall", "Auton", "Driver", "Win %"):  # nullable ints show blank, not "None"
        df[col] = df[col].astype("Int64")
    if q:
        m = df["Team"].str.contains(q, case=False) | df["Name"].str.contains(q, case=False) | df["_school"].str.contains(q, case=False)
        df = df[m]
    df = df.drop(columns="_school").reset_index(drop=True)

    def tier_css(v):
        bg, fg = ui.TIER_STYLE.get(v, ui.TIER_STYLE["Unrated"])
        return f"background-color:{bg};color:{fg};font-weight:600"

    styled = df.style.map(tier_css, subset=["Tier"]).map(lambda v: "color:#6BD58A", subset=["Strengths"]).map(lambda v: "color:#FF8A8D", subset=["Weaknesses"]).apply(
        lambda row: ["color:#E0703A;font-weight:700" if "★" in row["Team"] else "" for _ in row], axis=1, subset=["Team"]
    )
    desk = st.container(key="desk_table")
    sel = desk.dataframe(
        styled, hide_index=True, width="stretch", height=min(38 * len(df) + 40, 720),
        on_select="rerun", selection_mode="single-cell", key="team_table",
        column_config={
            "Rank": st.column_config.TextColumn(help="Official event rank (once qualifiers start)", width="small"),
            "Strengths": st.column_config.TextColumn(width="medium", help="From stats, track record, scouting taps and live feedback"),
            "Weaknesses": st.column_config.TextColumn(width="medium", help="From stats, track record, scouting taps and live feedback"),
            "Overall": st.column_config.ProgressColumn("Overall", min_value=0, max_value=100, format="%d", width="medium",
                                                      help="Weighted percentile vs this event's teams"),
            "OPR": st.column_config.NumberColumn(help="Estimated points contributed per match", format="%.1f"),
            "CCWM": st.column_config.NumberColumn(help="OPR minus points allowed (DPR): effect on the margin", format="%+.1f"),
            "Auton": st.column_config.NumberColumn(help="Best programming skills score this season"),
            "Driver": st.column_config.NumberColumn(help="Best driver skills score this season"),
            "Swing ±": st.column_config.NumberColumn(help="Std. dev. of its alliance scores (lower = steadier)", format="%.1f"),
            "Win %": st.column_config.NumberColumn(format="%d%%"),
        })
    with st.container(key="mobile_cards"):  # shown instead of the table on phones (CSS media query)
        for _, row in df.iterrows():
            no = row["Team"].replace(" ★", "")
            rt = R[by_number[no]["id"]]
            good = [lab for lab, g, _ in T[rt.team_id] if g and not lab.startswith("Experienced")]
            key_str = (good or [lab for lab, g, _ in T[rt.team_id] if g] or rt.strengths or [""])[0]
            with st.container(border=True):
                you = f' <span style="color:{ui.YOU}">★ you</span>' if rt.team_id == MY_TEAM_ID else ""
                score = "–" if rt.overall is None else rt.overall
                meta = (f'{ui.tier_chip(rt.tier)}<span class="vs-chip">Data: {rt.confidence}</span>'
                        + (f'<span style="color:#6BD58A">▲ {ui.escape(key_str)}</span>' if key_str else
                           '<span style="color:#8B909B">Waiting for first matches</span>' if rt.overall is None else ""))
                st.html(f'<div class="vs-tcard"><div><div class="num">{ui.escape(no)}{you}</div>'
                        f'<div class="nm">{ui.escape(row["Name"])}</div></div>'
                        f'<div class="sc" style="color:{ui.score_color(rt.overall or 0)}">{score}</div>'
                        f'<div class="meta">{meta}</div></div>')
                if st.button("View profile →", key=f"m_open_{no}", width="stretch"):
                    team_dialog(no)
    picked = [c[0] for c in sel.selection.cells]
    key = (picked[0], q, tuple(sel.selection.cells)) if picked else None
    if key and key != st.session_state.get("opened_row"):
        st.session_state.opened_row = key
        team_dialog(df.iloc[picked[0]]["Team"].replace(" ★", ""))
    elif not key:
        st.session_state.opened_row = None
        linked = st.query_params.get("team", "").upper()  # ?team=1234A opens that profile
        if linked in by_number and st.session_state.get("opened_link") != linked:
            st.session_state.opened_link = linked
            team_dialog(linked)


# --- live feedback -------------------------------------------------------------
EFFECT_OPTS = list(feedback.EFFECTS)
effect_label = lambda e: feedback.EFFECTS[e][0] + (f"  (×{feedback.EFFECTS[e][1]})" if feedback.EFFECTS[e][1] not in (0, 1) else "")



def impact(number, effect):
    """What applying `effect` to a team would change, for the preview."""
    rt = R[by_number[number]["id"]]
    label, mult, _ = feedback.EFFECTS[effect]
    base = rt.overall_base if rt.overall_base is not None else rt.overall
    parts = []
    if effect == "out":
        parts.append("removed from pick lists; predictions treat the robot as absent")
    elif base is None:
        parts.append(f"no rating yet - expected points ×{mult}")
    else:
        parts.append(f"overall {rt.overall} → {round(base * mult)}")
    if effect != "out" and rt.stats["opr"] is not None and mult != 1:
        parts.append(f"expected points {rt.stats['opr']:.1f} → {rt.stats['opr'] * mult:.1f} per match")
    if mult == 1 and effect != "out":
        parts.append("clears any earlier problem - back to normal ratings")
    if rt.live:
        parts.append(f"replaces current status “{rt.live['label']}”")
    return " · ".join(parts)


with tab_live:
    st.html('<div class="vs-scout-h">Live updates<span>what\'s happening right now - ratings, predictions and picks adjust</span></div>')
    fn = st.session_state.setdefault("fb_n", 0)
    pending = st.session_state.get("fb_pending")
    if not pending:
        with st.container(border=True):
            st.html('<div class="vs-lbl" style="margin-top:0">Quick update</div>')
            q1, q2 = st.columns([2, 1])
            q_team = q1.selectbox("Team", numbers, index=None, key=f"fb_team_{fn}", placeholder="Pick a team…",
                                  format_func=lambda n: f"{n} · {by_number[n]['name'] or ''}", label_visibility="collapsed")
            q_dur = q2.selectbox("Applies", list(feedback.DURATIONS), key=f"fb_qdur_{fn}", format_func=feedback.DURATIONS.get,
                                 label_visibility="collapsed")
            cols = st.columns(3)
            for i, (lab, eff) in enumerate(feedback.QUICK):
                if cols[i % 3].button(lab, key=f"fb_q_{eff}_{fn}", width="stretch", disabled=not q_team):
                    st.session_state.fb_pending = [dict(team=q_team, effect=eff, duration=q_dur,
                                                        text=f"{q_team}: {lab}", summary=lab)]
                    st.rerun()
            st.html('<div class="vs-lbl">Or describe it</div>')
            txt_slot = st.container()
            said = mic.mic(key=f"fb_mic_{fn}")
            if said:
                st.session_state[f"fb_text_{fn}"] = f'{st.session_state.get(f"fb_text_{fn}", "")} {said}'.strip()
            fb_text = txt_slot.text_area("Update", key=f"fb_text_{fn}", label_visibility="collapsed", height=80,
                                        placeholder='e.g. "80001C robot is broken today" · "295A fixed their intake"')
            if st.button("Read update", width="stretch", disabled=not fb_text.strip(),
                         help="Claude reads it and shows you exactly what it will change before anything is applied"):
                with st.spinner("Reading the update…"):
                    parsed = feedback.parse(fb_text.strip(), numbers)
                if parsed:
                    st.session_state.fb_pending = [dict(team=u.team_number, effect=u.effect, duration=u.duration,
                                                        text=fb_text.strip(), summary=u.summary) for u in parsed]
                    st.rerun()
                else:
                    st.warning("Couldn't find a team from this event in that update - include the team number, e.g. 80001C.")
    else:
        with st.container(border=True):
            st.html('<div class="vs-lbl" style="margin-top:0;color:#F5B95A">Check before applying</div>')
            for i, u in enumerate(pending):
                c1, c2, c3 = st.columns([1, 1.3, 1.1], vertical_alignment="center")
                c1.markdown(f"**{u['team']}**  \n<span style='color:#9AA0AB;font-size:.85rem'>“{ui.escape(u['text'][:60])}”</span>",
                            unsafe_allow_html=True)
                u["effect"] = c2.selectbox("Effect", EFFECT_OPTS, index=EFFECT_OPTS.index(u["effect"]), key=f"pend_eff_{fn}_{i}",
                                           format_func=effect_label, label_visibility="collapsed")
                u["duration"] = c3.selectbox("Applies", list(feedback.DURATIONS), key=f"pend_dur_{fn}_{i}",
                                             index=list(feedback.DURATIONS).index(u["duration"]),
                                             format_func=feedback.DURATIONS.get, label_visibility="collapsed")
                st.caption(f"→ {impact(u['team'], u['effect'])}")
            fb_author = st.text_input("Your name", key="fb_author", placeholder="Your name (optional)", label_visibility="collapsed")
            a1, a2 = st.columns(2)
            if a1.button(f"Apply {'update' if len(pending) == 1 else f'{len(pending)} updates'}", type="primary", width="stretch"):
                for u in pending:
                    feedback.insert(conn, EID, u["team"], u["effect"], u["duration"], u["text"], u["summary"], fb_author or None)
                st.session_state["fb_msg"] = " · ".join(f"{u['team']}: {feedback.EFFECTS[u['effect']][0]}" for u in pending)
                st.session_state.fb_pending = None
                st.session_state.fb_n = fn + 1
                bump(); st.rerun()
            if a2.button("Cancel", width="stretch"):
                st.session_state.fb_pending = None; st.rerun()
    if msg := st.session_state.pop("fb_msg", None):
        st.success(f"Applied: {msg}")

    items = feedback.list_for_event(conn, EID)
    active = [x for x in items if x["status"] == "active"]
    removed = [x for x in items if x["status"] == "removed"]
    if not items:
        st.caption("No updates yet. Post one above when something changes at the event.")

    def fb_card(x):
        label, mult, good = feedback.EFFECTS[x["effect"]]
        color = "#6BD58A" if good else "#FF8A8D" if good is False else "#B4B8C2"
        if x["status"] == "removed":
            state = "removed"
        elif x["expired"]:
            state = "expired"
        elif x["in_effect"]:
            state = "in effect"
        elif x["effect"] == "info":
            state = "note"
        else:
            state = "overridden by a newer update"
        when = datetime.fromisoformat(x["created_at"]).astimezone().strftime("%b %-d, %-I:%M %p")
        edited = f' · edited {len(x["revisions"])}×' if x["revisions"] else ""
        dim = "opacity:.55;" if state in ("removed", "expired", "overridden by a newer update") else ""
        st.html(f'<div class="vs-panel" style="padding:14px 18px;margin-bottom:6px;{dim}">'
                f'<div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">'
                f'<b style="font-size:1.1rem">{ui.escape(x["number"])}</b>'
                f'<span class="vs-chip" style="background:{color}22;color:{color};border:1px solid {color}55">{label}'
                f'{f" ×{mult}" if mult not in (0, 1) else ""}</span>{ui.chip(state)}'
                f'<span style="color:#8B909B;font-size:.85rem">{feedback.DURATIONS[x["duration"]]} · {when}'
                f'{" · " + ui.escape(x["author"]) if x["author"] else ""}{edited}</span></div>'
                f'<div style="margin-top:6px">“{ui.escape(x["raw_text"])}”</div></div>')

    for x in active:
        fb_card(x)
        b1, b2, _ = st.columns([1, 1, 6])
        if b1.button("Edit", key=f"fb_edit_{x['id']}"):
            st.session_state.fb_editing = x["id"]
        if b2.button("Remove", key=f"fb_rm_{x['id']}"):
            feedback.set_status(conn, x["id"], "removed"); bump(); st.rerun()
        if st.session_state.get("fb_editing") == x["id"]:
            with st.form(f"fb_form_{x['id']}"):
                new_text = st.text_area("Update text", x["raw_text"])
                e1, e2 = st.columns(2)
                new_eff = e1.selectbox("Effect", EFFECT_OPTS, index=EFFECT_OPTS.index(x["effect"]), format_func=effect_label)
                new_dur = e2.selectbox("Applies", list(feedback.DURATIONS), index=list(feedback.DURATIONS).index(x["duration"]),
                                       format_func=feedback.DURATIONS.get)
                s1, s2, _ = st.columns([1, 1, 4])
                if s1.form_submit_button("Save", type="primary"):
                    feedback.revise(conn, x["id"], new_text.strip(), new_eff, new_dur)
                    st.session_state.fb_editing = None; bump(); st.rerun()
                if s2.form_submit_button("Cancel"):
                    st.session_state.fb_editing = None; st.rerun()
            if x["revisions"]:
                with st.expander(f"Revision history ({len(x['revisions'])})"):
                    for rv in reversed(x["revisions"]):
                        at = datetime.fromisoformat(rv["updated_at"]).astimezone().strftime("%b %-d, %-I:%M %p")
                        st.caption(f'{at} · {feedback.EFFECTS[rv["effect"]][0]} · {feedback.DURATIONS[rv["duration"]]} — “{rv["raw_text"]}”')
    if removed:
        with st.expander(f"Removed ({len(removed)})"):
            for x in removed:
                fb_card(x)
                if st.button("Restore", key=f"fb_restore_{x['id']}"):
                    feedback.set_status(conn, x["id"], "active"); bump(); st.rerun()
    st.caption("The newest update for a team is the one that counts, so “fixed now” replaces “broken”. Effects scale the "
               "team's overall score and expected points (match stats are never changed); “Not competing” drops the team "
               "from pick lists. “Today only” updates expire at midnight.")


# --- scouting notes ------------------------------------------------------------
with tab_notes:
    all_notes = db.rows(conn, """SELECT o.*, t.number FROM scouting_observations o JOIN teams t ON t.id = o.team_id
                                 WHERE o.event_id = ? OR (o.event_id IS NULL AND o.team_id IN
                                       (SELECT team_id FROM event_teams WHERE event_id = ?))
                                 ORDER BY o.created_at DESC, o.id DESC""", (EID, EID))
    st.html(f'<div class="vs-scout-h">Scouting notes<span>{len(all_notes)} at this event - review, clean up, delete</span></div>')
    undo = st.session_state.get("notes_undo")
    if undo:
        u1, u2 = st.columns([4, 1], vertical_alignment="center")
        u1.success(f"Deleted {len(undo)} note{'s' if len(undo) != 1 else ''}.")
        if u2.button("↶ Undo", width="stretch", key="notes_undo_btn"):
            db.restore_observations(conn, undo)
            for tid in {u["team_id"] for u in undo}:
                scouting.recheck_conflicts(conn, tid, EID)
            st.session_state.notes_undo = None; bump(); st.rerun()
    if not all_notes:
        st.caption("No scouting notes yet. Open a team and use “Scout this team” to add some.")
    else:
        f1, f2, f3 = st.columns([1, 1, 1.4])
        teams_with = sorted({n["number"] for n in all_notes})
        f_team = f1.selectbox("Team", ["All teams"] + teams_with, key="nf_team", label_visibility="collapsed")
        scouts = sorted({n["author"] for n in all_notes if n["author"]})
        f_author = f2.selectbox("Scout", ["All scouts"] + scouts + (["(no name)"] if any(not n["author"] for n in all_notes) else []),
                                key="nf_author", label_visibility="collapsed")
        f_q = f3.text_input("Search", key="nf_q", placeholder="Search note text", label_visibility="collapsed").strip().lower()
        shown = [n for n in all_notes
                 if (f_team == "All teams" or n["number"] == f_team)
                 and (f_author == "All scouts" or (n["author"] or "(no name)") == f_author)
                 and (not f_q or f_q in n["raw_text"].lower())]
        ids = [n["id"] for n in shown]
        picked = [i for i in ids if st.session_state.get(f"nsel_{i}")]
        a1, a2, a3 = st.columns([1.2, 1.2, 2], vertical_alignment="center")
        if a1.button("Select all shown", width="stretch", disabled=not shown):
            for i in ids:
                st.session_state[f"nsel_{i}"] = True
            st.rerun()
        if a2.button("Clear selection", width="stretch", disabled=not picked):
            for i in ids:
                st.session_state[f"nsel_{i}"] = False
            st.rerun()
        if a3.button(f"🗑 Delete selected ({len(picked)})", type="primary", width="stretch", disabled=not picked):
            st.session_state.notes_undo = scouting.delete_notes(conn, picked, EID)
            for i in picked:
                st.session_state.pop(f"nsel_{i}", None)
            bump(); st.rerun()
        st.caption(f"Showing {len(shown)} of {len(all_notes)}. Deleting a note also updates ratings, badges and predictions.")
        status_style = {"active": ("#6BD58A", "in use"), "conflicting": ("#F5B95A", "conflicts with another note"),
                        "superseded": ("#8B909B", "marked outdated")}
        for n in shown:
            col, label = status_style.get(n["status"], ("#B4B8C2", n["status"]))
            when = datetime.fromisoformat(n["created_at"]).astimezone().strftime("%b %-d, %-I:%M %p")
            kind = "tapped" if "tapped" in (n["source"] or "") else "typed / spoken"
            with st.container(border=True):
                c1, c2, c3 = st.columns([0.35, 6, 1], vertical_alignment="center")
                c1.checkbox("Select", key=f"nsel_{n['id']}", label_visibility="collapsed")
                c2.html(f'<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:.85rem;color:#9AA0AB">'
                        f'<b style="color:#F2F3F5;font-size:1rem">{ui.escape(n["number"])}</b>'
                        f'<span style="color:{col}">● {label}</span><span>{ui.escape(n["category"] or "note")} · {kind}</span>'
                        f'<span>{when}{" · " + ui.escape(n["author"]) if n["author"] else ""}</span></div>'
                        f'<div style="margin-top:4px">{ui.escape(n["raw_text"])}</div>')
                if c3.button("🗑", key=f"ndel_{n['id']}", help="Delete this note", width="stretch"):
                    st.session_state.notes_undo = scouting.delete_notes(conn, [n["id"]], EID)
                    bump(); st.rerun()


# --- alliance picks ------------------------------------------------------------
with tab_picks:
    default = numbers.index(MY_TEAM_NUMBER) if MY_TEAM_NUMBER in numbers else 0
    base_no = st.selectbox("Picking for", numbers, index=default, key="pick_for")
    base = R[by_number[base_no]["id"]]
    if base.overall is None:
        st.info(f"{base_no} has no matches or skills runs this season yet, so picks are ranked on partner strength alone. "
                "They'll start covering your weak spots once you have results.")
    else:
        weak = [label for k, label, *_ in ratings.COMPONENTS if base.pct.get(k, 50) < 40]
        st.caption(f"{base_no}: {base.overall} overall · weaker areas: {', '.join(weak) or 'none'}")
    picks = ratings.pick_list(R, base.team_id, model)
    if not picks:
        st.info("No rated teams at this event yet.")
    for i, p in enumerate(picks[:10], 1):
        r = p["rating"]
        why = r.strengths[:2] + ([f"Covers your {', '.join(p['covers']).lower()}"] if p["covers"] else [])
        badges = "".join(ui.trait_chip(*x) for x in T[r.team_id])
        st.html(f'<div class="vs-panel" style="padding:14px 20px"><div class="vs-pick">'
                f'<div style="text-align:center">{ui.ring(p["score"], 80, "PICK", ui.score_color(p["score"]))}</div><div>'
                f'<div class="t"><span style="color:#8B909B">#{i}</span> {ui.escape(r.number)} '
                f'<span style="font-weight:500;color:#B4B8C2">· {ui.escape(r.name)}</span></div>'
                f'<div style="margin:4px 0">{ui.tier_chip(r.tier)}{ui.chip(r.profile)}{ui.chip(f"Data: {r.confidence}")}</div>'
                + (f'<div style="margin:4px 0">{badges}</div>' if badges else "")
                + f'<div class="s">{ui.escape(" · ".join(why) or "Solid all-round numbers")}</div></div>'
                f'<div style="display:flex;gap:26px"><div class="big">{p["exp_points"]:.0f}<small>EXP. PTS TOGETHER</small></div>'
                f'<div class="big">{p["p_vs_typical"]:.0%}<small>VS TYPICAL ALLIANCE</small></div></div></div></div>')
    if picks:
        st.caption("Pick score = 60% partner overall + 40% fit, where fit weights the components you're weakest in more heavily. "
                   "“vs typical alliance” = win chance of the pair against an average alliance at this event.")


# --- match predictor -----------------------------------------------------------
with tab_pred:
    label = lambda n: f"{n} · {by_number[n]['name'] or ''}"
    left, right = st.columns(2, gap="large")
    with left:
        st.html(f'<div class="vs-lbl" style="color:{ui.RED}">● Red alliance</div>')
        r1 = st.selectbox("Red 1", numbers, key="red1", format_func=label, placeholder="Red team 1",
                          index=numbers.index(MY_TEAM_NUMBER) if MY_TEAM_NUMBER in numbers else None,
                          label_visibility="collapsed")
        r2 = st.selectbox("Red 2", [n for n in numbers if n != r1], key="red2", index=None,
                          format_func=label, placeholder="Red team 2", label_visibility="collapsed")
    with right:
        st.html(f'<div class="vs-lbl" style="color:{ui.BLUE}">● Blue alliance</div>')
        taken = {r1, r2}
        b1 = st.selectbox("Blue 1", [n for n in numbers if n not in taken], key="blue1", index=None,
                          format_func=label, placeholder="Blue team 1", label_visibility="collapsed")
        b2 = st.selectbox("Blue 2", [n for n in numbers if n not in taken | {b1}], key="blue2", index=None,
                          format_func=label, placeholder="Blue team 2", label_visibility="collapsed")
    red = [n for n in (r1, r2) if n]
    blue = [n for n in (b1, b2) if n]
    if st.button("Predict", type="primary", disabled=not (red and blue)):
        st.session_state.last_pred = predictor.predict_numbers(conn, EID, red, blue, model=model)
    p = st.session_state.get("last_pred")
    if p:
        p_ml = None
        if ML.usable and [x["team"] for x in p["teams"]["red"]] == list(red) and [x["team"] for x in p["teams"]["blue"]] == list(blue):
            p_ml = ml.predict(ML, [by_number[n]["id"] for n in red], [by_number[n]["id"] for n in blue], model, R,
                              feedback.current_status(conn, EID))
            p["ml_red_win_prob"] = round(p_ml, 3)  # the AI sees it too
        rt = ", ".join(x["team"] for x in p["teams"]["red"])
        bt = ", ".join(x["team"] for x in p["teams"]["blue"])
        sd = p["score_sd"]
        rng = lambda v: f"{max(0, v - sd):.0f}–{v + sd:.0f}"
        no_data = [x["team"] for side in ("red", "blue") for x in p["teams"][side] if x["matches"] == 0]
        lvl = p["confidence"]
        col, bg = {"low": ("#F5B95A", "#2E2210"), "medium": ("#9DBDF8", "#142038"), "high": ("#6BD58A", "#12301C")}[lvl]
        why = ([f"{', '.join(no_data)} {'has' if len(no_data) == 1 else 'have'} no match data this season"] if no_data else []) + \
              [g for g in p["data_gaps"] if not any(g.startswith(n + " ") for n in no_data)][:3]
        if not why and lvl != "high":
            why = ["Points model and Elo disagree, or the margin is small"]
        def headline(ai=None, disc=None):
            """Score panel. With an AI call it shows the blend: 50% stats model + 50% AI
            (Claude's final call, averaged with ChatGPT's when ChatGPT reviewed it)."""
            panel = (disc or {}).get("ai_probs") if ai and (disc or {}).get("version") in (3, 4) else None
            gpt = (disc or {}).get("review", {}).get("red_win_prob") if ai and not panel else None
            clip = lambda v: min(max(v, 0.0), 1.0)
            comps = [("Stats model", p["red_win_prob"])]
            if p_ml is not None:
                comps.append(("ML model", p_ml))
            if ai:
                if panel:
                    ai_p = disc["ai_median"]
                else:
                    ai_p = clip(ai.red_win_prob) if gpt is None else (clip(ai.red_win_prob) + clip(gpt)) / 2
                comps.append(("AI", ai_p))
            pr = sum(v for _, v in comps) / len(comps)  # equal weight to each source
            rs, bs = ((p["red_score"] + ai.red_score) / 2, (p["blue_score"] + ai.blue_score) / 2) if ai \
                else (p["red_score"], p["blue_score"])
            shown = [f'Stats <b>{p["red_win_prob"]:.0%}</b>'] + ([f'ML <b>{p_ml:.0%}</b>'] if p_ml is not None else []) \
                + ([f'{AGENT_LABEL[a]} <b>{v:.0%}</b>' for a, v in panel.items()] if panel else
                   ([f'Claude <b>{ai.red_win_prob:.0%}</b>'] if ai else []) + ([f'ChatGPT <b>{gpt:.0%}</b>'] if gpt is not None else []))
            how = " + ".join(n for n, _ in comps) if len(comps) > 1 else "stats model only"
            note = ("" if ai else " The AI's call is added once its analysis is ready.")
            basis = (f'<div class="vs-blend">Red win chance: {" · ".join(shown)}'
                     + (f' → <b>combined {pr:.0%}</b>' if len(comps) > 1 else "")
                     + f'<span>Equal parts {how}{" (the AI share is the panel median)" if panel else " (Claude and ChatGPT averaged)" if gpt is not None else ""}. '
                       f'Scores: stats model{" averaged with the AI" if ai else ""}.{note}</span></div>')
            tag = "Combined prediction · " + " + ".join(["stats"] + (["ML"] if p_ml is not None else []) + (["AI"] if ai else []))
            return (f'<div class="vs-predict"><div class="vs-panel" style="margin:0">'
                    f'<div class="vs-lbl" style="margin-top:0">{tag}</div><div class="vs-vs">'
                    f'<div><div class="sc" style="color:{ui.RED}">{rs:.0f}</div><div class="tm">{ui.escape(rt)}</div>'
                    f'<div class="tm" style="font-size:.8rem">likely {rng(rs)}</div></div>'
                    f'<div style="color:#8B909B">vs</div>'
                    f'<div><div class="sc" style="color:{ui.BLUE}">{bs:.0f}</div><div class="tm">{ui.escape(bt)}</div>'
                    f'<div class="tm" style="font-size:.8rem">likely {rng(bs)}</div></div>'
                    f'</div>{ui.split_bar(pr)}{basis}</div>'
                    f'<div class="vs-unc" style="background:{bg};border:1px solid {col}66">'
                    f'<div class="lvl" style="color:{col}">{"⚠️ " if lvl == "low" else ""}{lvl.title()} confidence</div>'
                    f'<div class="pm">Scores could be off by ±{sd:.0f} pts</div>'
                    + (f'<ul>{"".join(f"<li>{ui.escape(w)}</li>" for w in why)}</ul>' if why else
                       '<div style="color:#B4B8C2">Plenty of match data on all four teams.</div>')
                    + (f'<div style="margin-top:8px;color:#B4B8C2;font-size:.85rem">Treat this as a rough guess - scout '
                       f'these teams before relying on it.</div>' if lvl == "low" else "") + '</div></div>')

        head = st.empty()  # stats numbers now; redrawn with the AI blend once the analysis is ready
        head.html(headline())
        if st.button("🧭 Plan this matchup in Strategy copilot", type="primary"):
            ours = [n for n in red + blue if n == MY_TEAM_NUMBER]
            side = "red" if MY_TEAM_NUMBER in red else "blue" if MY_TEAM_NUMBER in blue else None
            st.session_state.copilot_pending = (
                f"Plan this matchup: red {' & '.join(red)} vs blue {' & '.join(blue)}. "
                + (f"We are {MY_TEAM_NUMBER} on {side}. " if side else "")
                + "Give us roles, auton coordination, scoring priorities and whether to play defense.")
            st.session_state.goto_tab = "Strategy copilot"
            st.rerun()
        items = "".join(f"<li>{ui.escape(f)}</li>" for f in p["factors"])
        st.html(f'<div class="vs-panel" style="margin-top:12px"><b>Key factors</b><ul style="margin:.4rem 0 0">{items}</ul></div>')
        ai = None
        if [x["team"] for x in p["teams"]["red"]] == list(red) and [x["team"] for x in p["teams"]["blue"]] == list(blue):
            rr = [R[by_number[n]["id"]] for n in red]
            bb = [R[by_number[n]["id"]] for n in blue]
            saved, diff = analysis.saved_match_analysis(conn, EID, rr, bb, p)
            ai = saved["report"] if saved else None
            mdisc = saved["discussion"] if saved else None
            when = saved and datetime.fromisoformat(saved["created_at"]).astimezone().strftime("%b %-d, %-I:%M %p")
            mfail = ("ai_match_fail", EID, tuple(red), tuple(blue), st.session_state.data_version)
            forced = st.session_state.pop("ai_match_force", False)
            if (saved is None or diff or forced) and mfail not in st.session_state:
                msg = ("Stats changed a lot since the saved analysis (" + "; ".join(diff) + ") - updating it…" if diff
                       else "Analyzing this matchup…")
                with st.status(msg, expanded=False) as stat:
                    try:
                        ours = MY_TEAM_NUMBER if MY_TEAM_NUMBER in red + blue else None
                        ai, mdisc = analysis.new_match_analysis(conn, EID, rr, bb, p, R, ours,
                                                                on_stage=lambda m: stat.update(label=m))
                        when = "just now"
                        stat.update(label="Analysis ready", state="complete")
                    except Exception as e:
                        st.session_state[mfail] = f"{type(e).__name__}: {e}"
                        stat.update(label="AI analysis unavailable", state="error")
            if mfail in st.session_state:
                st.error(f"AI analysis unavailable ({st.session_state[mfail]})" + (" - showing the last saved one." if ai else ""))
            if ai:
                c1, c2 = st.columns([5, 1.2], vertical_alignment="center")
                c1.caption(f"AI match analysis from {when} · updates automatically when the stats change a lot")
                if c2.button("↻ Regenerate", key="ai_match_regen", type="tertiary"):
                    st.session_state.ai_match_force = True
                    st.session_state.pop(mfail, None)
                    st.rerun()
        if ai:
            head.html(headline(ai, mdisc))
            win = {"red": f'<span style="color:{ui.RED}">Red</span>', "blue": f'<span style="color:{ui.BLUE}">Blue</span>',
                   "toss-up": "Toss-up"}[ai.predicted_winner]
            mrefs = ref_index(rr + bb, {})
            st.html(f'<div class="vs-ai"><div class="tag">AI match analysis · {reviewers(mdisc)} · '
                    f'<span title="{AI_CERT_HELP}">AI certainty: {ai.certainty} ⓘ</span></div>'
                    + top3(("Decisive factor", "#6BD58A", ai.decisive_factor, "", ""),
                           ("Biggest uncertainty", "#F5B95A", ai.biggest_uncertainty, "", ""),
                           ("Recommended action", "#9DBDF8", ai.recommended_action, "", ""))
                    + f'<div style="display:flex;gap:22px;align-items:baseline;flex-wrap:wrap">'
                    f'<div style="font-size:1.25rem;font-weight:800">{win} {"" if ai.predicted_winner == "toss-up" else "wins"}</div>'
                    f'<div style="font-size:1.15rem;font-weight:700"><span style="color:{ui.RED}">{ai.red_score}</span> – '
                    f'<span style="color:{ui.BLUE}">{ai.blue_score}</span></div>'
                    f'<div class="ev">Stats model: {p["red_score"]:.0f}–{p["blue_score"]:.0f}, red {p["red_win_prob"]:.0%}</div></div>'
                    f'<p class="ev" style="margin-top:6px">{ui.escape(ai.vs_model)}</p></div>')
            discussion_view(mdisc, match=True)
            with st.expander("Full analysis: reasons, game plans, upset risks"):
                st.html(f'<div class="vs-ai" style="margin:0"><h4>Why</h4><ul>{claims(ai.reasoning, mrefs)}</ul>'
                        f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:18px">'
                        f'<div><h4 style="color:{ui.RED}">Red game plan</h4><ul>{bullets(ai.red_game_plan)}</ul></div>'
                        f'<div><h4 style="color:{ui.BLUE}">Blue game plan</h4><ul>{bullets(ai.blue_game_plan)}</ul></div>'
                        f'<div><h4>Upset risks</h4><ul>{bullets(ai.upset_risks)}</ul></div></div>'
                        f'<div class="ev">Match chips (e.g. 80001C-M2) are that team\'s 2nd most recent match; hover for details.</div></div>')
        with st.expander("Advanced: per-team numbers"):
            st.dataframe(pd.DataFrame([{**x, "alliance": side, "notes": "; ".join(x["notes"])}
                                       for side in ("red", "blue") for x in p["teams"][side]]),
                         hide_index=True, width="stretch")
        with st.expander("Advanced: record the actual result"):
            r1, r2 = st.columns(2)
            ar = r1.number_input("Red actual", min_value=0, step=1)
            ab = r2.number_input("Blue actual", min_value=0, step=1)
            if st.button("Save result"):
                conn.execute("UPDATE predictions SET actual_red=?, actual_blue=? WHERE id=?", (ar, ab, p["prediction_id"]))
                conn.commit(); st.success("Saved.")

    with st.expander("Advanced: the machine-learning model"):
        if ML.w is None:
            st.caption(f"Not enough scored matches this season to train it yet (needs {ml.MIN_TRAIN}).")
        else:
            t_ = ML.test
            st.markdown(f"**Logistic regression** trained on **{ML.n_train}** matches this season. Each match is described only "
                        "by what was known *before* it was played, so it never learns from the future.")
            st.html(ui.cards([(f"{t_['ml_acc']:.0%}", f"ML accuracy · {t_['n']} unseen matches"),
                              (f"{t_['ml_brier']:.3f}", "ML Brier (lower = better)"),
                              (f"{t_['elo_brier']:.3f}", "Elo-only Brier, same matches"),
                              ("in use" if ML.usable else "left out", "in the blend")]))
            st.caption("What it learned (weight per standard deviation; bigger = matters more): " + " · ".join(
                f"{f} {w:+.2f}" for f, w in ml.weights(ML)))
    with st.expander("Advanced: model accuracy & backtest"):
        ev_rows = db.rows(conn, "SELECT * FROM predictions WHERE event_id=? AND actual_red IS NOT NULL", (EID,))
        if ev_rows:
            hits = [(r["red_win_prob"] > 0.5) == (r["actual_red"] > r["actual_blue"]) for r in ev_rows if r["actual_red"] != r["actual_blue"]]
            st.caption(f"Live predictions with recorded results: {len(ev_rows)} · correct winner: {sum(hits)}/{len(hits)}")
        if st.button("Run chronological backtest on this event"):
            with st.spinner("Replaying matches…"):
                bt = predictor.backtest(conn, EID)
            if bt["n"] == 0:
                st.info("Not enough scored matches at this event to backtest yet.")
            else:
                st.html(ui.cards([(bt["n"], "Matches"), (f"{bt['accuracy']:.0%}", "Winner accuracy"),
                                  (bt["brier"], "Brier (coin flip 0.25)"), (bt["score_mae"], "Score MAE")]))
                st.dataframe(pd.DataFrame(bt["calibration"]), hide_index=True)
                st.caption("Each match is predicted using only matches that finished before it.")


# --- strategy copilot ----------------------------------------------------------
def _has_tool_use(msg):
    content = msg["content"] if isinstance(msg["content"], list) else []
    return any((b.get("type") if isinstance(b, dict) else getattr(b, "type", None)) == "tool_use" for b in content)


def complete_turns(api):
    """History up to the last finished turn (an assistant reply with no pending tool call).
    A turn cut off mid-way (page rerun, phone locked) would otherwise leave a tool call with
    no result, and the API rejects every later message with a 400."""
    for i in range(len(api) - 1, -1, -1):
        if api[i]["role"] == "assistant" and not _has_tool_use(api[i]):
            return api[:i + 1]
    return []


def _ask(question):
    st.session_state.copilot_pending = question


with tab_copilot:
    key = f"chat_{EID}"
    if key not in st.session_state:
        st.session_state[key] = {"api": [], "display": []}
    chat = st.session_state[key]
    chat["api"] = complete_turns(chat["api"])
    if chat["display"] and chat["display"][-1][0] == "user":  # an answer was interrupted last time
        chat["display"].append(("assistant", "⚠️ That answer was interrupted (the page reloaded or the phone went to sleep). "
                                             "Ask again - this can take up to a minute."))
    opts = ["–"] + numbers
    our = st.selectbox("Our team", opts, key="our_team",
                       index=opts.index(MY_TEAM_NUMBER) if MY_TEAM_NUMBER in opts else 0)
    msgs_area = st.container()   # conversation sits above the input, new messages included
    sugg_area = st.container()
    typed = st.chat_input("Ask about opponents, partners, or match strategy… or tap the mic below")
    said = mic.mic(key=f"copilot_mic_{EID}_{len(chat['display'])}")  # fresh key per turn; asks as soon as you stop talking
    prompt = typed or said or st.session_state.pop("copilot_pending", None)

    with msgs_area:
        for role, text in chat["display"]:
            st.chat_message(role).markdown(text)
    if not chat["display"] and not prompt:
        us = our if our != "–" else (MY_TEAM_NUMBER or "us")
        top = [r.number for r in sorted((x for x in R.values() if x.overall is not None and x.number != us),
                                        key=lambda x: -x.overall)]
        ideas = []
        if len(top) >= 2:
            ideas.append(f"How should {us} play against {top[0]} and {top[1]}?")
        ideas.append(f"Who are the best alliance picks for {us}?")
        if top:
            ideas.append(f"What are {top[0]}'s biggest weaknesses?")
        ideas.append(f"Which teams should we scout first at {SHORT_NAME}?")
        with sugg_area:
            st.caption("Tap a question to ask it:")
            cols = st.columns(2)
            for i, idea in enumerate(ideas):
                cols[i % 2].button(idea, key=f"idea_{i}", width="stretch", on_click=_ask, args=(idea,))
    if prompt:
        with msgs_area:
            st.chat_message("user").markdown(prompt)
            content = prompt if our == "–" else f"[We are team {our}] {prompt}"
            chat["display"].append(("user", prompt))
            work = list(chat["api"]) + [{"role": "user", "content": content}]  # committed only if the turn finishes
            with st.chat_message("assistant"):
                status = st.status("Thinking… (usually 20–60 seconds - keep this page open)", expanded=False)
                try:
                    from vexscout.agent import Agent
                    agent = Agent(conn, EID)
                    agent._model = model
                    answer = agent.chat(work, on_tool=lambda n, a: status.write(f"🔧 `{n}` {json.dumps(a)}"))
                    chat["api"] = work
                    status.update(label="Done", state="complete")
                    bump()  # the agent may have recorded observations
                except Exception as e:
                    status.update(label="Copilot unavailable", state="error")
                    answer = (f"⚠️ Couldn't reach Claude ({type(e).__name__}: {e}). "
                              "The teams, picks, and predictor tabs still work offline.")
                st.markdown(answer)
            chat["display"].append(("assistant", answer))
