# app.py
# ------------------------------------------------------------
# NSE Weekly Breakout/Breakdown Screener (Bullish & Bearish)
# + Auto-refresh timer + Manual "Refresh now"
# + Snapshots (save/compare) + ML anomaly
# + Auto-train ML next-move predictor (LONG/SHORT/WAIT)
#   with regression fallback when labels have only one class
# ------------------------------------------------------------

import os, glob, time, math, warnings
from typing import Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# ML / AI
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Data sources
try:
    from nsepython import nse_optionchain_scrapper
    NSEPY = True
except Exception:
    NSEPY = False

try:
    import yfinance as yf
    YFIN = True
except Exception:
    YFIN = False

warnings.filterwarnings("ignore")

SNAP_DIR = "snapshots"
os.makedirs(SNAP_DIR, exist_ok=True)

st.set_page_config(page_title="NSE Breakout/Breakdown Screener", layout="wide")
st.title("📈 NSE Weekly Screener — Bullish Breakouts & Bearish Breakdowns")

# ---------------- Session State (for refresh bookkeeping) ----------------
if "last_refresh_ts" not in st.session_state:
    st.session_state.last_refresh_ts = time.time()

def _mark_refreshed():
    st.session_state.last_refresh_ts = time.time()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Universe & Mode")
    default_syms = "RELIANCE,HDFCBANK,TCS,INFY,ITC,LT,ICICIBANK,SBIN,TATAMOTORS,AXISBANK"
    symbols = [s.strip().upper() for s in st.text_area(
        "NSE Symbols (comma-separated)", value=default_syms
    ).split(",") if s.strip()]

    direction = st.selectbox("Direction", ["Both", "Bullish", "Bearish"])

    st.subheader("Weights (Probability Scoring)")
    w_ce_oi  = st.slider("CE OI build-up (Bullish) / PE OI build-up (Bearish)", 0.0, 1.0, 0.25, 0.05)
    w_pcr    = st.slider("PCR bias (bullish if <0.8 / bearish if >1.2)", 0.0, 1.0, 0.20, 0.05)
    w_iv     = st.slider("IV bias (CE≥PE bullish / PE≥CE bearish)", 0.0, 1.0, 0.15, 0.05)
    w_vol    = st.slider("Volume dominance (CE bullish / PE bearish)", 0.0, 1.0, 0.25, 0.05)
    w_trend  = st.slider("Price vs trigger (>= breakout / <= breakdown)", 0.0, 1.0, 0.15, 0.05)

    st.subheader("Levels & Filters")
    rr_min = st.slider("Show only R:R ≥", 1.0, 4.0, 1.8, 0.1)
    atr_t  = st.slider("Target = Entry ± ATR ×", 0.5, 3.0, 1.0, 0.1)
    atr_s  = st.slider("Stop = Entry ∓ ATR ×", 0.3, 2.0, 0.7, 0.1)

    st.subheader("Intraday Rules (Display Only)")
    vol_mult_30m_bull = st.slider("Bullish: CE vol > × avg (30m)", 1.0, 4.0, 2.0, 0.1)
    vol_mult_30m_bear = st.slider("Bearish: PE vol > × avg (30m)", 1.0, 4.0, 2.0, 0.1)
    sustain_mins = st.slider("Hold above/below trigger for (mins)", 5, 45, 15, 5)
    pcr_avoid_bull = st.slider("Bullish: avoid if intraday PCR >", 0.8, 1.8, 1.2, 0.05)
    pcr_avoid_bear = st.slider("Bearish: avoid if intraday PCR <", 0.4, 1.2, 0.8, 0.05)
    iv_floor_bull = st.slider("Bullish: CE IV ≥ (%)", 8, 35, 16, 1)
    iv_floor_bear = st.slider("Bearish: PE IV ≥ (%)", 8, 35, 16, 1)

    st.subheader("🔄 Refresh")
    auto_refresh_on = st.toggle("Enable auto-refresh", value=False,
                                help="Turn on to refresh the data automatically.")
    auto_refresh_secs = st.number_input("Interval (seconds)", min_value=5, max_value=600, value=30, step=5)
    hard_refresh = st.toggle("Hard refresh (clear cache on button)", value=False,
                             help="Clears cached data when you click Refresh now")

    st.subheader("🤖 ML Prediction Settings")
    fwd_days = st.number_input("Label horizon: forward days", min_value=1, max_value=5, value=1)
    decision_thr = st.slider("Decision threshold (prob)", 0.5, 0.8, 0.55, 0.01)

# -------- Auto-refresh hook (re-run page every N seconds when ON) --------
if auto_refresh_on:
    _ = st_autorefresh(interval=auto_refresh_secs * 1000, key="auto_refresh_timer")

# Smart cache-busting salt (changes each tick) for cached functions
salt = int(time.time() // auto_refresh_secs) if (auto_refresh_on and auto_refresh_secs > 0) else 0

# --------------- Manual Refresh UI (top bar) ---------------
r1, r2 = st.columns([1, 3])
with r1:
    if st.button("🔁 Refresh now", help="Force an immediate data refresh"):
        if hard_refresh:
            try: get_chain.clear()
            except: pass
            try: get_atr.clear()
            except: pass
        _mark_refreshed()
        st.rerun()
with r2:
    last = datetime.fromtimestamp(st.session_state.last_refresh_ts).strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"Last refreshed: **{last}**  |  Auto-refresh: **{'ON' if auto_refresh_on else 'OFF'}** ({auto_refresh_secs}s)")

# ---------------- Data Helpers ----------------
@st.cache_data(ttl=120)
def get_chain(symbol: str, _salt: int = 0):
    """Option-chain fetch (salt used to bust cache on timer)."""
    if not NSEPY:
        return None
    try:
        return nse_optionchain_scrapper(symbol)
    except Exception:
        return None

@st.cache_data(ttl=600)
def get_atr(symbol: str, days=90, _salt: int = 0):
    """ATR fetch (salt used to bust cache on timer)."""
    if not YFIN:
        return None
    try:
        df = yf.Ticker(f"{symbol}.NS").history(period=f"{days}d", interval="1d")
        if df.empty:
            return None
        hc = df["High"]; lc = df["Low"]; cc = df["Close"]; pc = cc.shift(1)
        tr = pd.concat([(hc-lc).abs(), (hc-pc).abs(), (lc-pc).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        return float(atr) if pd.notna(atr) else None
    except Exception:
        return None

def to_frames(payload):
    if not payload:
        return None, None, np.nan
    rec = payload.get("records", {})
    spot = rec.get("underlyingValue", np.nan)
    data = payload.get("filtered", {}).get("data", []) or rec.get("data", [])
    ce_rows, pe_rows = [], []
    for r in data:
        sp = r.get("strikePrice")
        if r.get("CE"):
            ce = r["CE"]
            ce_rows.append(dict(
                strikePrice=sp,
                openInterest=ce.get("openInterest"),
                changeinOpenInterest=ce.get("changeinOpenInterest"),
                totalTradedVolume=ce.get("totalTradedVolume"),
                impliedVolatility=ce.get("impliedVolatility"),
                lastPrice=ce.get("lastPrice"),
            ))
        if r.get("PE"):
            pe = r["PE"]
            pe_rows.append(dict(
                strikePrice=sp,
                openInterest=pe.get("openInterest"),
                changeinOpenInterest=pe.get("changeinOpenInterest"),
                totalTradedVolume=pe.get("totalTradedVolume"),
                impliedVolatility=pe.get("impliedVolatility"),
                lastPrice=pe.get("lastPrice"),
            ))
    df_ce = pd.DataFrame(ce_rows).dropna(subset=["strikePrice"]).sort_values("strikePrice")
    df_pe = pd.DataFrame(pe_rows).dropna(subset=["strikePrice"]).sort_values("strikePrice")
    for df in (df_ce, df_pe):
        for c in ["openInterest","changeinOpenInterest","totalTradedVolume","impliedVolatility","lastPrice"]:
            if c in df:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    return df_ce, df_pe, float(spot) if pd.notna(spot) else np.nan

def max_pain_fast(df_ce, df_pe):
    if df_ce is None or df_pe is None or df_ce.empty or df_pe.empty:
        return np.nan
    m = pd.merge(
        df_ce[["strikePrice","openInterest"]].rename(columns={"openInterest":"CE_OI"}),
        df_pe[["strikePrice","openInterest"]].rename(columns={"openInterest":"PE_OI"}),
        on="strikePrice", how="inner"
    )
    m["TOTAL"] = pd.to_numeric(m["CE_OI"], errors="coerce").fillna(0) + pd.to_numeric(m["PE_OI"], errors="coerce").fillna(0)
    if m.empty:
        return np.nan
    return float(m.sort_values("TOTAL", ascending=False).iloc[0]["strikePrice"])

def pick_trigger_bullish(df_ce, spot):
    if df_ce is None or df_ce.empty or pd.isna(spot):
        return np.nan
    above = df_ce[df_ce["strikePrice"] >= spot]
    if above.empty:
        return float(df_ce["strikePrice"].iloc[-1])
    tmp = above.copy()
    tmp["oi"] = pd.to_numeric(tmp["openInterest"], errors="coerce").fillna(0.0)
    tmp["doi"] = pd.to_numeric(tmp["changeinOpenInterest"], errors="coerce").fillna(0.0)
    tmp["score"] = tmp["oi"] * (1 + np.clip(tmp["doi"], 0, None))
    tmp["dist"] = (tmp["strikePrice"] - spot).abs()
    tmp = tmp.sort_values(["score","dist"], ascending=[False, True])
    return float(tmp.iloc[0]["strikePrice"])

def pick_trigger_bearish(df_pe, spot):
    if df_pe is None or df_pe.empty or pd.isna(spot):
        return np.nan
    below = df_pe[df_pe["strikePrice"] <= spot]
    if below.empty:
        return float(df_pe["strikePrice"].iloc[0])
    tmp = below.copy()
    tmp["oi"] = pd.to_numeric(tmp["openInterest"], errors="coerce").fillna(0.0)
    tmp["doi"] = pd.to_numeric(tmp["changeinOpenInterest"], errors="coerce").fillna(0.0)
    tmp["score"] = tmp["oi"] * (1 + np.clip(tmp["doi"], 0, None))
    tmp["dist"] = (tmp["strikePrice"] - spot).abs()
    tmp = tmp.sort_values(["score","dist"], ascending=[False, True])
    return float(tmp.iloc[0]["strikePrice"])

def score_symbol(df_ce, df_pe, spot, trigger, atr, weights, side, atr_t, atr_s):
    ce_oi  = pd.to_numeric(df_ce.get("openInterest"), errors="coerce").fillna(0).sum() if df_ce is not None else 0.0
    pe_oi  = pd.to_numeric(df_pe.get("openInterest"), errors="coerce").fillna(0).sum() if df_pe is not None else 0.0
    ce_doi = pd.to_numeric(df_ce.get("changeinOpenInterest"), errors="coerce").fillna(0).sum() if df_ce is not None else 0.0
    pe_doi = pd.to_numeric(df_pe.get("changeinOpenInterest"), errors="coerce").fillna(0).sum() if df_pe is not None else 0.0
    ce_vol = pd.to_numeric(df_ce.get("totalTradedVolume"), errors="coerce").fillna(0).sum() if df_ce is not None else 0.0
    pe_vol = pd.to_numeric(df_pe.get("totalTradedVolume"), errors="coerce").fillna(0).sum() if df_pe is not None else 0.0
    ce_iv  = pd.to_numeric(df_ce.get("impliedVolatility"), errors="coerce").median() if (df_ce is not None and not df_ce.empty) else np.nan
    pe_iv  = pd.to_numeric(df_pe.get("impliedVolatility"), errors="coerce").median() if (df_pe is not None and not df_pe.empty) else np.nan
    pcr = (pe_oi / ce_oi) if ce_oi > 0 else np.nan

    score = 0.0
    oi_score = np.tanh(max(0.0, ce_doi if side=="bull" else pe_doi) / 1e5)
    score += weights["oi"] * oi_score

    if not pd.isna(pcr):
        if side == "bull":
            pcr_score = 1.0 if pcr <= 0.8 else (0.0 if pcr >= 1.2 else (1.2 - pcr) / 0.4)
        else:
            pcr_score = 1.0 if pcr >= 1.2 else (0.0 if pcr <= 0.8 else (pcr - 0.8) / 0.4)
        score += weights["pcr"] * pcr_score

    if not pd.isna(ce_iv) and not pd.isna(pe_iv):
        if side == "bull":
            iv_score = 1.0 if ce_iv >= pe_iv else max(0.0, 1.0 + (ce_iv - pe_iv) / max(1.0, pe_iv))
        else:
            iv_score = 1.0 if pe_iv >= ce_iv else max(0.0, 1.0 + (pe_iv - ce_iv) / max(1.0, ce_iv))
        score += weights["iv"] * float(np.clip(iv_score, 0, 1))

    v_ratio = (ce_vol / (pe_vol + 1.0)) if side == "bull" else (pe_vol / (ce_vol + 1.0))
    vol_score = np.tanh(v_ratio - 1.0) * 0.5 + 0.5
    score += weights["vol"] * float(np.clip(vol_score, 0, 1))

    trend_score = 0.5
    if not pd.isna(spot) and not pd.isna(trigger):
        trend_score = 1.0 if (spot >= trigger if side == "bull" else spot <= trigger) else 0.4
    score += weights["trend"] * trend_score

    total_w = sum(weights.values())
    prob = round(float(np.clip(score / total_w * 100.0, 0, 100)) if total_w > 0 else 0.0, 1)

    entry_lo, entry_hi, target, stop = np.nan, np.nan, np.nan, np.nan
    if not pd.isna(trigger):
        if side == "bull":
            entry_lo = trigger
            entry_hi = trigger * 1.005
            if atr and not pd.isna(atr):
                target = trigger + atr_t * atr
                stop   = trigger - atr_s * atr
            else:
                target = trigger * 1.015
                stop   = trigger * 0.99
        else:
            entry_hi = trigger
            entry_lo = trigger * 0.995
            if atr and not pd.isna(atr):
                target = trigger - atr_t * atr
                stop   = trigger + atr_s * atr
            else:
                target = trigger * 0.985
                stop   = trigger * 1.01

    rr = np.nan
    if not any(pd.isna([entry_lo, target, stop])):
        if side == "bull" and (target > entry_lo) and (stop < entry_lo):
            rr = (target - entry_lo) / (entry_lo - stop)
        if side == "bear" and (entry_hi > target) and (stop > entry_hi):
            e = (entry_lo + entry_hi) / 2.0
            rr = (e - target) / (stop - e)
    rr = float(rr) if not pd.isna(rr) else np.nan

    return dict(
        spot=float(spot) if pd.notna(spot) else np.nan,
        trigger=float(trigger) if pd.notna(trigger) else np.nan,
        entry_zone=(round(entry_lo,2) if pd.notna(entry_lo) else None,
                    round(entry_hi,2) if pd.notna(entry_hi) else None),
        target=float(round(target,2)) if pd.notna(target) else None,
        stop=float(round(stop,2)) if pd.notna(stop) else None,
        rr=float(round(rr,2)) if not pd.isna(rr) else None,
        prob=prob,
        pcr=float(round(pcr,2)) if not pd.isna(pcr) else None,
        ce_oi=float(ce_oi), pe_oi=float(pe_oi),
        ce_doi=float(ce_doi), pe_doi=float(pe_doi),
        ce_vol=float(ce_vol), pe_vol=float(pe_vol),
        ce_iv=float(ce_iv) if not pd.isna(ce_iv) else None,
        pe_iv=float(pe_iv) if not pd.isna(pe_iv) else None,
        max_pain=float(max_pain_fast(df_ce, df_pe)) if not (df_ce is None or df_pe is None) else None,
        atr=float(atr) if atr else None,
    )

def analyze(symbol, atr_t, atr_s):
    payload = get_chain(symbol, salt)
    if not payload:
        return {"symbol": symbol, "error": "Option chain fetch failed."}
    df_ce, df_pe, spot = to_frames(payload)
    atr = get_atr(symbol, _salt=salt)
    weights = {"oi":w_ce_oi,"pcr":w_pcr,"iv":w_iv,"vol":w_vol,"trend":w_trend}
    trg_bull = pick_trigger_bullish(df_ce, spot)
    bull = score_symbol(df_ce, df_pe, spot, trg_bull, atr, weights, side="bull", atr_t=atr_t, atr_s=atr_s)
    trg_bear = pick_trigger_bearish(df_pe, spot)
    bear = score_symbol(df_ce, df_pe, spot, trg_bear, atr, weights, side="bear", atr_t=atr_t, atr_s=atr_s)
    return {"symbol": symbol, "bull": bull, "bear": bear}

# --------------- Run (fetch & compute) ---------------
rows = []
with st.spinner("Fetching option chains & computing signals..."):
    for sym in symbols:
        try:
            out = analyze(sym, atr_t, atr_s)
        except Exception as e:
            out = {"symbol": sym, "error": str(e)}
        rows.append(out)

def build_df(rows, side: str):
    recs = []
    for r in rows:
        sym = r.get("symbol")
        blk = r.get(side)
        if not blk:
            recs.append({"symbol": sym, "error": r.get("error","analysis failed")})
            continue
        d = {"symbol": sym} | blk
        recs.append(d)
    df = pd.DataFrame(recs)
    if "rr" in df:
        df["_pass"] = df["rr"].apply(lambda x: isinstance(x, (int,float)) and x >= rr_min)
    else:
        df["_pass"] = False
    return df.sort_values(["_pass","prob"], ascending=[False, False]).reset_index(drop=True)

df_bull = build_df(rows, "bull")
df_bear = build_df(rows, "bear")

# Mark refreshed after successful fetch
_mark_refreshed()

# --------- UI Cards ---------
def draw_cards(df, side_name, intraday_rules):
    st.subheader(f"🏆 Top 5 — {side_name}")
    top = df.head(5)
    if top.empty:
        st.info("No candidates passed filters.")
        return
    cols = st.columns(len(top)) if len(top) < 5 else st.columns(5)
    for i, (_, row) in enumerate(top.iterrows()):
        with cols[i]:
            st.markdown(f"### {i+1}. **{row['symbol']}**")
            if isinstance(row.get("error"), str) and row["error"]:
                st.error(row["error"]); continue
            spot = row.get("spot"); trg = row.get("trigger")
            ez = row.get("entry_zone"); tg = row.get("target"); sl = row.get("stop")
            st.metric("Probability", f"{row.get('prob',0):.1f}%")
            st.write(f"**Spot**: {'' if spot is None or pd.isna(spot) else round(spot,2)}")
            st.write(f"**Trigger**: {'' if trg is None or pd.isna(trg) else round(trg,2)}")
            if ez and ez[0] and ez[1]:
                st.write(f"**Entry Zone**: {ez[0]} – {ez[1]}")
            elif ez and ez[0]:
                st.write(f"**Entry**: {ez[0]}")
            st.write(f"**Target**: {'' if tg is None or pd.isna(tg) else tg}")
            st.write(f"**Stop-loss**: {'' if sl is None or pd.isna(sl) else sl}")
            st.write(f"**R:R**: {'' if row.get('rr') in [None, np.nan] else row['rr']}")
            st.caption(f"PCR: {'' if row.get('pcr') in [None, np.nan] else row['pcr']}, "
                       f"MaxPain: {'' if row.get('max_pain') in [None, np.nan] else row['max_pain']}, "
                       f"ATR: {'' if row.get('atr') in [None, np.nan] else round(row['atr'],2)}")
            st.markdown(intraday_rules, help="Execution checklist to avoid false signals")

bull_rules = f"""
- ✅ **Enter only if CE volume > {vol_mult_30m_bull}× avg in first 30 mins**
- ✅ **Spot sustains ABOVE trigger for {sustain_mins} mins**
- ✅ **ATM/Breakout CE IV ≥ {iv_floor_bull}%** during trigger
- ❌ Avoid if **intraday PCR > {pcr_avoid_bull}** or CE IV collapses
"""

bear_rules = f"""
- ✅ **Enter only if PE volume > {vol_mult_30m_bear}× avg in first 30 mins**
- ✅ **Spot sustains BELOW trigger for {sustain_mins} mins**
- ✅ **ATM/Breakdown PE IV ≥ {iv_floor_bear}%** during trigger
- ❌ Avoid if **intraday PCR < {pcr_avoid_bear}** or PE IV collapses
"""

if direction in ("Both","Bullish"):
    draw_cards(df_bull, "Bullish Breakouts", bull_rules)
if direction in ("Both","Bearish"):
    draw_cards(df_bear, "Bearish Breakdowns", bear_rules)

# --------- Tables ---------
def tidy(df, side_label):
    x = df.copy()
    def fmt_ez(z):
        if not isinstance(z, tuple) or z[0] in [None, np.nan]:
            return ""
        return f"{z[0]} – {z[1]}" if z[1] not in [None, np.nan] else f"{z[0]}"
    x["Entry Zone"] = x["entry_zone"].apply(fmt_ez)
    x["Target"]     = x["target"].apply(lambda v: "" if v in [None, np.nan] else f"{v}")
    x["Stop"]       = x["stop"].apply(lambda v: "" if v in [None, np.nan] else f"{v}")
    x["R:R"]        = x["rr"].apply(lambda v: "" if v in [None, np.nan] else f"{v:.2f}")
    x["Prob%"]      = x["prob"].apply(lambda v: f"{v:.1f}%")
    x["Spot"]       = x["spot"].apply(lambda v: "" if v in [None, np.nan] else f"{v:.2f}")
    x["Trigger"]    = x["trigger"].apply(lambda v: "" if v in [None, np.nan] else f"{v:.2f}")
    x["PCR"]        = x["pcr"].apply(lambda v: "" if v in [None, np.nan] else f"{v:.2f}")
    x["MaxPain"]    = x["max_pain"].apply(lambda v: "" if v in [None, np.nan] else f"{v:.2f}")
    x["ATR"]        = x["atr"].apply(lambda v: "" if v in [None, np.nan] else f"{v:.2f}")
    keep = ["symbol","Spot","Trigger","Entry Zone","Target","Stop","R:R","Prob%","PCR","MaxPain",
            "ce_oi","pe_oi","ce_doi","pe_doi","ce_vol","pe_vol","ce_iv","pe_iv","ATR"]
    x = x[keep]
    x = x.rename(columns={"symbol":"Symbol","ce_oi":"CE_OI","pe_oi":"PE_OI",
                          "ce_doi":"CE_dOI","pe_doi":"PE_DOI","ce_vol":"VOL_CE",
                          "pe_vol":"VOL_PE","ce_iv":"IV_CE","pe_iv":"IV_PE"})
    st.subheader(f"📋 Full Results — {side_label}")
    st.dataframe(x, use_container_width=True, height=420)

if direction in ("Both","Bullish"):
    tidy(df_bull, "Bullish")
if direction in ("Both","Bearish"):
    tidy(df_bear, "Bearish")

# --------- Optional OI Walls Plot (top pick per side) ---------
def oi_plot(symbol):
    payload = get_chain(symbol, salt)
    if not payload:
        st.info("Option chain fetch failed for chart."); return
    df_ce, df_pe, _ = to_frames(payload)
    if df_ce is None or df_pe is None or df_ce.empty or df_pe.empty:
        st.info("No OI data available for chart."); return
    m = pd.merge(
        df_ce[["strikePrice","openInterest"]].rename(columns={"openInterest":"CE_OI"}),
        df_pe[["strikePrice","openInterest"]].rename(columns={"openInterest":"PE_OI"}),
        on="strikePrice", how="inner"
    ).sort_values("strikePrice")
    fig = go.Figure()
    fig.add_bar(x=m["strikePrice"], y=m["CE_OI"], name="CE OI")
    fig.add_bar(x=m["strikePrice"], y=m["PE_OI"], name="PE OI")
    fig.update_layout(barmode="overlay", xaxis_title="Strike", yaxis_title="Open Interest",
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

st.subheader("📊 OI Wall Snapshot")
cols = st.columns(2) if direction == "Both" else st.columns(1)
if direction in ("Both","Bullish") and not df_bull.empty:
    with cols[0]:
        st.caption(f"Bullish top: {df_bull.iloc[0]['symbol']}")
        oi_plot(df_bull.iloc[0]['symbol'])
if direction in ("Both","Bearish") and not df_bear.empty:
    with (cols[1] if direction=="Both" else cols[0]):
        st.caption(f"Bearish top: {df_bear.iloc[0]['symbol']}")
        oi_plot(df_bear.iloc[0]['symbol'])

st.caption("⚠️ Educational tool. Markets carry risk; do your own due diligence.")

# ---------------- Snapshots & ML Comparison ----------------
def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def build_snapshot_df(df_bull: pd.DataFrame, df_bear: pd.DataFrame) -> pd.DataFrame:
    def _pick(df: pd.DataFrame, side: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=[
                "symbol","side","spot","trigger","prob","rr","pcr","target","stop",
                "ce_oi","pe_oi","ce_doi","pe_doi","ce_vol","pe_vol","ce_iv","pe_iv","max_pain","atr"
            ])
        cols = {
            "symbol":"symbol", "spot":"spot", "trigger":"trigger", "prob":"prob", "rr":"rr",
            "pcr":"pcr", "target":"target", "stop":"stop",
            "ce_oi":"ce_oi","pe_oi":"pe_oi","ce_doi":"ce_doi","pe_doi":"pe_doi",
            "ce_vol":"ce_vol","pe_vol":"pe_vol","ce_iv":"ce_iv","pe_iv":"pe_iv",
            "max_pain":"max_pain","atr":"atr"
        }
        x = df.copy()
        for k in cols:
            if k not in x.columns: x[k] = np.nan
        y = x[list(cols.keys())].rename(columns=cols)
        y.insert(1, "side", side)
        return y

    sb = _pick(df_bull, "bull")
    sr = _pick(df_bear, "bear")
    snap = pd.concat([sb, sr], ignore_index=True)
    snap["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return snap

def save_snapshot_csv(snap: pd.DataFrame) -> str:
    tag = _now_tag()
    path = os.path.join(SNAP_DIR, f"snapshot_{tag}.csv")
    snap.to_csv(path, index=False)
    return path

def list_snapshots() -> list:
    return sorted(glob.glob(os.path.join(SNAP_DIR, "snapshot_*.csv")))

def load_latest_snapshot() -> Tuple[pd.DataFrame, str] | Tuple[None, None]:
    files = list_snapshots()
    if not files:
        return None, None
    last = files[-1]
    try:
        df = pd.read_csv(last)
        return df, last
    except Exception:
        return None, None

def compare_snapshots(curr: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    keys = ["symbol","side"]
    num_cols = ["prob","rr","pcr","spot","trigger","target","stop",
                "ce_oi","pe_oi","ce_doi","pe_doi","ce_vol","pe_vol","ce_iv","pe_iv","max_pain","atr"]
    a = curr[keys + num_cols].copy()
    b = prev[keys + num_cols].copy()
    for c in num_cols:
        a[c] = pd.to_numeric(a[c], errors="coerce")
        b[c] = pd.to_numeric(b[c], errors="coerce")
    merged = pd.merge(a, b, on=keys, suffixes=("_curr","_prev"), how="inner")
    for c in num_cols:
        merged[f"{c}_Δ"] = merged[f"{c}_curr"] - merged[f"{c}_prev"]
    components = ["prob_Δ","rr_Δ","ce_doi_Δ","ce_vol_Δ","pe_doi_Δ","pe_vol_Δ","pcr_Δ"]
    scaler = StandardScaler()
    Z = scaler.fit_transform(merged[[c for c in components if c in merged.columns]].fillna(0.0))
    merged["change_score"] = Z.sum(axis=1)
    keep = keys + [f"{c}_Δ" for c in ["prob","rr","pcr","ce_doi","pe_doi","ce_vol","pe_vol"]] + ["change_score"]
    keep = [c for c in keep if c in merged.columns]
    out = merged[keep].sort_values("change_score", ascending=False)
    return out

def ml_anomaly_scores(history_files: list, current_snap: pd.DataFrame) -> pd.DataFrame | None:
    if len(history_files) < 5:
        return None
    frames = []
    for f in history_files[:-1]:
        try:
            df = pd.read_csv(f)
            frames.append(df)
        except Exception:
            pass
    if not frames:
        return None
    hist = pd.concat(frames, ignore_index=True)
    feats = ["prob","rr","pcr","ce_doi","pe_doi","ce_vol","pe_vol","ce_iv","pe_iv","atr"]
    H = hist[["symbol","side"] + feats].copy()
    for c in feats:
        H[c] = pd.to_numeric(H[c], errors="coerce").fillna(0.0)
    C = current_snap[["symbol","side"] + feats].copy()
    for c in feats:
        C[c] = pd.to_numeric(C[c], errors="coerce").fillna(0.0)
    scaler = StandardScaler()
    Hs = scaler.fit_transform(H[feats])
    Cs = scaler.transform(C[feats])
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=42)
    iso.fit(Hs)
    scores = iso.score_samples(Cs)
    anomaly = -scores
    out = C[["symbol","side"]].copy()
    out["anomaly_score"] = anomaly
    return out.sort_values("anomaly_score", ascending=False)

st.markdown("---")
st.subheader("🧿 Snapshots & ML Comparison")

snap_now = build_snapshot_df(df_bull, df_bear)

c1, c2, c3 = st.columns([1,1,2])
with c1:
    if st.button("💾 Save Snapshot (CSV)"):
        p = save_snapshot_csv(snap_now)
        st.success(f"Snapshot saved: {p}")

with c2:
    if st.button("🔎 Compare with Last Snapshot"):
        prev, path = load_latest_snapshot()
        if prev is None:
            st.info("No previous snapshot found. Save one first.")
        else:
            comp = compare_snapshots(snap_now, prev)
            st.caption(f"Comparing CURRENT vs LAST: {path}")
            if comp.empty:
                st.info("No overlapping symbols/sides to compare.")
            else:
                st.dataframe(comp, use_container_width=True, height=360)

with c3:
    files = list_snapshots()
    if len(files) >= 5:
        ml = ml_anomaly_scores(files, snap_now)
        if ml is not None and not ml.empty:
            st.write("🤖 **ML Drift/Anomaly Check** (higher = more unusual vs history)")
            st.dataframe(ml, use_container_width=True, height=360)
        else:
            st.caption("Need ≥5 snapshots for ML anomaly check.")
    else:
        st.caption("Save a few snapshots (≥5) to enable ML drift detection.")

# ---------------- ML: Auto-train to predict next move ----------------
ML_FEATURES = [
    "prob","rr","pcr","ce_doi","pe_doi","ce_vol","pe_vol","ce_iv","pe_iv","atr",
    "spot","trigger","target","stop",
]

def _nse_to_yf(sym: str) -> str:
    return f"{sym}.NS"

def _yahoo_close(symbol_yf: str, date_str: str):
    try:
        end = pd.to_datetime(date_str) + pd.Timedelta(days=2)
        start = pd.to_datetime(date_str) - pd.Timedelta(days=7)
        df = yf.Ticker(symbol_yf).history(start=start, end=end, interval="1d")
        if df.empty: return np.nan
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        target_day = pd.to_datetime(date_str)
        df = df[df["Date"] <= target_day]
        if df.empty: return np.nan
        return float(df.iloc[-1]["Close"])
    except Exception:
        return np.nan

def _build_trainset_from_snapshots(snapshot_files: list, fwd_days: int = 1) -> pd.DataFrame:
    if not snapshot_files: return pd.DataFrame()
    rows = []
    for fp in snapshot_files:
        try:
            snap = pd.read_csv(fp)
            if snap.empty: continue
            # ensure fields
            for c in ML_FEATURES:
                if c not in snap: snap[c] = np.nan
                snap[c] = pd.to_numeric(snap[c], errors="coerce")
            snap["t"] = pd.to_datetime(snap["timestamp"], errors="coerce").dt.tz_localize(None)
            snap["d"] = snap["t"].dt.date.astype("str")
            snap["yf"] = snap["symbol"].astype(str).apply(_nse_to_yf)
            snap["close_t"] = snap.apply(lambda r: _yahoo_close(r["yf"], r["d"]), axis=1)
            snap["d_fwd"] = (pd.to_datetime(snap["d"]) + pd.Timedelta(days=fwd_days)).dt.date.astype("str")
            snap["close_fwd"] = snap.apply(lambda r: _yahoo_close(r["yf"], r["d_fwd"]), axis=1)
            snap["ret_fwd"] = (snap["close_fwd"] - snap["close_t"]) / snap["close_t"]
            snap["y_up"] = (snap["ret_fwd"] > 0.002).astype(int)  # > +0.2% = "Up"
            use = snap[["symbol","side","y_up","ret_fwd"] + ML_FEATURES].copy()
            use["side_is_bull"] = (use["side"].astype(str).str.lower()=="bull").astype(int)
            rows.append(use)
        except Exception:
            continue
    if not rows: return pd.DataFrame()
    data = pd.concat(rows, ignore_index=True)
    for c in ML_FEATURES:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    return data.dropna(subset=["y_up"])

def _train_ml_model(train_df: pd.DataFrame):
    """
    Train classifier when there are 2 classes; otherwise fallback to regression.
    Returns a dict: {"mode": "clf"|"reg", "pipe": Pipeline, "auc": float|None}
    """
    if train_df is None or train_df.empty:
        return {"mode": None, "pipe": None, "auc": None}

    feature_cols = ML_FEATURES + ["side_is_bull"]
    X = train_df[feature_cols].fillna(0.0)
    y_clf = train_df["y_up"].astype(int)
    y_reg = pd.to_numeric(train_df["ret_fwd"], errors="coerce").fillna(0.0)

    uniq = np.unique(y_clf)
    has_two_classes = uniq.size >= 2

    if has_two_classes:
        # ---- Classifier path ----
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y_clf, test_size=0.25, random_state=42, stratify=y_clf
            )
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y_clf, test_size=0.25, random_state=42
            )
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("gb", GradientBoostingClassifier(random_state=42))
        ])
        pipe.fit(X_tr, y_tr)
        try:
            auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
        except Exception:
            auc = None
        return {"mode": "clf", "pipe": pipe, "auc": auc}

    # ---- Regression fallback ----
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(random_state=42))
    ])
    pipe.fit(X, y_reg)
    return {"mode": "reg", "pipe": pipe, "auc": None}

def _predict_current(model: dict, current_snap: pd.DataFrame, thr: float = 0.55) -> pd.DataFrame:
    """
    Predict next move for current snapshot rows.
    - If classifier: use predict_proba
    - If regressor: map predicted return to pseudo-prob via logistic
    """
    if model is None or model.get("pipe") is None or current_snap is None or current_snap.empty:
        return pd.DataFrame()

    mode = model.get("mode")
    pipe = model.get("pipe")

    cur = current_snap.copy()
    for c in ML_FEATURES:
        if c not in cur: cur[c] = 0.0
        cur[c] = pd.to_numeric(cur[c], errors="coerce").fillna(0.0)
    cur["side_is_bull"] = (cur["side"].astype(str).str.lower()=="bull").astype(int)

    X = cur[ML_FEATURES + ["side_is_bull"]].fillna(0.0)

    if mode == "clf":
        prob_up = pipe.predict_proba(X)[:, 1]
    else:
        pred_ret = pipe.predict(X)
        scale = 0.004  # logistic scale ~0.4% → ~0.73 prob
        prob_up = 1.0 / (1.0 + np.exp(-(pred_ret / max(1e-6, scale))))

    cur["prob_up"] = np.round(prob_up, 3)
    cur["prob_down"] = np.round(1.0 - cur["prob_up"], 3)
    cur["action"] = np.where(
        cur["side_is_bull"]==1,
        np.where(cur["prob_up"]>=thr, "LONG", "WAIT"),
        np.where(cur["prob_down"]>=thr, "SHORT", "WAIT")
    )
    cur["confidence"] = np.where(cur["side_is_bull"]==1, cur["prob_up"], cur["prob_down"])
    cur["confidence"] = np.round(cur["confidence"], 3)
    return cur[["symbol","side","prob_up","prob_down","action","confidence"]]

# ---------------- UI: Auto-train & Predict Next Move ----------------
st.markdown("---")
st.subheader("🤖 Auto-Train & Predict Next Move")

all_snaps = list_snapshots()
train_df = _build_trainset_from_snapshots(all_snaps, fwd_days=int(fwd_days))

if train_df.empty:
    st.info("Not enough snapshot history to train yet. Save several snapshots over time, then try again.")
else:
    model = _train_ml_model(train_df)
    pipe, auc = model.get("pipe"), model.get("auc")
    if pipe is None:
        st.warning("Training failed to produce a model.")
    else:
        if model.get("mode") == "clf" and auc is not None:
            st.caption(f"Validation AUC (holdout): **{auc:.3f}** — classifier")
        elif model.get("mode") == "reg":
            st.caption("Regression fallback in use (single-class labels). Showing pseudo-probabilities from predicted returns.")

        snap_now_ml = build_snapshot_df(df_bull, df_bear)
        preds = _predict_current(model, snap_now_ml, thr=float(decision_thr))

        if preds.empty:
            st.info("No current rows to score.")
        else:
            st.dataframe(preds.sort_values(["action","confidence"], ascending=[True, False]),
                         use_container_width=True, height=360)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Suggested LONGs (Bull side)**")
                long_picks = preds[preds["action"]=="LONG"].sort_values("confidence", ascending=False)
                st.table(long_picks[["symbol","confidence"]].head(10) if not long_picks.empty else pd.DataFrame())
            with c2:
                st.markdown("**Suggested SHORTs (Bear side)**")
                short_picks = preds[preds["action"]=="SHORT"].sort_values("confidence", ascending=False)
                st.table(short_picks[["symbol","confidence"]].head(10) if not short_picks.empty else pd.DataFrame())
