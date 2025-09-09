# === app.py — NSE Option Chain Dashboard + Inline Kite Login (single-file) ===
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from nsepython import nse_optionchain_scrapper
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timezone
from dataclasses import dataclass
import json, os
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit Config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="NSE Option Chain Dashboard+", layout="wide")
st.title("📊 NSE Option Chain Dashboard — Pro Edition (Auto-Refresh)")

# ──────────────────────────────────────────────────────────────────────────────
# ⚙️ Controls
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")
    refresh_interval = st.number_input(
        "🔄 Auto-Refresh (seconds)", min_value=10, max_value=300, value=30, step=5
    )
    window = st.slider("EMA window for PCR smoothing", min_value=3, max_value=15, value=7)
    focus_strikes = st.slider("ATM window for charts (± strikes)", min_value=3, max_value=15, value=8)
    table_band = st.slider("Table rows around ATM (± strikes)", 3, 15, 5)
    show_heat = st.checkbox("Show PCR heat bands", value=True)

    st.divider()
    st.subheader("🤖 Signal Settings")
    conf_threshold = st.slider("Confidence threshold", 0.50, 0.90, 0.60, 0.01)
    persist_model = st.checkbox("💾 Persist learning to disk", value=True)
    run_bt = st.button("▶️ Backtest (this session)")

    st.divider()
    st.subheader("💼 Position Sizing")
    acct_equity = st.number_input("Account size (₹)", min_value=10000, value=200000, step=5000)
    risk_pct = st.slider("Risk per trade (%)", 0.5, 5.0, 1.0, 0.1)
    max_expo_pct = st.slider("Max exposure per trade (%)", 1.0, 25.0, 10.0, 0.5)
    lot_size = st.number_input("Lot size (NIFTY=50, BANKNIFTY=15)", min_value=1, value=50, step=1)

    st.divider()
    st.subheader("🪁 Kite")
    use_kite = st.checkbox(
        "Use Kite (login required)",
        value=True,
        help="Reads kite_session.json and fetches Spot via Kite LTP; falls back to NSE data if unavailable."
    )
    place_order_ready = st.checkbox("Enable order preview section", value=True)

st_autorefresh(interval=refresh_interval * 1000, key="refreshdata")

symbol_type = st.radio("Select Type", ["Index", "Stock"], horizontal=True)
if symbol_type == "Index":
    symbol = st.selectbox("Select Index", ["NIFTY", "BANKNIFTY"])
else:
    symbol = st.text_input("Enter Stock Symbol (e.g. RELIANCE, TCS, INFY)")

expiry_date = None
spot_price = None

# ──────────────────────────────────────────────────────────────────────────────
# 🪁 Zerodha Kite — inline login + client accessor
# ──────────────────────────────────────────────────────────────────────────────
KITE_SESSION_PATH = Path("kite_session.json")

def _load_creds():
    # Prefer Streamlit Secrets; fallback to env vars
    api_key = st.secrets.get("KITE_API_KEY", os.getenv("KITE_API_KEY", ""))
    api_secret = st.secrets.get("KITE_API_SECRET", os.getenv("KITE_API_SECRET", ""))
    redirect_url = st.secrets.get("KITE_REDIRECT_URL", os.getenv("KITE_REDIRECT_URL", "http://localhost:8501/"))
    return api_key, api_secret, redirect_url

def _save_session(api_key: str, access_token: str):
    KITE_SESSION_PATH.write_text(json.dumps({
        "api_key": api_key,
        "access_token": access_token,
        "saved_at": datetime.now(timezone.utc).isoformat()
    }, indent=2))

def _delete_session():
    try:
        KITE_SESSION_PATH.unlink(missing_ok=True)
    except Exception:
        pass

def _session_exists():
    return KITE_SESSION_PATH.exists()

@st.cache_resource
def _kite_from_session():
    """Construct KiteConnect if session file exists & is valid."""
    if not _session_exists():
        return None
    try:
        from kiteconnect import KiteConnect
        data = json.loads(KITE_SESSION_PATH.read_text())
        kite = KiteConnect(api_key=data["api_key"])
        kite.set_access_token(data["access_token"])
        return kite
    except Exception:
        return None

def get_kite():
    """Public accessor used by the rest of the app."""
    if not use_kite:
        return None
    return _kite_from_session()

def kite_index_ltp_symbol(sym: str) -> str:
    s = sym.upper()
    if s == "NIFTY": return "NSE:NIFTY 50"
    if s == "BANKNIFTY": return "NSE:NIFTY BANK"
    return f"NSE:{s}"

def get_spot_from_kite(sym: str) -> float | None:
    kite = get_kite()
    if not kite:
        return None
    try:
        q = kite.ltp([kite_index_ltp_symbol(sym)])
        return float(list(q.values())[0]["last_price"])
    except Exception:
        return None

# ── Sidebar login UI (inline)
with st.sidebar:
    st.subheader("🪁 Kite Login")
    api_key, api_secret, redirect_url = _load_creds()

    if _session_exists():
        st.success("Kite: Logged in")
        if st.button("Logout"):
            _delete_session()
            st.cache_resource.clear()  # clears _kite_from_session
            st.rerun()
    else:
        if not api_key or not api_secret:
            st.error("Add KITE_API_KEY and KITE_API_SECRET in Secrets or env.")
        else:
            try:
                from kiteconnect import KiteConnect
                kite_tmp = KiteConnect(api_key=api_key)
                login_url = kite_tmp.login_url()  # creates v=3 URL
            except Exception as e:
                login_url = None
                st.error(f"Could not build login URL: {e}")

            if login_url:
                st.markdown(f"**Redirect URL (must match in Kite app):** `{redirect_url}`")
                st.link_button("🔑 Login with Kite", login_url, use_container_width=True)
                st.caption("After login, Zerodha redirects back with `?request_token=...`")

                # Auto-capture request_token from URL; allow manual paste too
                qp = st.query_params
                req_token = qp.get("request_token", [None])
                request_token = req_token[0] if isinstance(req_token, list) else req_token
                request_token = st.text_input("Paste request_token (auto if redirected):", value=request_token or "")

                if st.button("Exchange token"):
                    if not request_token:
                        st.warning("Paste the request_token first.")
                    else:
                        try:
                            data = kite_tmp.generate_session(request_token, api_secret=api_secret)
                            _save_session(api_key, data["access_token"])
                            st.success("✅ Logged in! Session saved to kite_session.json.")
                            st.query_params.clear()   # clean URL
                            st.cache_resource.clear() # rebuild _kite_from_session
                            st.rerun()
                        except Exception as e:
                            st.error(f"Token exchange failed: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Order preview helper (non-invasive)
# ──────────────────────────────────────────────────────────────────────────────
def maybe_place_order_ui(suggest_side: str, tradingsymbol: str, qty: int, sl: float, tp: float):
    """Non-invasive order preview (does NOT place unless you un-comment)."""
    if not place_order_ready or suggest_side not in ("CALL", "PUT"):
        return
    kite = get_kite()
    if not kite:
        st.info("Login to Kite to enable order preview.")
        return
    st.subheader("🧪 Order preview (Kite)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write(f"Symbol: `{tradingsymbol}`")
    with c2:
        st.write(f"Side: **BUY {suggest_side}**")
    with c3:
        st.write(f"Qty: **{qty}**")

    st.write(f"Proposed SL: **₹{sl}**, TP: **₹{tp}** (manage on your terminal)")

    # If you want to truly place an order, un-comment below (use carefully):
    # if st.button("Place MARKET order now"):
    #     order_id = kite.place_order(
    #         variety=kite.VARIETY_REGULAR,
    #         exchange=kite.EXCHANGE_NFO if "NIFTY" in tradingsymbol or "BANKNIFTY" in tradingsymbol else kite.EXCHANGE_NSE,
    #         tradingsymbol=tradingsymbol,
    #         transaction_type=kite.TRANSACTION_TYPE_BUY,
    #         quantity=int(qty),
    #         product=kite.PRODUCT_MIS,   # or NRML
    #         order_type=kite.ORDER_TYPE_MARKET
    #     )
    #     st.success(f"Order placed: {order_id}")

# ──────────────────────────────────────────────────────────────────────────────
# Caching layer (reduces flicker + API calls)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=20)
def fetch_option_chain(sym: str):
    try:
        data = nse_optionchain_scrapper(sym)
        return data, datetime.now(timezone.utc).isoformat()
    except Exception as e:
        return None, str(e)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def to_df(data, expiry):
    recs = [d for d in data["records"]["data"] if d.get("expiryDate") == expiry]
    ce = [d["CE"] for d in recs if "CE" in d]
    pe = [d["PE"] for d in recs if "PE" in d]
    df_ce = pd.DataFrame(ce); df_pe = pd.DataFrame(pe)
    for df in (df_ce, df_pe):
        for c in ["strikePrice","openInterest","changeinOpenInterest","impliedVolatility",
                  "totalTradedVolume","totalBuyQuantity","totalSellQuantity","lastPrice"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    df_ce = df_ce.sort_values("strikePrice").reset_index(drop=True)
    df_pe = df_pe.sort_values("strikePrice").reset_index(drop=True)
    common = np.intersect1d(df_ce["strikePrice"], df_pe["strikePrice"]).astype(float)
    df_ce = df_ce[df_ce["strikePrice"].isin(common)].reset_index(drop=True)
    df_pe = df_pe[df_pe["strikePrice"].isin(common)].reset_index(drop=True)
    return df_ce, df_pe

def compute_oi_table(df_ce, df_pe):
    df_oi = pd.DataFrame({
        "Strike": df_ce["strikePrice"].astype(int),
        "CE_OI": df_ce["openInterest"].fillna(0),
        "PE_OI": df_pe["openInterest"].fillna(0),
        "CE_dOI": df_ce["changeinOpenInterest"].fillna(0),
        "PE_dOI": df_pe["changeinOpenInterest"].fillna(0),
        "IV_CE": df_ce["impliedVolatility"].fillna(0),
        "IV_PE": df_pe["impliedVolatility"].fillna(0),
        "LTP_CE": df_ce["lastPrice"].fillna(0),
        "LTP_PE": df_pe["lastPrice"].fillna(0),
        "VOL_CE": df_ce.get("totalTradedVolume", pd.Series([0]*len(df_ce))).fillna(0),
        "VOL_PE": df_pe.get("totalTradedVolume", pd.Series([0]*len(df_pe))).fillna(0),
    })
    df_oi["PCR"] = np.where(df_oi["CE_OI"] > 0, df_oi["PE_OI"] / df_oi["CE_OI"], np.nan)
    return df_oi

def ema(series: pd.Series, span: int = 7):
    return series.ewm(span=span, adjust=False, min_periods=max(2, span//2)).mean()

def robust_max_pain(df_oi: pd.DataFrame):
    strikes = df_oi["Strike"].values
    ce_oi = df_oi["CE_OI"].values; pe_oi = df_oi["PE_OI"].values
    losses = []
    for K in strikes:
        put_loss  = np.sum(np.maximum(K - strikes, 0) * pe_oi)
        call_loss = np.sum(np.maximum(strikes - K, 0) * ce_oi)
        losses.append(put_loss + call_loss)
    return int(strikes[int(np.argmin(losses))])

def bias_score(df_oi: pd.DataFrame):
    def nz_norm(x):
        if x.max() - x.min() == 0: return x*0 + 0.5
        return (x - x.min())/(x.max()-x.min())
    pe_oi_n = nz_norm(df_oi["PE_OI"]); ce_oi_n = nz_norm(df_oi["CE_OI"])
    pcr = df_oi["PCR"].replace([np.inf,-np.inf], np.nan).fillna(1.0)
    pcr_n = nz_norm(pcr.clip(0,5))
    doi_edge = nz_norm(df_oi["PE_dOI"] - df_oi["CE_dOI"])
    iv_skew = nz_norm(df_oi["IV_PE"] - df_oi["IV_CE"])
    blend = 0.35*pcr_n.mean() + 0.25*doi_edge.mean() + 0.25*(pe_oi_n.mean()-ce_oi_n.mean()+0.5) + 0.15*iv_skew.mean()
    return int(np.clip(100*blend, 0, 100))

def nearest_levels(df_oi: pd.DataFrame, spot: float):
    strike_vals = df_oi["Strike"].values
    atm = int(strike_vals[np.argmin(np.abs(strike_vals - spot))])
    below = df_oi[df_oi["Strike"] <= atm]; above = df_oi[df_oi["Strike"] >= atm]
    sup = below.sort_values(["PE_OI","Strike"], ascending=[False, False]).head(1).iloc[0]
    res = above.sort_values(["CE_OI","Strike"], ascending=[False, True]).head(1).iloc[0]
    return atm, int(sup["Strike"]), int(res["Strike"]), sup, res

def style_max_cells(s: pd.Series, color="#2DD4BF", bold=True, border=True, tol=1e-9):
    numeric = pd.to_numeric(s, errors="coerce")
    vmax = numeric.max()
    out = []
    for _, val in zip(s, numeric):
        if pd.notna(val) and pd.notna(vmax) and abs(val - vmax) <= tol:
            css = f"background-color:{color};"
            if bold:  css += " font-weight:700;"
            if border: css += " border:1px solid #10B981;"
            out.append(css)
        else:
            out.append("")
    return out

# ──────────────────────────────────────────────────────────────────────────────
#                   AI / SIGNAL ENGINE
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class SignalOutput:
    side: str
    confidence: float
    reason: str
    entry_strike: int | None
    sl: float | None
    tp: float | None
    qty: int | None = None

def nz(x, dv=0.0):
    return dv if (x is None or (isinstance(x, float) and np.isnan(x))) else x

def slope(y: pd.Series, lookback: int = 5):
    y = pd.to_numeric(y, errors="coerce").fillna(method="ffill").fillna(method="bfill")
    n = min(lookback, len(y))
    if n < 3: return 0.0
    x = np.arange(n); yy = y.iloc[-n:].values
    m, _ = np.linalg.lstsq(np.vstack([x,np.ones(n)]).T, yy, rcond=None)[0]
    return float(m)

def extract_features(df_oi: pd.DataFrame, spot: float | None, mp: int, window:int) -> dict:
    feats={}
    feats["pcr_now"]=float(df_oi["PCR"].replace([np.inf,-np.inf],np.nan).fillna(1.0).median())
    feats["pcr_ema_mid"]=float(df_oi["PCR_EMA"].replace([np.inf,-np.inf],np.nan).fillna(method="ffill").fillna(method="bfill").median())
    feats["pcr_ema_slope"]=slope(df_oi["PCR_EMA"],lookback=7)
    feats["doi_edge"]=float((df_oi["PE_dOI"]-df_oi["CE_dOI"]).sum())
    feats["iv_skew"]=float((df_oi["IV_PE"]-df_oi["IV_CE"]).median())
    feats["oi_diff_total"]=float(df_oi["PE_OI"].sum()-df_oi["CE_OI"].sum())
    feats["dist_to_maxpain"]=float(spot-mp) if spot else 0.0
    try:
        atm,supK,resK,supRow,resRow=nearest_levels(df_oi, spot if spot else df_oi["Strike"].median())
        feats["support_strength"]=float(nz(supRow["PE_OI"]))
        feats["resistance_strength"]=float(nz(resRow["CE_OI"]))
        feats["atm"]=int(atm)
    except Exception:
        feats["support_strength"]=0.0; feats["resistance_strength"]=0.0
        feats["atm"]=int(df_oi["Strike"].median())
    def prct_rank(s):
        s=pd.Series(s).astype(float)
        if s.max()==s.min(): return 0.5
        return float((s.tail(1).rank(pct=True)).iloc[0])
    feats["doi_edge_pr"]=prct_rank(df_oi["PE_dOI"]-df_oi["CE_dOI"])
    feats["pcr_ema_pr"]=prct_rank(df_oi["PCR_EMA"].clip(0,3.0))
    return feats

def logistic_score(feats: dict, w: dict) -> float:
    z = (
        w["bias"] +
        w["pcr_ema_mid"]*feats["pcr_ema_mid"] +
        w["pcr_slope"]   *feats["pcr_ema_slope"] +
        w["doi_edge"]    *(feats["doi_edge"]/1e5) +
        w["iv_skew"]     *feats["iv_skew"] +
        w["oi_diff"]     *(feats["oi_diff_total"]/1e6) +
        w["mp_pull"]     *(-abs(feats["dist_to_maxpain"])/500.0) +
        w["supp_vs_res"] *((feats["support_strength"]-feats["resistance_strength"])/1e6) +
        w["doi_edge_pr"] *(feats["doi_edge_pr"]-0.5)*2.0 +
        w["pcr_ema_pr"]  *(feats["pcr_ema_pr"] -0.5)*2.0
    )
    return 1.0/(1.0+np.exp(-z))

def default_weights():
    return {
        "bias":0.0,"pcr_ema_mid":0.8,"pcr_slope":2.2,"doi_edge":0.8,"iv_skew":-0.6,
        "oi_diff":0.7,"mp_pull":0.4,"supp_vs_res":0.9,"doi_edge_pr":0.8,"pcr_ema_pr":0.9
    }

def option_step_for_symbol(sym: str) -> int:
    s=sym.upper()
    if s=="NIFTY": return 50
    if s=="BANKNIFTY": return 100
    return 50

def decide_action(prob_bull: float, feats: dict, df_oi: pd.DataFrame, spot: float | None, sym: str,
                  acct_equity: float, risk_pct: float, max_expo_pct: float, lot_size: int) -> SignalOutput:
    conf = abs(prob_bull - 0.5) * 2.0
    side, reason, entry_strike, sl, tp, qty = "WAIT","",None,None,None,None
    step = option_step_for_symbol(sym)
    atm = feats.get("atm", int(df_oi["Strike"].median()))

    if prob_bull >= 0.55:
        side="CALL"; entry_strike=int(np.floor(atm/step)*step)
        reason="Rising PCR & PE ΔOI dominance; support > resistance."
    elif prob_bull <= 0.45:
        side="PUT"; entry_strike=int(np.ceil(atm/step)*step)
        reason="Falling PCR or CE ΔOI dominance; IV skew to puts; resistance > support."
    else:
        reason="Mixed signals around neutrality."

    if side in ("CALL","PUT"):
        leg = "LTP_CE" if side=="CALL" else "LTP_PE"
        row = df_oi.iloc[(df_oi["Strike"] - entry_strike).abs().argmin()]
        prem = float(nz(row.get(leg, 0.0), 0.0))
        if prem <= 0: prem = max(5.0, (spot or atm)*0.002)

        rr_tp = 0.40 + 0.30*conf
        rr_sl = 0.25 - 0.10*conf
        tp = round(prem*(1.0+rr_tp), 2)
        sl = round(prem*(1.0-rr_sl), 2)

        risk_rupees = acct_equity * (risk_pct/100.0)
        per_lot_risk = max(prem - sl, 0.01) * lot_size
        max_lots_by_risk = int(np.floor(risk_rupees / per_lot_risk)) if per_lot_risk>0 else 0
        max_lots_by_expo = int(np.floor((acct_equity*(max_expo_pct/100.0)) / (prem*lot_size))) if prem>0 else 0
        lots = max(0, min(max_lots_by_risk, max_lots_by_expo))
        qty = lots * lot_size
        if lots <= 0:
            reason += " | ⚠️ Position too large for risk/exposure—consider smaller risk % or more liquid strike."

    return SignalOutput(side=side, confidence=float(conf), reason=reason, entry_strike=entry_strike, sl=sl, tp=tp, qty=qty)

def format_confidence_badge(sig: SignalOutput):
    if sig.side=="CALL": color="#d1f7c4"; emoji="🟢"
    elif sig.side=="PUT": color="#ffd6d6"; emoji="🔴"
    else: color="#fff2b2"; emoji="🟡"
    return f"""
    <div style="background:{color};padding:12px;border-radius:12px;text-align:center;font-size:18px;font-weight:700">
      {emoji} Suggested: {sig.side} • Confidence: {sig.confidence:.0%}
      <div style="font-size:14px;font-weight:600;margin-top:6px;">{sig.reason}</div>
    </div>
    """

def online_update_weights(weights: dict, prob_bull: float, last_spot: float | None, current_spot: float | None, lr: float = 0.05):
    if last_spot is None or current_spot is None: return weights
    y = 1.0 if current_spot > last_spot else 0.0
    grad = (prob_bull - y)
    w = weights.copy(); w["bias"] -= lr*grad
    return w

# ---------- Persistence ----------
WEIGHTS_PATH = "model_weights.json"

def load_weights_from_disk(path=WEIGHTS_PATH):
    if os.path.exists(path):
        try:
            with open(path,"r") as f:
                w = json.load(f)
                base = default_weights()
                base.update({k: float(v) for k,v in w.items() if k in base})
                return base
        except Exception:
            pass
    return default_weights()

def save_weights_to_disk(w: dict, path=WEIGHTS_PATH):
    try:
        with open(path,"w") as f:
            json.dump(w, f, indent=2)
        return True
    except Exception:
        return False

# ──────────────────────────────────────────────────────────────────────────────
# Session state for learning & backtesting
# ──────────────────────────────────────────────────────────────────────────────
if "sig_weights" not in st.session_state:
    st.session_state.sig_weights = load_weights_from_disk() if persist_model else default_weights()
if "prev_spot" not in st.session_state:
    st.session_state.prev_spot = None
if "snapshots" not in st.session_state:
    st.session_state.snapshots = []

with st.sidebar:
    colA, colB = st.columns(2)
    if colA.button("💾 Save Weights"):
        ok = save_weights_to_disk(st.session_state.sig_weights)
        st.toast("Saved model weights." if ok else "Failed to save weights.", icon="💾" if ok else "⚠️")
    if colB.button("♻️ Reset Weights"):
        st.session_state.sig_weights = default_weights()
        st.toast("Weights reset to defaults.", icon="♻️")

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if symbol:
    data, meta = fetch_option_chain(symbol.upper())
    if data is None:
        st.error(f"❌ Could not fetch data: {meta}")
    else:
        expiry_list = data["records"].get("expiryDates", [])
        # Prefer Kite spot if available; else fallback to NSEPython response
        kite_spot = get_spot_from_kite(symbol) if use_kite else None
        spot_price = kite_spot if kite_spot is not None else data["records"].get("underlyingValue")
        if not expiry_list:
            st.warning("⚠️ No expiry list returned.")
        expiry_date = st.selectbox("Select Expiry Date", expiry_list)
        st.metric("📌 Spot", f"{spot_price}")
        st.caption(f"Last updated (UTC): {meta}" + (" • via Kite LTP" if kite_spot is not None else " • via NSE"))

if symbol and expiry_date:
    try:
        data, _ = fetch_option_chain(symbol.upper())
        df_ce, df_pe = to_df(data, expiry_date)
        if df_ce.empty or df_pe.empty:
            st.warning("No option rows for chosen expiry.")
        else:
            df_oi = compute_oi_table(df_ce, df_pe)
            df_oi["PCR_EMA"] = ema(df_oi["PCR"], span=window)

            overall_pcr = (df_oi["PE_OI"].sum() / df_oi["CE_OI"].replace(0, np.nan).sum())
            mp = robust_max_pain(df_oi); score = bias_score(df_oi)
            overall_pcr_safe = float(pd.to_numeric(pd.Series([overall_pcr]), errors="coerce").fillna(1.0).iloc[0])
            score_safe = int(nz(score, 50))

            if score_safe >= 60: tag=("🟢 Bullish Bias","#d1f7c4")
            elif score_safe <= 40: tag=("🔴 Bearish Bias","#ffd6d6")
            else: tag=("🟡 Neutral Bias","#fff2b2")
            st.markdown(
                f"""
                <div style="background:{tag[0]};padding:12px;border-radius:12px;text-align:center;font-size:18px;font-weight:700">
                {tag[0]} • 📊 Overall PCR: <b>{overall_pcr_safe:.2f}</b> • 🎯 Max Pain: <b>{mp}</b> • 🧮 Bias Score: <b>{score_safe}/100</b>
                </div>
                """, unsafe_allow_html=True
            )

            # ---- Signal Engine ----
            feats = extract_features(df_oi, spot_price, mp, window)
            prob_bull = logistic_score(feats, st.session_state.sig_weights)
            signal = decide_action(prob_bull, feats, df_oi, spot_price, symbol,
                                   acct_equity, risk_pct, max_expo_pct, lot_size)

            # Confidence gate
            gated_side = signal.side if signal.confidence >= conf_threshold else "WAIT"
            st.markdown(format_confidence_badge(
                SignalOutput(gated_side, signal.confidence, signal.reason, signal.entry_strike, signal.sl, signal.tp, signal.qty)
            ), unsafe_allow_html=True)

            if gated_side in ("CALL","PUT"):
                if signal.qty and signal.qty>0:
                    st.info(f"➡️ Action: **{gated_side}** {signal.entry_strike}  | Qty **{signal.qty}**  | SL **₹{signal.sl}**  | TP **₹{signal.tp}**")
                    # ── OPTIONAL: show order preview for the recommended strike
                    if place_order_ready:
                        # naive example tradingsymbol (adjust to your expiry code scheme):
                        # e.g., 'NIFTY25SEP24650CE' / 'BANKNIFTY25SEP48000CE'
                        exp_hint = (expiry_date or "").upper().replace("-", "")[:6]  # very rough
                        cepe = "CE" if gated_side=="CALL" else "PE"
                        tsym = f"{symbol.upper()}{exp_hint}{int(signal.entry_strike)}{cepe}"
                        maybe_place_order_ui(gated_side, tsym, signal.qty, signal.sl, signal.tp)
                else:
                    st.warning("⚠️ Risk/Exposure limits imply 0 lots. Tweak risk %, exposure %, or lot size.")

            # Online nudge + optional persistence
            st.session_state.sig_weights = online_update_weights(
                st.session_state.sig_weights, prob_bull, st.session_state.prev_spot, spot_price, lr=0.03
            )
            st.session_state.prev_spot = spot_price
            if persist_model:
                save_weights_to_disk(st.session_state.sig_weights)

            # ------------------ MAIN CHART ------------------
            st.subheader("📈 OI + Smoothed PCR • Key Levels")
            fig = go.Figure()
            x = df_oi["Strike"].astype(int)
            fig.add_bar(x=x, y=df_oi["CE_OI"], name="CE OI (Calls — Resistance)", opacity=0.55)
            fig.add_bar(x=x, y=df_oi["PE_OI"], name="PE OI (Puts — Support)", opacity=0.55)
            fig.add_trace(go.Scatter(x=x, y=df_oi["PCR_EMA"], mode="lines+markers", name=f"PCR EMA({window})", yaxis="y2"))
            if show_heat:
                for i, K in enumerate(x):
                    p = df_oi.loc[df_oi.index[i], "PCR_EMA"]
                    if np.isnan(p): color="rgba(0,0,0,0)"
                    elif p >= 1.05: color="rgba(0,180,0,0.10)"
                    elif p <= 0.95: color="rgba(200,0,0,0.10)"
                    else: color="rgba(255,165,0,0.08)"
                    fig.add_vrect(x0=K-0.5, x1=K+0.5, fillcolor=color, opacity=0.25, line_width=0, layer="below")
            if spot_price:
                fig.add_vline(x=spot_price, line_width=2, line_dash="dash",
                              annotation_text=f"Spot {int(spot_price)}", annotation_position="top right")
            fig.add_vline(x=mp, line_width=2, line_dash="dot",
                          annotation_text=f"MaxPain {mp}", annotation_position="top left")
            if spot_price:
                atm_chart, supK, resK, supRow, resRow = nearest_levels(df_oi, spot_price)
                left, right = min(supK, resK), max(supK, resK)
                fig.add_vrect(x0=left, x1=right, fillcolor="lightblue", opacity=0.20, layer="below", line_width=0, annotation_text="Trade Zone")
                fig.add_annotation(x=supK, y=max(df_oi["PE_OI"]), text=f"Support {supK}", showarrow=True, arrowhead=2)
                fig.add_annotation(x=resK, y=max(df_oi["CE_OI"]), text=f"Resistance {resK}", showarrow=True, arrowhead=2)
            fig.update_layout(xaxis_title="Strike Price", yaxis_title="Open Interest",
                              yaxis2=dict(title="PCR (EMA)", overlaying="y", side="right", showgrid=False),
                              barmode="overlay", legend=dict(orientation="h", y=-0.22),
                              margin=dict(l=10, r=10, t=50, b=10), height=650, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # ------------------ FOCUSED CHART ------------------
            if spot_price:
                st.subheader("🎯 Near-ATM Action (ΔOI + IV Skew)")
                atm, _, _, _, _ = nearest_levels(df_oi, spot_price)
                step_chart = option_step_for_symbol(symbol)
                band = focus_strikes * step_chart
                msk = (df_oi["Strike"] >= atm - band) & (df_oi["Strike"] <= atm + band)
                dff = df_oi[msk].copy(); fx = dff["Strike"].astype(int)
                fig2 = go.Figure()
                fig2.add_bar(x=fx, y=dff["CE_dOI"], name="ΔOI CE", opacity=0.65)
                fig2.add_bar(x=fx, y=dff["PE_dOI"], name="ΔOI PE", opacity=0.65)
                fig2.add_trace(go.Scatter(x=fx, y=dff["IV_CE"], mode="lines+markers", name="IV CE", yaxis="y2"))
                fig2.add_trace(go.Scatter(x=fx, y=dff["IV_PE"], mode="lines+markers", name="IV PE", yaxis="y2"))
                fig2.update_layout(xaxis_title="Strike Price (near ATM)", yaxis_title="ΔOI",
                                   yaxis2=dict(title="IV %", overlaying="y", side="right", showgrid=False),
                                   barmode="group", legend=dict(orientation="h", y=-0.22),
                                   margin=dict(l=10, r=10, t=50, b=10), height=520, hovermode="x unified")
                st.plotly_chart(fig2, use_container_width=True)

            # ------------------ DATA TABLES ------------------
            st.subheader("📋 Detail Tables")
            if spot_price:
                atm_tbl = int(df_oi.iloc[(df_oi['Strike'] - spot_price).abs().argmin()]["Strike"])
            else:
                atm_tbl = int(df_oi.loc[np.argmin(np.abs(df_oi["Strike"] - df_oi["Strike"].median())), "Strike"])
            step_tbl = option_step_for_symbol(symbol)
            msk_tbl = (df_oi["Strike"] >= atm_tbl - table_band * step_tbl) & (df_oi["Strike"] <= atm_tbl + table_band * step_tbl)
            tbl = df_oi.loc[msk_tbl, ["Strike","CE_OI","PE_OI","CE_dOI","PE_dOI","PCR","PCR_EMA",
                                      "IV_CE","IV_PE","LTP_CE","LTP_PE","VOL_CE","VOL_PE"]].copy().sort_values("Strike")

            def highlight_atm_row(r):
                return ["background-color:#1E90FF" if r["Strike"]==atm_tbl else "" for _ in r]
            def highlight_strike_col(s):
                return ["background-color:#9ACD32; font-weight:700" if v==atm_tbl else
                        "background-color:rgba(250,204,21,0.25); font-weight:600" for v in s]

            show_cols = ["VOL_CE","CE_OI","CE_dOI","IV_CE","LTP_CE",
                         "Strike",
                         "LTP_PE","IV_PE","PE_dOI","PE_OI","VOL_PE",
                         "PCR","PCR_EMA"]

            styler = (tbl[show_cols].style
                      .format({"PCR":"{:.2f}","PCR_EMA":"{:.2f}","IV_CE":"{:.1f}","IV_PE":"{:.1f}",
                               "CE_OI":"{:,}","PE_OI":"{:,}","CE_dOI":"{:,}","PE_dOI":"{:,}",
                               "LTP_CE":"{:.2f}","LTP_PE":"{:.2f}","VOL_CE":"{:,}","VOL_PE":"{:,}"})
                      .background_gradient(subset=["PCR_EMA"], cmap="RdYlGn", vmin=0.6, vmax=1.4)
                      .apply(highlight_atm_row, axis=1)
                      .apply(highlight_strike_col, subset=["Strike"])
                     )

            # recommended strike border
            rec_strike = signal.entry_strike if (gated_side in ("CALL","PUT")) else None
            def highlight_reco_row(r):
                if rec_strike is None: return ["" for _ in r]
                return ["border-left:3px solid #3B82F6;" if r["Strike"]==rec_strike else "" for _ in r]
            styler = styler.apply(highlight_reco_row, axis=1)

            # Apply max-highlighting LAST
            max_cols = ["VOL_CE","CE_OI","CE_dOI","PE_dOI","PE_OI","VOL_PE"]
            for c in max_cols:
                if c in tbl.columns:
                    styler = styler.apply(style_max_cells, subset=[c], axis=0)

            st.dataframe(styler, use_container_width=True, height=450)
            st.caption(f"Showing ±{table_band} strikes around ATM {atm_tbl} (step {step_tbl}).")

            # ---------- Save snapshot for backtest ----------
            try:
                st.session_state.snapshots.append({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "symbol": symbol, "spot": spot_price, "mp": mp, "df": df_oi.copy(),
                })
                if len(st.session_state.snapshots)>180:
                    st.session_state.snapshots = st.session_state.snapshots[-180:]
            except Exception:
                pass

            # ---------- Backtest (this session) ----------
            if run_bt:
                snaps = [s for s in st.session_state.snapshots if s["symbol"]==symbol]
                if len(snaps)<2:
                    st.warning("Need at least 2 snapshots in this session to run backtest.")
                else:
                    trades=[]; eq=0.0; wins=0; total=0
                    for i in range(len(snaps)-1):
                        s0, s1 = snaps[i], snaps[i+1]
                        df0, df1 = s0["df"].copy(), s1["df"].copy()
                        df0["PCR_EMA"] = ema(df0["PCR"], span=window)
                        feats0 = extract_features(df0, s0["spot"], s0["mp"], window)
                        p0 = logistic_score(feats0, st.session_state.sig_weights)
                        sig0 = decide_action(p0, feats0, df0, s0["spot"], symbol, acct_equity, risk_pct, max_expo_pct, lot_size)
                        if sig0.side=="WAIT" or sig0.confidence<conf_threshold or sig0.entry_strike is None: continue
                        leg = "LTP_CE" if sig0.side=="CALL" else "LTP_PE"
                        row0 = df0.iloc[(df0["Strike"]-sig0.entry_strike).abs().argmin()]
                        row1 = df1.iloc[(df1["Strike"]-sig0.entry_strike).abs().argmin()]
                        entry = float(nz(row0.get(leg, np.nan), np.nan)); exitp = float(nz(row1.get(leg, np.nan), np.nan))
                        if any([np.isnan(entry), entry<=0, np.isnan(exitp), exitp<=0]): continue
                        pnl = round((exitp-entry)* (sig0.qty if sig0.qty else lot_size), 2)
                        eq += pnl; total += 1; wins += 1 if pnl>0 else 0
                        trades.append({"in":s0["ts"], "out":s1["ts"], "side":sig0.side, "strike":sig0.entry_strike,
                                       "entry":round(entry,2),"exit":round(exitp,2),"qty":sig0.qty or lot_size,"pnl_₹":pnl,
                                       "conf":round(sig0.confidence,2)})
                    c1,c2,c3 = st.columns(3)
                    with c1: st.metric("Trades", f"{total}")
                    with c2: st.metric("Win rate", f"{(wins/total*100):.0f}%" if total>0 else "—")
                    with c3: st.metric("P&L (₹)", f"{eq:.2f}")
                    st.dataframe(pd.DataFrame(trades), use_container_width=True, height=300)

    except Exception as e:
        st.error(f"❌ Processing error: {e}")
