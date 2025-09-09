import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from nsepython import nse_optionchain_scrapper
from streamlit_autorefresh import st_autorefresh

import os
from datetime import datetime

# ML imports
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import joblib
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="NSE Option Chain Dashboard", layout="wide")
st.title("📊 NSE Option Chain Dashboard (Signals + ML Forecasting)")

refresh_interval = st.number_input("🔄 Set Auto-Refresh Interval (seconds)",
                                   min_value=10, max_value=300, value=30, step=5)
st_autorefresh(interval=refresh_interval * 1000, key="refreshdata")

symbol_type = st.radio("Select Type", ["Index", "Stock"])
if symbol_type == "Index":
    symbol = st.selectbox("Select Index", ["NIFTY", "BANKNIFTY"])
else:
    symbol = st.text_input("Enter Stock Symbol (e.g. RELIANCE, TCS, INFY)")

expiry_date, spot_price = None, None

if symbol:
    try:
        data = nse_optionchain_scrapper(symbol.upper())
        expiry_list = data["records"]["expiryDates"]
        expiry_date = st.selectbox("Select Expiry Date", expiry_list)
        spot_price = data["records"]["underlyingValue"]
        st.metric("📌 Current Spot Price", f"{spot_price}")
    except Exception as e:
        st.warning(f"⚠️ Could not fetch expiry dates. {e}")

# ----------------------------
# Helper Functions
# ----------------------------
def classify_signal(price_change, doi, side="CE"):
    PRICE_CHANGE_THRESHOLD, DOI_THRESHOLD = 0.3, 500
    if abs(price_change) < PRICE_CHANGE_THRESHOLD or abs(doi) < DOI_THRESHOLD:
        return "Neutral"
    if side == "CE":
        if price_change > 0 and doi > 0: return "Buy Call"
        if price_change < 0 and doi > 0: return "Sell Call"
        if price_change > 0 and doi < 0: return "Short Cover Call"
        if price_change < 0 and doi < 0: return "Weak Call"
    else:
        if price_change > 0 and doi > 0: return "Buy Put"
        if price_change < 0 and doi > 0: return "Sell Put"
        if price_change > 0 and doi < 0: return "Short Cover Put"
        if price_change < 0 and doi < 0: return "Weak Put"
    return "Neutral"

# Paths
HISTORY_PATH = "data/signal_history.csv"
MODEL_PATH = "data/signal_model.joblib"

# ----------------------------
# Main
# ----------------------------
if symbol and expiry_date:
    try:
        data = nse_optionchain_scrapper(symbol.upper())
        filtered_data = [d for d in data["records"]["data"] if d["expiryDate"] == expiry_date]
        ce_data = [d["CE"] for d in filtered_data if "CE" in d]
        pe_data = [d["PE"] for d in filtered_data if "PE" in d]
        df_ce, df_pe = pd.DataFrame(ce_data), pd.DataFrame(pe_data)

        st.success(f"✅ Auto-updated Option Chain for {symbol.upper()} ({expiry_date}) [Refresh: {refresh_interval}s]")

        # ---------------- Extended CE vs PE Data ----------------
        df_ce_table = df_ce[["strikePrice","openInterest","changeinOpenInterest","totalTradedVolume","lastPrice","change","impliedVolatility"]]
        df_ce_table.columns = ["Strike","CE OI","ΔOI CE","CE Volume","LTP CE","PriceChange CE","IV CE"]
        df_pe_table = df_pe[["strikePrice","openInterest","changeinOpenInterest","totalTradedVolume","lastPrice","change","impliedVolatility"]]
        df_pe_table.columns = ["Strike","PE OI","ΔOI PE","PE Volume","LTP PE","PriceChange PE","IV PE"]
        df_combined = pd.merge(df_ce_table, df_pe_table, on="Strike")

        # ---------------- Focus Near ATM ----------------
        atm_strike = df_combined.iloc[(df_combined["Strike"] - spot_price).abs().argsort()[:1]]["Strike"].values[0]
        N = 3  # look ±3 strikes
        df_focus = df_combined[(df_combined["Strike"] >= atm_strike - N*100) &
                               (df_combined["Strike"] <= atm_strike + N*100)]

        st.subheader(f"🎯 ATM Strike: {atm_strike} | Showing ±{N} strikes")
        st.dataframe(df_focus, use_container_width=True)

        # ---------------- Signal Generator ----------------
        st.subheader("🔔 Trading Signals (Near ATM)")
        df_focus["CE Signal"] = df_focus.apply(lambda r: classify_signal(r["PriceChange CE"], r["ΔOI CE"], "CE"), axis=1)
        df_focus["PE Signal"] = df_focus.apply(lambda r: classify_signal(r["PriceChange PE"], r["ΔOI PE"], "PE"), axis=1)
        st.dataframe(df_focus[["Strike","CE Signal","PE Signal"]], use_container_width=True)

        # ---------------- Save history ----------------
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df_focus["Timestamp"] = timestamp
        if os.path.exists(HISTORY_PATH):
            df_history = pd.read_csv(HISTORY_PATH)
            df_history = pd.concat([df_history, df_focus], ignore_index=True)
        else:
            df_history = df_focus.copy()
        df_history.to_csv(HISTORY_PATH, index=False)
        st.success(f"✅ History updated at {HISTORY_PATH}")

        # ---------------- ML Prediction (CE + PE) ----------------
        if SKLEARN_AVAILABLE and len(df_history) > 300:
            st.subheader("🤖 ML Forecasting (Next-step CE & PE Signals)")

            # ---- CE Forecast ----
            df_history["Target_Next_CE"] = df_history["CE Signal"].shift(-1)
            df_hist_ce = df_history.dropna(subset=["Target_Next_CE"]).copy()
            X_ce = df_hist_ce[["LTP CE","PriceChange CE","ΔOI CE","CE OI",
                               "LTP PE","PriceChange PE","ΔOI PE","PE OI"]]
            y_ce = df_hist_ce["Target_Next_CE"]

            X_train_ce, X_test_ce, y_train_ce, y_test_ce = train_test_split(
                X_ce, y_ce, test_size=0.2, random_state=42, stratify=y_ce
            )
            model_ce = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
            model_ce.fit(X_train_ce, y_train_ce)
            y_pred_ce = model_ce.predict(X_test_ce)

            st.text("Classification Report (Next CE Signal):")
            st.text(classification_report(y_test_ce, y_pred_ce))

            # ---- PE Forecast ----
            df_history["Target_Next_PE"] = df_history["PE Signal"].shift(-1)
            df_hist_pe = df_history.dropna(subset=["Target_Next_PE"]).copy()
            X_pe = df_hist_pe[["LTP CE","PriceChange CE","ΔOI CE","CE OI",
                               "LTP PE","PriceChange PE","ΔOI PE","PE OI"]]
            y_pe = df_hist_pe["Target_Next_PE"]

            X_train_pe, X_test_pe, y_train_pe, y_test_pe = train_test_split(
                X_pe, y_pe, test_size=0.2, random_state=42, stratify=y_pe
            )
            model_pe = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
            model_pe.fit(X_train_pe, y_train_pe)
            y_pred_pe = model_pe.predict(X_test_pe)

            st.text("Classification Report (Next PE Signal):")
            st.text(classification_report(y_test_pe, y_pred_pe))

            # ---- Save models ----
            joblib.dump(model_ce, MODEL_PATH.replace(".joblib","_ce.joblib"))
            joblib.dump(model_pe, MODEL_PATH.replace(".joblib","_pe.joblib"))

            # ---- Predict latest ----
            latest_features = df_focus[["LTP CE","PriceChange CE","ΔOI CE","CE OI",
                                        "LTP PE","PriceChange PE","ΔOI PE","PE OI"]]
            next_preds_ce = model_ce.predict(latest_features)
            next_preds_pe = model_pe.predict(latest_features)

            df_focus["Predicted Next CE Signal"] = next_preds_ce
            df_focus["Predicted Next PE Signal"] = next_preds_pe

            # ---- Show updated table ----
            st.dataframe(
                df_focus[["Strike","CE Signal","Predicted Next CE Signal","PE Signal","Predicted Next PE Signal"]],
                use_container_width=True
            )

            # ---- Overall Forecast ----
            majority_ce = pd.Series(next_preds_ce).mode()[0]
            majority_pe = pd.Series(next_preds_pe).mode()[0]
            st.success(f"📊 Overall Next-step Prediction → CE: **{majority_ce}** | PE: **{majority_pe}**")

    except Exception as e:
        st.error(f"❌ Error: {e}")


# ---------------- Candle + Signal Graph ----------------
st.subheader("📊 Price Action with CE & PE Signals")

if os.path.exists(HISTORY_PATH):
    df_hist = pd.read_csv(HISTORY_PATH)

    # Use last 100 records for plotting
    df_plot = df_hist.tail(100).copy()

    # Fake OHLC from CE LTP (since NSE API doesn't give OHLC directly here)
    df_plot["Open"] = df_plot["LTP CE"].shift(1).fillna(df_plot["LTP CE"])
    df_plot["High"] = df_plot[["Open","LTP CE"]].max(axis=1) + abs(df_plot["PriceChange CE"])
    df_plot["Low"]  = df_plot[["Open","LTP CE"]].min(axis=1) - abs(df_plot["PriceChange CE"])
    df_plot["Close"] = df_plot["LTP CE"]
    df_plot["Volume"] = df_plot["CE Volume"] if "CE Volume" in df_plot else 1000

    # Moving Average
    df_plot["EMA"] = df_plot["Close"].ewm(span=9, adjust=False).mean()

    fig = go.Figure()

    # Candlestick (Price)
    fig.add_trace(go.Candlestick(
        x=df_plot["Timestamp"],
        open=df_plot["Open"], high=df_plot["High"],
        low=df_plot["Low"], close=df_plot["Close"],
        name="Price"
    ))

    # EMA Line
    fig.add_trace(go.Scatter(
        x=df_plot["Timestamp"], y=df_plot["EMA"],
        mode="lines", line=dict(color="orange", width=2),
        name="EMA (9)"
    ))

    # --- CE Signals ---
    buy_ce = df_plot[df_plot["CE Signal"].str.contains("Buy", na=False)]
    sell_ce = df_plot[df_plot["CE Signal"].str.contains("Sell", na=False)]

    fig.add_trace(go.Scatter(
        x=buy_ce["Timestamp"], y=buy_ce["Close"],
        mode="markers", marker=dict(symbol="triangle-up", color="green", size=12),
        name="CE Buy"
    ))
    fig.add_trace(go.Scatter(
        x=sell_ce["Timestamp"], y=sell_ce["Close"],
        mode="markers", marker=dict(symbol="triangle-down", color="red", size=12),
        name="CE Sell"
    ))

    # --- PE Signals ---
    buy_pe = df_plot[df_plot["PE Signal"].str.contains("Buy", na=False)]
    sell_pe = df_plot[df_plot["PE Signal"].str.contains("Sell", na=False)]

    fig.add_trace(go.Scatter(
        x=buy_pe["Timestamp"], y=buy_pe["Close"],
        mode="markers", marker=dict(symbol="triangle-up", color="blue", size=12),
        name="PE Buy"
    ))
    fig.add_trace(go.Scatter(
        x=sell_pe["Timestamp"], y=sell_pe["Close"],
        mode="markers", marker=dict(symbol="triangle-down", color="purple", size=12),
        name="PE Sell"
    ))

    # Volume Bars
    fig.add_trace(go.Bar(
        x=df_plot["Timestamp"], y=df_plot["Volume"],
        name="Volume", marker=dict(color="lightgrey"), yaxis="y2"
    ))

    # Layout with dual y-axis
    fig.update_layout(
        title="Price Action + CE & PE Signals",
        xaxis=dict(title="Time", rangeslider=dict(visible=False)),
        yaxis=dict(title="Price", side="right"),
        yaxis2=dict(title="Volume", overlaying="y", side="left", showgrid=False),
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)
