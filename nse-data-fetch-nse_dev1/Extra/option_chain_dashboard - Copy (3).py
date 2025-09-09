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
st.title("📊 NSE Option Chain Dashboard (Modular Signals + ML)")

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
# Section Toggles
# ----------------------------
show_extended = st.checkbox("📊 Show Extended CE vs PE Analysis", value=True)
show_signals = st.checkbox("🔔 Show ATM ±3 Signal Generator", value=True)
show_timeline = st.checkbox("📈 Show Signal Evolution Timeline", value=True)
show_ml = st.checkbox("🤖 Enable ML Prediction", value=False)

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
HISTORY_PATH = "data\signal_history.csv"
MODEL_PATH = "data\signal_model.joblib"

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
        if show_extended:
            df_ce_table = df_ce[["strikePrice","openInterest","changeinOpenInterest","totalTradedVolume","lastPrice","change","impliedVolatility"]]
            df_ce_table.columns = ["Strike","CE OI","ΔOI CE","CE Volume","LTP CE","PriceChange CE","IV CE"]
            df_pe_table = df_pe[["strikePrice","openInterest","changeinOpenInterest","totalTradedVolume","lastPrice","change","impliedVolatility"]]
            df_pe_table.columns = ["Strike","PE OI","ΔOI PE","PE Volume","LTP PE","PriceChange PE","IV PE"]
            df_combined = pd.merge(df_ce_table, df_pe_table, on="Strike")

            st.subheader("📊 Extended Option Chain Data")
            st.dataframe(df_combined, use_container_width=True)
        else:
            df_ce_table = df_ce[["strikePrice","openInterest","changeinOpenInterest","totalTradedVolume","lastPrice","change"]]
            df_ce_table.columns = ["Strike","CE OI","ΔOI CE","CE Volume","LTP CE","PriceChange CE"]
            df_pe_table = df_pe[["strikePrice","openInterest","changeinOpenInterest","totalTradedVolume","lastPrice","change"]]
            df_pe_table.columns = ["Strike","PE OI","ΔOI PE","PE Volume","LTP PE","PriceChange PE"]
            df_combined = pd.merge(df_ce_table, df_pe_table, on="Strike")

        # ---------------- Focus Near ATM ----------------
        atm_strike = df_combined.iloc[(df_combined["Strike"] - spot_price).abs().argsort()[:1]]["Strike"].values[0]
        N = 3  # look ±3 strikes
        df_focus = df_combined[(df_combined["Strike"] >= atm_strike - N*100) &
                               (df_combined["Strike"] <= atm_strike + N*100)]

        st.subheader(f"🎯 ATM Strike: {atm_strike} | Showing ±{N} strikes")

        # ---------------- Signal Generator ----------------
        if show_signals:
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

        # ---------------- Signal Evolution Timeline ----------------
        if show_timeline and os.path.exists(HISTORY_PATH):
            st.subheader("📈 Signal Evolution Timeline")

            df_hist = pd.read_csv(HISTORY_PATH).tail(200)  # last 200 points

            # Map signals to numeric scale
            signal_map = {
                "Buy Call": 2, "Short Cover Call": 1, "Neutral": 0,
                "Sell Call": -2, "Weak Call": -1,
                "Buy Put": 2, "Short Cover Put": 1,
                "Sell Put": -2, "Weak Put": -1
            }
            df_hist["CE_Signal_Num"] = df_hist["CE Signal"].map(signal_map)
            df_hist["PE_Signal_Num"] = df_hist["PE Signal"].map(signal_map)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_hist["Timestamp"], y=df_hist["CE_Signal_Num"],
                                     mode="lines+markers", name="CE Signal", line=dict(color="blue")))
            fig.add_trace(go.Scatter(x=df_hist["Timestamp"], y=df_hist["PE_Signal_Num"],
                                     mode="lines+markers", name="PE Signal", line=dict(color="red")))

            fig.update_layout(
                title="Signal Evolution (Last 200 updates)",
                xaxis_title="Time",
                yaxis=dict(title="Signal Strength (-2=Sell, 2=Buy)"),
                height=500,
                legend=dict(orientation="h", y=-0.3)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Trade Stats
            total_trades = len(df_hist)
            flips = (df_hist["CE Signal"] != df_hist["CE Signal"].shift()).sum()
            winrate, wl_ratio = None, None

            if "LTP CE" in df_hist.columns:
                df_hist["Next_LTP"] = df_hist["LTP CE"].shift(-1)
                df_hist["Correct"] = df_hist.apply(
                    lambda r: (r["CE Signal"] == "Buy Call" and r["Next_LTP"] > r["LTP CE"]) or
                              (r["CE Signal"] == "Sell Call" and r["Next_LTP"] < r["LTP CE"]),
                    axis=1
                )
                winrate = df_hist["Correct"].mean() * 100
                wins = df_hist["Correct"].sum()
                losses = (~df_hist["Correct"]).sum()
                wl_ratio = round(wins / (losses + 1), 2)

            st.markdown(f"""
            **📊 Trade Stats**  
            - Total Trades: {total_trades}  
            - Signal Flips: {flips}  
            - Winrate: {winrate:.1f}%  
            - W/L Ratio: {wl_ratio}  
            """)

        # ---------------- ML Prediction ----------------
        if show_ml and SKLEARN_AVAILABLE and len(df_history) > 300:
            st.subheader("🤖 ML Forecasting (Next-step CE Signal)")
            df_history["Target_Next_CE"] = df_history["CE Signal"].shift(-1)
            df_hist_valid = df_history.dropna(subset=["Target_Next_CE"]).copy()

            X = df_hist_valid[["LTP CE","PriceChange CE","ΔOI CE","CE OI","LTP PE","PriceChange PE","ΔOI PE","PE OI"]]
            y = df_hist_valid["Target_Next_CE"]

            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
            model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.text("Classification Report (Next CE Signal):")
            st.text(classification_report(y_test, y_pred))

            joblib.dump(model, MODEL_PATH)

            latest_features = df_focus[["LTP CE","PriceChange CE","ΔOI CE","CE OI","LTP PE","PriceChange PE","ΔOI PE","PE OI"]]
            next_preds = model.predict(latest_features)
            df_focus["Predicted Next CE Signal"] = next_preds
            st.dataframe(df_focus[["Strike","CE Signal","Predicted Next CE Signal","PE Signal"]], use_container_width=True)

            majority_signal = pd.Series(next_preds).mode()[0]
            st.success(f"📊 Overall Next-step Prediction: **{majority_signal}**")

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


