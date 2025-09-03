import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from nsepython import nse_optionchain_scrapper

from streamlit_autorefresh import st_autorefresh

# ----------------------------
# Streamlit Config
# ----------------------------
st.set_page_config(page_title="NSE Option Chain Dashboard", layout="wide")
st.title("📊 NSE Option Chain Dashboard (Auto-Refresh)")

# ----------------------------
# User-Defined Refresh Interval
# ----------------------------
refresh_interval = st.number_input(
    "🔄 Set Auto-Refresh Interval (seconds)", min_value=10, max_value=300, value=30, step=5
)

# Auto-refresh
st_autorefresh(interval=refresh_interval * 1000, key="refreshdata")

# ----------------------------
# User Input
# ----------------------------
symbol_type = st.radio("Select Type", ["Index", "Stock"])

if symbol_type == "Index":
    symbol = st.selectbox("Select Index", ["NIFTY", "BANKNIFTY"])
else:
    symbol = st.text_input("Enter Stock Symbol (e.g. RELIANCE, TCS, INFY)")

expiry_date = None
spot_price = None

if symbol:
    try:
        data = nse_optionchain_scrapper(symbol.upper())
        expiry_list = data["records"]["expiryDates"]
        expiry_date = st.selectbox("Select Expiry Date", expiry_list)
        spot_price = data["records"]["underlyingValue"]  # LTP
        st.metric("📌 Current Spot Price", f"{spot_price}")
    except:
        st.warning("⚠️ Could not fetch expiry dates. Try again later.")

# ----------------------------
# Helper Functions
# ----------------------------
def calculate_max_pain(df_ce, df_pe):
    strikes = df_ce["strikePrice"].values
    ce_oi = df_ce["openInterest"].values
    pe_oi = df_pe["openInterest"].values

    pain = []
    for strike in strikes:
        loss = 0
        for s, ce, pe in zip(strikes, ce_oi, pe_oi):
            if s > strike:
                loss += (s - strike) * pe
            else:
                loss += (strike - s) * ce
        pain.append(loss)

    return strikes[pain.index(min(pain))]

def calculate_bias_score(df_oi, overall_pcr):
    try:
        ce_oi = df_oi["CE_OI"].sum()
        pe_oi = df_oi["PE_OI"].sum()
        oi_ratio = pe_oi / (ce_oi + 1)

        score = 50
        if overall_pcr > 1:  
            score += 20
        elif overall_pcr < 0.7:  
            score -= 20

        if oi_ratio > 1:
            score += 10
        else:
            score -= 10

        return max(0, min(100, score))
    except:
        return 50

# ----------------------------
# Auto Fetch Option Chain
# ----------------------------
if symbol and expiry_date:
    try:
        data = nse_optionchain_scrapper(symbol.upper())
        filtered_data = [d for d in data["records"]["data"] if d["expiryDate"] == expiry_date]

        ce_data = [d["CE"] for d in filtered_data if "CE" in d]
        pe_data = [d["PE"] for d in filtered_data if "PE" in d]

        df_ce = pd.DataFrame(ce_data)
        df_pe = pd.DataFrame(pe_data)

        st.success(f"✅ Auto-updated Option Chain for {symbol.upper()} (Expiry: {expiry_date}) "
                   f"[Refresh: {refresh_interval}s]")

        # ----------------------------
        # Merge OI Data
        # ----------------------------
        df_oi = pd.DataFrame({
            "StrikePrice": df_ce["strikePrice"],
            "CE_OI": df_ce["openInterest"],
            "PE_OI": df_pe["openInterest"]
        })

        df_oi["PCR"] = df_oi.apply(
            lambda row: row["PE_OI"] / row["CE_OI"] if row["CE_OI"] > 0 else None, axis=1
        )

        # ----------------------------
        # Strongest Support & Resistance
        # ----------------------------
        strongest_support = df_oi.loc[df_oi["PE_OI"].idxmax()]
        strongest_resistance = df_oi.loc[df_oi["CE_OI"].idxmax()]

        # ----------------------------
        # Overall PCR
        # ----------------------------
        overall_pcr = df_pe["openInterest"].sum() / df_ce["openInterest"].sum()

        # ----------------------------
        # Sentiment Badge with PCR
        # ----------------------------
        if strongest_support["PE_OI"] > strongest_resistance["CE_OI"]:
            sentiment = "🟢 Bullish Bias"
            badge_color = "#90EE90"
        elif strongest_resistance["CE_OI"] > strongest_support["PE_OI"]:
            sentiment = "🔴 Bearish Bias"
            badge_color = "#FF7F7F"
        else:
            sentiment = "🟡 Neutral Bias"
            badge_color = "#ADD8E6"

        if overall_pcr > 1.05:
            pcr_color = "green"
        elif overall_pcr < 0.95:
            pcr_color = "red"
        else:
            pcr_color = "orange"

        st.markdown(
            f"""
            <div style="background-color:{badge_color}; 
                        padding:12px; 
                        border-radius:12px; 
                        text-align:center; 
                        font-size:18px; 
                        font-weight:bold;">
                {sentiment} <br>
                📊 Overall PCR: <span style="color:{pcr_color};">{overall_pcr:.2f}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------
        # Combined Chart
        # ----------------------------
        st.subheader("📈 Support/Resistance + Strike-wise PCR")

        fig = go.Figure()

        fig.add_bar(x=df_oi["StrikePrice"], y=df_oi["CE_OI"],
                    name="CE OI (Calls - Resistance)", marker_color="red", opacity=0.6)
        fig.add_bar(x=df_oi["StrikePrice"], y=df_oi["PE_OI"],
                    name="PE OI (Puts - Support)", marker_color="green", opacity=0.6)

        fig.add_trace(go.Scatter(
            x=df_oi["StrikePrice"], y=df_oi["PCR"],
            mode="lines+markers", name="PCR (Put/Call Ratio)",
            line=dict(color="blue", width=2), yaxis="y2"
        ))

        fig.add_trace(go.Scatter(
            x=[strongest_support["StrikePrice"]], y=[strongest_support["PE_OI"]],
            mode="markers+text", name="Strongest Support",
            marker=dict(color="green", size=14, symbol="star"),
            text=[f"Support: {int(strongest_support['StrikePrice'])}"], textposition="top center"
        ))

        fig.add_trace(go.Scatter(
            x=[strongest_resistance["StrikePrice"]], y=[strongest_resistance["CE_OI"]],
            mode="markers+text", name="Strongest Resistance",
            marker=dict(color="red", size=14, symbol="star"),
            text=[f"Resistance: {int(strongest_resistance['StrikePrice'])}"], textposition="top center"
        ))

        colors = df_oi["PCR"].apply(
            lambda x: "rgba(0,200,0,0.2)" if x and x > 1 else "rgba(200,0,0,0.2)"
        )
        for i, strike in enumerate(df_oi["StrikePrice"]):
            fig.add_vrect(
                x0=strike - 0.5, x1=strike + 0.5,
                fillcolor=colors.iloc[i], opacity=0.2,
                layer="below", line_width=0
            )

        if spot_price:
            st.subheader("🎯 Trade Zone Analysis")

            atm_strike = df_oi.iloc[(df_oi["StrikePrice"] - spot_price).abs().argsort()[:1]]

            above_atm = df_oi[df_oi["StrikePrice"] >= spot_price]
            nearest_res = above_atm.loc[above_atm["CE_OI"].idxmax()]

            below_atm = df_oi[df_oi["StrikePrice"] <= spot_price]
            nearest_sup = below_atm.loc[below_atm["PE_OI"].idxmax()]

            st.write(f"📌 **ATM Strike:** {int(atm_strike['StrikePrice'].values[0])}")
            st.write(f"🟢 **Nearest Support:** {int(nearest_sup['StrikePrice'])} (PE OI: {nearest_sup['PE_OI']})")
            st.write(f"🔴 **Nearest Resistance:** {int(nearest_res['StrikePrice'])} (CE OI: {nearest_res['CE_OI']})")
            st.success(f"🎯 **Trade Zone Range → {int(nearest_sup['StrikePrice'])}  -  {int(nearest_res['StrikePrice'])}**")

            fig.add_vrect(
                x0=int(nearest_sup["StrikePrice"]), x1=int(nearest_res["StrikePrice"]),
                fillcolor="lightblue", opacity=0.2,
                layer="below", line_width=0,
                annotation_text="Trade Zone", annotation_position="top left"
            )

        fig.update_layout(
            title=f"Option Chain Analysis for {symbol.upper()} ({expiry_date})",
            xaxis=dict(title="Strike Price"),
            yaxis=dict(title="Open Interest"),
            yaxis2=dict(title="PCR", overlaying="y", side="right", showgrid=False),
            barmode="overlay", legend=dict(orientation="h", y=-0.2), height=650
        )

        st.plotly_chart(fig, use_container_width=True)

        overall_pcr = df_pe["openInterest"].sum() / df_ce["openInterest"].sum()
        st.metric("📊 Overall Put-Call Ratio (PCR)", f"{overall_pcr:.2f}")

        st.subheader("📌 Intraday Bias / Market Sentiment")
        bias_score = calculate_bias_score(df_oi, overall_pcr)
        if bias_score > 60:
            st.success(f"🟢 Bullish Bias Score: {bias_score}/100")
        elif bias_score < 40:
            st.error(f"🔴 Bearish Bias Score: {bias_score}/100")
        else:
            st.warning(f"🟡 Neutral Bias Score: {bias_score}/100")

        max_pain = calculate_max_pain(df_ce, df_pe)
        st.subheader("🎯 Max Pain (Expiry Magnet)")
        st.info(f"👉 Max Pain is at Strike: **{int(max_pain)}**")

        # ----------------------------
        # ΔOI + IV Tables with Extended Data
        # ----------------------------
        st.subheader("📊 Option Chain Detailed View (CE vs PE)")

        df_ce_table = df_ce[[
            "strikePrice", "openInterest", "changeinOpenInterest",
            "totalTradedVolume", "lastPrice", "change", "impliedVolatility"
        ]]
        df_pe_table = df_pe[[
            "strikePrice", "openInterest", "changeinOpenInterest",
            "totalTradedVolume", "lastPrice", "change", "impliedVolatility"
        ]]

        df_ce_table.columns = [
            "Strike", "CE OI", "ΔOI CE", "CE Volume", "LTP CE", "PriceChange CE", "IV CE"
        ]
        df_pe_table.columns = [
            "Strike", "PE OI", "ΔOI PE", "PE Volume", "LTP PE", "PriceChange PE", "IV PE"
        ]

        df_combined = pd.merge(df_ce_table, df_pe_table, on="Strike")

        def highlight_doi(val):
            if val > 0:
                return "background-color: rgba(0,200,0,0.3)"
            elif val < 0:
                return "background-color: rgba(200,0,0,0.3)"
            return ""

        def highlight_iv(val):
            if val > 40:
                return "background-color: rgba(255,165,0,0.3)"
            elif val < 20:
                return "background-color: rgba(0,200,0,0.2)"
            return ""

        styled_table = df_combined.style.applymap(highlight_doi, subset=["ΔOI CE", "ΔOI PE"]) \
                                        .applymap(highlight_iv, subset=["IV CE", "IV PE"])

        st.dataframe(styled_table, use_container_width=True)

        # ----------------------------
        # Extended Relation Analysis
        # ----------------------------
        def analyze_strike_relations(df_combined, spot_price):
            try:
                atm_strike = df_combined.iloc[(df_combined["Strike"] - spot_price).abs().argsort()[:1]]["Strike"].values[0]

                df_focus = df_combined[(df_combined["Strike"] >= atm_strike - 5*100) &
                                       (df_combined["Strike"] <= atm_strike + 5*100)]

                df_focus["SentimentScore"] = (
                    (df_focus["ΔOI PE"] - df_focus["ΔOI CE"]) / 1000
                    + (df_focus["PriceChange PE"] - df_focus["PriceChange CE"]) * 2
                )

                st.subheader(f"📊 CE vs PE Extended Relations (±5 Strikes around ATM: {atm_strike})")

                fig = go.Figure()
                fig.add_bar(x=df_focus["Strike"], y=df_focus["CE Volume"], name="CE Volume", marker_color="red", opacity=0.6)
                fig.add_bar(x=df_focus["Strike"], y=df_focus["PE Volume"], name="PE Volume", marker_color="green", opacity=0.6)

                fig.add_trace(go.Scatter(
                    x=df_focus["Strike"], y=df_focus["LTP CE"], mode="lines+markers",
                    name="LTP CE", line=dict(color="orange", dash="dot")
                ))
                fig.add_trace(go.Scatter(
                    x=df_focus["Strike"], y=df_focus["LTP PE"], mode="lines+markers",
                    name="LTP PE", line=dict(color="blue", dash="dot")
                ))

                fig.add_trace(go.Scatter(
                    x=df_focus["Strike"], y=df_focus["SentimentScore"],
                    mode="lines+markers", name="Net Sentiment Score",
                    line=dict(color="purple", width=3)
                ))

                fig.update_layout(
                    title=f"Strike-wise CE vs PE (Volume + LTP + ΔOI + Price Change + Sentiment) near ATM {atm_strike}",
                    xaxis=dict(title="Strike Price"),
                    yaxis=dict(title="Values"),
                    barmode="group",
                    legend=dict(orientation="h", y=-0.2),
                    height=600
                )

                st.plotly_chart(fig, use_container_width=True)

                df_focus["Bias"] = df_focus["SentimentScore"].apply(
                    lambda x: "🟢 Bullish" if x > 0 else ("🔴 Bearish" if x < 0 else "🟡 Neutral")
                )

                st.dataframe(
                    df_focus[[
                         "IV CE", "CE Volume", "CE OI",  "ΔOI CE", "PriceChange CE", "LTP CE",
                        "Strike",
                        "LTP PE", "PriceChange PE",  "ΔOI PE", "PE OI", "PE Volume",  "IV PE", 
                        "SentimentScore", "Bias"
                    ]],
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"⚠️ Extended relation analysis failed: {e}")

        analyze_strike_relations(df_combined, spot_price)

    except Exception as e:
        st.error(f"❌ Error fetching data: {e}")

        # ----------------------------
    # Max Value Summary with Strike
    # ----------------------------
    st.subheader("📊 Maximum Values with Strike Price")

    max_summary = []

    # Call Metrics
    max_summary.append(("Max Call Volume", 
                        df_combined.loc[df_combined["CE Volume"].idxmax(), "CE Volume"],
                        df_combined.loc[df_combined["CE Volume"].idxmax(), "Strike"]))
    max_summary.append(("Max Call Net OI", 
                        df_combined.loc[df_combined["CE OI"].idxmax(), "CE OI"],
                        df_combined.loc[df_combined["CE OI"].idxmax(), "Strike"]))
    max_summary.append(("Max Call Change in OI", 
                        df_combined.loc[df_combined["ΔOI CE"].idxmax(), "ΔOI CE"],
                        df_combined.loc[df_combined["ΔOI CE"].idxmax(), "Strike"]))

    # Put Metrics
    max_summary.append(("Max Put Change in OI", 
                        df_combined.loc[df_combined["ΔOI PE"].idxmax(), "ΔOI PE"],
                        df_combined.loc[df_combined["ΔOI PE"].idxmax(), "Strike"]))
    max_summary.append(("Max Put Net OI", 
                        df_combined.loc[df_combined["PE OI"].idxmax(), "PE OI"],
                        df_combined.loc[df_combined["PE OI"].idxmax(), "Strike"]))
    max_summary.append(("Max Put Volume", 
                        df_combined.loc[df_combined["PE Volume"].idxmax(), "PE Volume"],
                        df_combined.loc[df_combined["PE Volume"].idxmax(), "Strike"]))

    # Convert to DataFrame
    df_max = pd.DataFrame(max_summary, columns=["Metric", "Max Value", "Strike Price"])

    st.table(df_max)

