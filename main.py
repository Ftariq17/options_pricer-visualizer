import numpy as np
import plotly.graph_objects as go
import pandas as pd
from scipy.stats import norm
import streamlit as st

# Black-Scholes Model
# -----------------------------
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Greeks Calculation

def greeks(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
             r * K * np.exp(-r * T) * (norm.cdf(d2) if option_type == 'call' else -norm.cdf(-d2))) / 365
    vega = (S * norm.pdf(d1) * np.sqrt(T)) / 100
    rho = ((K * T * np.exp(-r * T) *
           (norm.cdf(d2) if option_type == 'call' else -norm.cdf(-d2)))) / 100

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta),
        "vega": float(vega),
        "rho": float(rho),
    }, d2

# Implied Volatility Estimation

def implied_volatility(S, K, T, r, market_price, option_type='call', tol=1e-5, max_iter=100):
    sigma = 0.2
    for _ in range(max_iter):
        price = black_scholes(S, K, T, r, sigma, option_type)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        sigma -= diff / vega
    return sigma

# Payoff Calculation Functions

def single_option_payoff(prices, K, premium, option_type="call", position="Buy"):
    intrinsic = np.maximum(prices - K, 0) if option_type == "call" else np.maximum(K - prices, 0)
    return intrinsic - premium if position == "Buy" else premium - intrinsic


def spread_payoff(prices, K1, K2, premium1, premium2, option_type="call", position1="Buy", position2="Sell"):
    payoff1 = single_option_payoff(prices, K1, premium1, option_type, position=position1)
    payoff2 = single_option_payoff(prices, K2, premium2, option_type, position=position2)
    return payoff1 + payoff2

# Plotting Functions
def plotly_payoff(prices, payoff, strike_prices):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices, y=payoff, mode='lines', name='Payoff', line=dict(color='#111AC7',width=3)))
    for k in strike_prices:
        fig.add_shape(
            type='line',
            x0=k, x1=k,
            y0=0, y1=1,
            xref='x',
            yref='paper',
            line=dict(color='red', width=2, dash='dot')
        )

    fig.update_layout(title='Options Payoff Diagram',
                      xaxis_title='Underlying Price',
                      yaxis_title='Profit / Loss',
                      showlegend=True)
    return fig


def plotly_theo_vs_payoff(prices, K, T, r, sigma, premium, option_type="call", position="Buy"):
    theo_values = [black_scholes(p, K, T, r, sigma, option_type) for p in prices]
    intrinsic = np.maximum(prices - K, 0) if option_type == "call" else np.maximum(K - prices, 0)
    payoff = intrinsic - premium if position == "Buy" else premium - intrinsic

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices, y=theo_values, mode='lines', name='Theoretical Value', line=dict(color='green',width=3)))
    fig.add_trace(go.Scatter(x=prices, y=payoff, mode='lines', name='Payoff at Expiry', line=dict(color='#111AC7', dash='dash')))
    fig.add_shape(
        type='line',
        x0=K, x1=K,
        y0=0, y1=1,
        xref='x',
        yref='paper',
        line=dict(color='red', width=2, dash='dot')
    )
    fig.add_shape(type='line', x0=min(prices), x1=max(prices), y0=0, y1=0,
                  line=dict(color='black', width=2, dash='dash'))
    fig.update_layout(title='Theoretical Value vs Payoff',
                      xaxis_title='Underlying Price',
                      yaxis_title='Value / P&L',
                      showlegend=True)
    return fig, theo_values, payoff


def plotly_spread_theo_vs_payoff(prices, K1, K2, T, r, sigma, premium1, premium2, option_type):
    theo1 = [black_scholes(p, K1, T, r, sigma, option_type) for p in prices]
    theo2 = [black_scholes(p, K2, T, r, sigma, option_type) for p in prices]
    theo_values = np.array(theo1) - np.array(theo2)
    payoff = spread_payoff(prices, K1, K2, premium1, premium2, option_type)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices, y=theo_values, mode='lines', name='Theoretical Value', line=dict(color='green',width=3)))
    fig.add_trace(go.Scatter(x=prices, y=payoff, mode='lines', name='Payoff at Expiry', line=dict(color='#111AC7', dash='dash',width=3)))
    fig.add_shape(
        type='line',
        x0=K1, x1=K1,
        y0=0, y1=1,
        xref='x',
        yref='paper',
        line=dict(color='red', width=2, dash='dot')
    )
    fig.add_shape(
        type='line',
        x0=K2, x1=K2,
        y0=0, y1=1,
        xref='x',
        yref='paper',
        line=dict(color='purple', width=2, dash='dot')
    )
    fig.add_shape(type='line', x0=min(prices), x1=max(prices), y0=0, y1=0,
                  line=dict(color='black', width=2, dash='dash'))
    fig.update_layout(title='Spread: Theoretical Value vs Payoff',
                      xaxis_title='Underlying Price',
                      yaxis_title='Value / P&L',
                      showlegend=True)
    return fig, theo_values, payoff

# Streamlit App
st.set_page_config(page_title="Options Pricer & Strategy Visualizer", layout="centered")
st.title("Options Pricing and Strategy Visualizer")
st.sidebar.header("Inputs")
strategy = st.sidebar.selectbox("Strategy Type", ["Single Option", "Bull Call Spread", "Bull Put Spread"])
view_mode = st.sidebar.selectbox("View Mode", ["Per Share", "Per 100 Shares"])
multiplier = 100 if view_mode == "Per 100 Shares" else 1
if strategy == "Single Option":
    vol_mode = st.sidebar.selectbox("Volatility Input Mode", ["Manual", "Implied Volatility"])
else:
    vol_mode = "Manual"
    st.sidebar.markdown("*Implied Volatility estimation is only available for Single Options.*")

if strategy == "Single Option":
    option_type = st.sidebar.radio("Option Type", ["call", "put"])
    position = st.sidebar.radio("Position", ["Buy", "Sell"])
else:
    option_type = "call" if strategy == "Bull Call Spread" else "put"
    position = "Spread"

S = st.sidebar.slider("Market Price/Spot Price (S)", 50, 150, 100)

if strategy == "Single Option":
    K = st.sidebar.slider("Strike Price (K)", 50, 150, 100)
else:
    option_type = "call" if strategy == "Bull Call Spread" else "put"
    K1 = st.sidebar.slider("Lower Strike (Buy)", 50, 145, 90)
    K2 = st.sidebar.slider("Upper Strike (Sell)", K1 + 1, 150, 110)

T = st.sidebar.slider("Time to Expiry (T)", 0.01, 2.0, 1.0)
r = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0) / 100

if vol_mode == "Manual":
    vol = st.sidebar.slider("Volatility (%)", 1.0, 100.0, 20.0) / 100
else:
    # Only for Single Option
    market_price = st.sidebar.number_input("Market Price", value=10.0, step=0.1)
    temp_option_type = option_type
    temp_strike = K
    vol = implied_volatility(S, temp_strike, T, r, market_price, temp_option_type)

prices = np.linspace(0.5 * S, 1.5 * S, 100)

if strategy == "Single Option":
    premium = black_scholes(S, K, T, r, vol, option_type)
    payoff = single_option_payoff(prices, K, premium, option_type, position)
    fig = plotly_payoff(prices, payoff, [K])
    st.subheader("Option Price")
    st.markdown(f"**Premium ({position}):** ${premium * multiplier:.2f}")
    if vol_mode == "Implied Volatility":
        st.markdown(f"**Estimated Implied Volatility:** `{vol:.2%}`")
    st.plotly_chart(fig,width=800,height=500)

    greeks_vals, d2 = greeks(S, K, T, r, vol, option_type)
    st.subheader("Strategy Summary")
    breakeven = K + premium if option_type == 'call' and position == "Buy" else (
        K - premium if option_type == 'put' and position == "Buy" else (
        K - premium if option_type == 'call' else K + premium))
    max_profit = float('inf') if position == "Buy" and option_type == 'call' else (
        K - premium if option_type == 'put' and position == "Buy" else premium)
    max_loss = premium if position == "Buy" else float('inf')
    prob_itm = norm.cdf(d2) if option_type == 'call' else norm.cdf(-d2)
    st.markdown(f"**Breakeven Price:** ${breakeven:.2f}")
    st.markdown(f"**Max Potential Profit:** {'Unlimited' if max_profit == float('inf') else f'${max_profit * multiplier:.2f}'}")
    st.markdown(f"**Max Potential Loss:** {'Unlimited' if max_loss == float('inf') else f'${max_loss * multiplier:.2f}'}")
    st.markdown(f"**Estimated Probability ITM:** {prob_itm * 100:.2f}%")

    st.subheader("Greeks")
    for key, val in greeks_vals.items():
        st.markdown(f"{key.capitalize()}: {val:.4f}")

    st.subheader("Theoretical Value vs Payoff")
    fig2, theo_values, payoff_values = plotly_theo_vs_payoff(prices, K, T, r, vol, premium, option_type, position)
    st.plotly_chart(fig2)
    df = pd.DataFrame({"Price": prices, "Theoretical Value": theo_values, "Payoff": payoff_values})
    st.download_button("Download Data (CSV)", df.to_csv(index=False), file_name="option_data.csv")

else:
    if strategy == "Bull Put Spread":
        option_type = "put"
        position1, position2 = "Buy", "Sell"  # Short put at higher strike, long put at lower strike
        premium1 = black_scholes(S, K1, T, r, vol, option_type)  # Short leg (K2)
        premium2 = black_scholes(S, K2, T, r, vol, option_type)  # Long leg (K1)
        net_premium = premium2 - premium1  # Net credit

    else:  # Bull Call Spread
        option_type = "call"
        position1, position2 = "Buy", "Sell"
        premium1 = black_scholes(S, K1, T, r, vol, option_type)
        premium2 = black_scholes(S, K2, T, r, vol, option_type)
        net_premium = premium1 - premium2

    payoff = spread_payoff(prices, K1, K2, premium1, premium2, option_type, position1, position2)
    fig = plotly_payoff(prices, payoff, [K1, K2])

    st.subheader("Strategy Summary")

    if strategy == "Bull Put Spread":
        breakeven = K2 - net_premium  # Short put strike minus premium received
        max_profit = net_premium
        max_loss = (K2 - K1) - net_premium
        st.markdown(f"**Net Premium Received:** ${net_premium * multiplier:.2f}")

    else:  # Bull Call Spread
        breakeven = K1 + net_premium  # Long call strike + premium paid
        max_profit = (K2 - K1) - net_premium
        max_loss = net_premium
        st.markdown(f"**Net Premium Paid:** ${net_premium * multiplier:.2f}")

    # Use d2 of short leg (K2) for ITM estimation
    d2_spread = (np.log(S / K2) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    prob_itm_spread = norm.cdf(d2_spread) if option_type == "call" else norm.cdf(-d2_spread)

    st.markdown(f"**Breakeven Price:** ${breakeven:.2f}")
    st.markdown(f"**Max Potential Profit:** ${max_profit * multiplier:.2f}")
    st.markdown(f"**Max Potential Loss:** ${max_loss * multiplier:.2f}")
    st.markdown(f"**Estimated Probability ITM (Max Profit Zone):** {prob_itm_spread * 100:.2f}%")

    st.plotly_chart(fig,width=800,height=500)


    # Greeks for each leg
    greeks_leg1, _ = greeks(S, K1, T, r, vol, option_type)
    greeks_leg2, _ = greeks(S, K2, T, r, vol, option_type)

    # Adjust signs based on position
    sign1 = 1 if position1 == "Buy" else -1
    sign2 = 1 if position2 == "Buy" else -1

    # Combine Greeks
    total_greeks = {
        greek: greeks_leg1[greek] * sign1 + greeks_leg2[greek] * sign2
        for greek in greeks_leg1
    }

    st.subheader("Combined Greeks (Strategy-Level)")
    for key, val in total_greeks.items():
        st.markdown(f"{key.capitalize()}: {val:.4f}")

    # Greek Sensitivity vs Spot Price Plot
    st.subheader("Greek Sensitivity vs Spot Price")
    greek_choice = st.selectbox("Select Greek to Plot", ["Delta", "Gamma", "Theta", "Vega"])

    spot_range = np.linspace(0.5 * S, 1.5 * S, 100)
    combined_values = []
    leg1_values = []
    leg2_values = []

    for spot in spot_range:
        g1, _ = greeks(spot, K1, T, r, vol, option_type)
        g2, _ = greeks(spot, K2, T, r, vol, option_type)
        val1 = g1[greek_choice.lower()] * sign1
        val2 = g2[greek_choice.lower()] * sign2
        leg1_values.append(val1)
        leg2_values.append(val2)
        combined_values.append(val1 + val2)

    color_map = {
        "Delta": "blue",
        "Gamma": "orange",
        "Theta": "green",
        "Vega": "purple"
    }


    def plotly_greek_sensitivity(spot_range, combined_values, leg1_values, leg2_values, S, greek_choice, K1, K2,
                                 position1, position2):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spot_range, y=combined_values, mode='lines', name=f"Combined {greek_choice}",
                                 line=dict(color=color_map[greek_choice],width=3)))
        fig.add_trace(go.Scatter(x=spot_range, y=leg2_values, mode='lines',
                         name=f"{position2} leg @ {K2}",
                         line=dict(color='black', dash='dash', width=2)))

        fig.add_trace(go.Scatter(x=spot_range, y=leg1_values, mode='lines',
                         name=f"{position1} leg @ {K1}",
                         line=dict(color='gray', dash='dash', width=2)))

        fig.add_shape(
            type='line',
            x0=S,
            x1=S,
            y0=0,
            y1=1,
            xref='x',
            yref='paper',
            line=dict(color='red', width=2, dash='dash')
        )
        fig.update_layout(title=f"{greek_choice} vs Spot Price (Spread)",
                          xaxis_title="Underlying Price",
                          yaxis_title=greek_choice,
                          showlegend=True)
        return fig


    fig_greek = plotly_greek_sensitivity(
        spot_range, combined_values, leg1_values, leg2_values,
        S, greek_choice, K1, K2, position1, position2
    )
    st.plotly_chart(fig_greek,width=800,height=500)

    st.subheader("Theoretical Value vs Payoff")

    fig2, theo_values, payoff_values = plotly_spread_theo_vs_payoff(
        prices, K1, K2, T, r, vol, premium1, premium2, option_type
    )

    st.plotly_chart(fig2,width=800,height=500)

    df = pd.DataFrame({"Price": prices, "Theoretical Value": theo_values, "Payoff": payoff_values})
    st.download_button("Download Data (CSV)", df.to_csv(index=False), file_name="spread_data.csv")

# Volatility Smile (Synthetic Skew)
st.subheader("Volatility Smile")
smile_strikes = np.linspace(S * 0.8, S * 1.2, 15)

def synthetic_price(S, K, T, r):
    base_vol = 0.2
    skew_strength = 0.002 + (0.01 / T)  # More skew when T is short
    skew = skew_strength * (K - S)
    synthetic_vol = base_vol - skew
    return black_scholes(S, K, T, r, synthetic_vol, 'call')

smile_iv = [implied_volatility(S, k, T, r, synthetic_price(S, k, T, r), 'call') for k in smile_strikes]

def plotly_volatility_smile(strikes, ivs):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strikes, y=ivs, mode='lines+markers', name='IV Smile'))
    fig.update_layout(title="Volatility Smile (Simulated)",
                      xaxis_title="Strike Price",
                      yaxis_title="Implied Volatility",
                      showlegend=False)
    return fig

fig_smile = plotly_volatility_smile(smile_strikes, smile_iv)
st.plotly_chart(fig_smile,width=800,height=500)

   
      
     

