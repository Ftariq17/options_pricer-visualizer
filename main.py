import numpy as np
import matplotlib.pyplot as plt
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

def plot_payoff(prices, payoff, strike_prices):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(prices, payoff, label="Payoff", color="#111AC7")
    for k in strike_prices:
        ax.axvline(k, color="red", linestyle=":", linewidth=1.5)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Underlying Price")
    ax.set_ylabel("Profit / Loss")
    ax.set_title("Options Payoff Diagram")
    ax.legend()
    ax.grid(True)
    return fig


def theo_vs_payoff_plot(prices, K, T, r, sigma, premium, option_type="call", position="Buy"):
    theo_values = [black_scholes(p, K, T, r, sigma, option_type) for p in prices]
    intrinsic = np.maximum(prices - K, 0) if option_type == "call" else np.maximum(K - prices, 0)
    payoff = intrinsic - premium if position == "Buy" else premium - intrinsic

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(prices, theo_values, label="Theoretical Value", color="green")
    ax.plot(prices, payoff, label="Payoff at Expiry", linestyle="--", color="#111AC7")
    ax.axhline(0, color="black", linestyle="--", linewidth=1.5)
    ax.axvline(x=K, color="red", linestyle=":", linewidth=1.5, label="Strike Price")
    ax.set_title("Theoretical Value vs Payoff")
    ax.set_xlabel("Underlying Price")
    ax.set_ylabel("Value / P&L")
    ax.grid(True)
    ax.legend()
    return fig, theo_values, payoff


def spread_theo_vs_payoff_plot(prices, K1, K2, T, r, sigma, premium1, premium2, option_type):
    theo1 = [black_scholes(p, K1, T, r, sigma, option_type) for p in prices]
    theo2 = [black_scholes(p, K2, T, r, sigma, option_type) for p in prices]
    theo_values = np.array(theo1) - np.array(theo2)
    payoff = spread_payoff(prices, K1, K2, premium1, premium2, option_type)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(prices, theo_values, label="Theoretical Value", color="green")
    ax.plot(prices, payoff, label="Payoff at Expiry", linestyle="--", color="#111AC7")
    ax.axhline(0, color="black", linestyle="--", linewidth=1.5)
    ax.axvline(x=K1, color="red", linestyle=":", linewidth=1.5, label="Buy Strike")
    ax.axvline(x=K2, color="purple", linestyle=":", linewidth=1.5, label="Sell Strike")
    ax.set_title("Spread: Theoretical Value vs Payoff")
    ax.set_xlabel("Underlying Price")
    ax.set_ylabel("Value / P&L")
    ax.grid(True)
    ax.legend()
    return fig, theo_values, payoff


# Streamlit App
st.set_page_config(page_title="Options Pricer & Strategy Visualizer", layout="centered")
st.title("Options Pricing and Strategy Visualizer")
st.sidebar.header("Inputs")
strategy = st.sidebar.selectbox("Strategy Type", ["Single Option", "Bull Call Spread", "Bull Put Spread"])
view_mode = st.sidebar.selectbox("View Mode", ["Per Share", "Per 100 Shares"])
multiplier = 100 if view_mode == "Per 100 Shares" else 1
vol_mode = st.sidebar.selectbox("Volatility Input Mode", ["Manual", "Implied Volatility"])

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
    market_price = st.sidebar.number_input("Market Price", value=10.0, step=0.1)
    temp_option_type = "call" if strategy == "Bull Call Spread" else (
        "put" if strategy == "Bull Put Spread" else "call")
    temp_strike = st.sidebar.slider("Strike for IV Estimation (K)", 50, 150, 100)
    vol = implied_volatility(S, temp_strike, T, r, market_price, temp_option_type)

prices = np.linspace(0.5 * S, 1.5 * S, 100)

if strategy == "Single Option":
    premium = black_scholes(S, K, T, r, vol, option_type)
    payoff = single_option_payoff(prices, K, premium, option_type, position)
    fig = plot_payoff(prices, payoff, [K])
    st.subheader("Option Price")
    st.markdown(f"**Premium ({position}):** ${premium * multiplier:.2f}")
    if vol_mode == "Implied Volatility":
        st.markdown(f"**Estimated Implied Volatility:** `{vol:.2%}`")
    st.pyplot(fig)

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
    fig2, theo_values, payoff_values = theo_vs_payoff_plot(prices, K, T, r, vol, premium, option_type, position)
    st.pyplot(fig2)
    df = pd.DataFrame({"Price": prices, "Theoretical Value": theo_values, "Payoff": payoff_values})
    st.download_button("Download Data (CSV)", df.to_csv(index=False), file_name="option_data.csv")

else:
    if strategy == "Bull Put Spread":
        option_type = "put"
        position1, position2 = "Sell", "Buy"  # Short put at higher strike, long put at lower strike
        premium1 = black_scholes(S, K2, T, r, vol, option_type)  # Short leg (K2)
        premium2 = black_scholes(S, K1, T, r, vol, option_type)  # Long leg (K1)
        net_premium = premium1 - premium2  # Net credit

    else:  # Bull Call Spread
        option_type = "call"
        position1, position2 = "Buy", "Sell"
        premium1 = black_scholes(S, K1, T, r, vol, option_type)
        premium2 = black_scholes(S, K2, T, r, vol, option_type)
        net_premium = premium1 - premium2

    payoff = spread_payoff(prices, K1, K2, premium1, premium2, option_type, position1, position2)
    fig = plot_payoff(prices, payoff, [K1, K2])

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

    st.pyplot(fig)

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

    fig_greek, ax_greek = plt.subplots(figsize=(6.5, 4.5))
    ax_greek.plot(spot_range, combined_values, label=f"Combined {greek_choice}", color=color_map[greek_choice])
    ax_greek.plot(spot_range, leg1_values, linestyle="--", label=f"{position1} leg @ {K1}", color="gray")
    ax_greek.plot(spot_range, leg2_values, linestyle="--", label=f"{position2} leg @ {K2}", color="black")
    ax_greek.axvline(S, linestyle="--", color="red", label="Current Spot")
    ax_greek.set_title(f"{greek_choice} vs Spot Price (Spread)")
    ax_greek.set_xlabel("Underlying Price")
    ax_greek.set_ylabel(greek_choice)
    ax_greek.grid(True)
    ax_greek.legend()
    st.pyplot(fig_greek)

    st.subheader("Theoretical Value vs Payoff")
    fig2, theo_values, payoff_values = spread_theo_vs_payoff_plot(prices, K1, K2, T, r, vol, premium1, premium2, option_type)
    st.pyplot(fig2)
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

fig_smile, ax_smile = plt.subplots(figsize=(6.5, 4.5))
ax_smile.plot(smile_strikes, smile_iv, marker='o')
ax_smile.set_title("Volatility Smile (Simulated)")
ax_smile.set_xlabel("Strike Price")
ax_smile.set_ylabel("Implied Volatility")
ax_smile.grid(True)
st.pyplot(fig_smile)