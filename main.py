import numpy as np
import plotly.graph_objects as go
import pandas as pd
from scipy.stats import norm
import streamlit as st
from scipy.optimize import brentq

color_map = {
    "Delta": "blue",
    "Gamma": "orange",
    "Theta": "green",
    "Vega": "purple"
}

# Black-Scholes Model
# -----------------------------
def black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "Call") -> float:
    """
    Calculates the Black-Scholes option price.

    Args:
        S (float): Current underlying asset price.
        K (float): Strike price of the option.
        T (float): Time to expiry in years. Must be > 0.
        r (float): Risk-free rate (annualized).
        sigma (float): Volatility (annualized). Must be > 0.
        option_type (str): Type of option ('Call' or 'Put').

    Returns:
        float: Option price. Returns np.nan if inputs are invalid.
    """
    if T <= 1e-6 or sigma <= 1e-6:
        return np.nan
    if S <= 0 or K <= 0:
        return np.nan

    option_type = option_type.capitalize()
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "Call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "Put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        return np.nan


# Greeks Calculation
def greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'Call') -> tuple[dict, float]:
    """
    Calculates the Black-Scholes Greeks (Delta, Gamma, Theta, Vega, Rho).

    Args:
        S (float): Current underlying asset price.
        K (float): Strike price of the option.
        T (float): Time to expiry in years. Must be > 0.
        r (float): Risk-free rate (annualized).
        sigma (float): Volatility (annualized). Must be > 0.
        option_type (str): Type of option ('Call' or 'Put').

    Returns:
        tuple[dict, float]: A dictionary of Greek values and d2.
                          Returns empty dict and np.nan if inputs are invalid.
    """
    if T <= 1e-6 or sigma <= 1e-6:
        return {"delta": np.nan, "gamma": np.nan, "theta": np.nan, "vega": np.nan, "rho": np.nan}, np.nan
    if S <= 0 or K <= 0:
        return {"delta": np.nan, "gamma": np.nan, "theta": np.nan, "vega": np.nan, "rho": np.nan}, np.nan

    option_type = option_type.capitalize()
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta = norm.cdf(d1) if option_type == 'Call' else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
             r * K * np.exp(-r * T) * (norm.cdf(d2) if option_type == 'Call' else -norm.cdf(-d2))) / 365
    vega = (S * norm.pdf(d1) * np.sqrt(T)) / 100  # per 1 vol point
    rho = ((K * T * np.exp(-r * T) *
            (norm.cdf(d2) if option_type == 'Call' else -norm.cdf(-d2)))) / 100  # per 1% rate

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta),
        "vega": float(vega),
        "rho": float(rho),
    }, d2


# Greek Sensitivity Plot for Single Option
def plotly_single_greek_sensitivity(prices: np.ndarray, K: float, T: float, r: float, vol: float,
                                    option_type: str, greek_choice: str, S: float, position: str = "Buy") -> go.Figure:
    sign = 1 if position == "Buy" else -1

    greek_vals = []
    for spot in prices:
        g, _ = greeks(spot, K, T, r, vol, option_type)
        greek_vals.append(sign * g.get(greek_choice.lower(), np.nan))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prices,
        y=greek_vals,
        mode='lines',
        name=f"{position} {option_type.capitalize()} - {greek_choice}",
        line=dict(color=color_map.get(greek_choice, 'blue'), width=3)
    ))
    fig.add_shape(
        type='line',
        x0=S, x1=S,
        y0=0, y1=1,
        xref='x', yref='paper',
        line=dict(color='red', width=2, dash='dash')
    )
    fig.update_layout(
        title=f"{greek_choice} vs Spot Price ({position} {option_type.capitalize()})",
        xaxis_title="Underlying Price",
        yaxis_title=f"{greek_choice} (theta per day; vega/rho per 1%)",
        showlegend=True
    )
    return fig


# Implied Volatility Estimation
def implied_volatility(S: float, K: float, T: float, r: float, market_price: float,
                       option_type: str = 'Call', tol: float = 1e-5, max_iter: int = 100,
                       initial_sigma: float = 0.2) -> float:
    """
    Estimates the implied volatility of an option using Newton-Raphson.
    Returns np.nan if convergence fails or inputs are invalid.
    """
    sigma = initial_sigma
    option_type = option_type.capitalize()
    for _ in range(max_iter):
        try:
            if T <= 1e-6 or sigma <= 1e-6:
                return np.nan
            price = black_scholes(S, K, T, r, sigma, option_type)
            if np.isnan(price):
                return np.nan
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            vega = S * norm.pdf(d1) * np.sqrt(T)
            if vega < tol:
                return np.nan
            diff = price - market_price
            if abs(diff) < tol:
                return sigma
            sigma -= diff / vega
            sigma = max(0.001, sigma)
        except (ZeroDivisionError, ValueError):
            return np.nan
    return np.nan


# --- New Function for Spread Pricing ---
def get_spread_price_for_iv(sigma: float, S: float, K1: float, K2: float, T: float, r: float,
                            option_type: str, position1: str, position2: str) -> float:
    """
    Calculates theoretical price of a two-legged spread for a given single volatility.
    """
    if sigma <= 0 or T <= 0:
        return np.nan

    option_type = option_type.capitalize()
    price1 = black_scholes(S, K1, T, r, sigma, option_type)
    price2 = black_scholes(S, K2, T, r, sigma, option_type)
    if np.isnan(price1) or np.isnan(price2):
        return np.nan

    net_price = 0.0
    net_price += price1 if position1 == "Buy" else -price1
    net_price += price2 if position2 == "Buy" else -price2
    return net_price


def implied_volatility_spread(S: float, K1: float, K2: float, T: float, r: float,
                              market_spread_premium: float, option_type: str,
                              position1: str, position2: str,
                              tol: float = 1e-5, max_iter: int = 100) -> float:
    """
    Find a single volatility that prices the spread at the observed market net premium.
    """
    if T <= 1e-6:
        return np.nan

    def func_to_solve(sigma_val):
        # (theoretical spread price - market premium); root at zero
        return get_spread_price_for_iv(sigma_val, S, K1, K2, T, r, option_type, position1, position2) - market_spread_premium

    low_vol_price = func_to_solve(0.01)
    high_vol_price = func_to_solve(5.0)

    if np.isnan(low_vol_price) or np.isnan(high_vol_price):
        return np.nan
    # FIX: correct bracketing check
    if low_vol_price * high_vol_price > 0:
        return np.nan  # Root not bracketed

    try:
        implied_v = brentq(func_to_solve, 0.01, 5.0, xtol=tol, maxiter=max_iter)
        return implied_v if 0.001 <= implied_v <= 5.0 else np.nan
    except (ValueError, RuntimeError):
        return np.nan


# Payoff Calculation Functions
def single_option_payoff(prices: np.ndarray, K: float, premium: float, option_type: str = "Call",
                         position: str = "Buy") -> np.ndarray:
    option_type = option_type.capitalize()
    if option_type == "Call":
        intrinsic = np.maximum(prices - K, 0)
    elif option_type == "Put":
        intrinsic = np.maximum(K - prices, 0)
    else:
        return np.full_like(prices, np.nan)

    if position == "Buy":
        return intrinsic - premium
    elif position == "Sell":
        return premium - intrinsic
    else:
        return np.full_like(prices, np.nan)


def spread_payoff(prices: np.ndarray, K1: float, K2: float, premium1: float, premium2: float,
                  option_type: str = "Call", position1: str = "Buy", position2: str = "Sell") -> np.ndarray:
    payoff1 = single_option_payoff(prices, K1, premium1, option_type, position=position1)
    payoff2 = single_option_payoff(prices, K2, premium2, option_type, position=position2)
    return payoff1 + payoff2


# Plotting Functions
def plotly_payoff(prices: np.ndarray, payoff: np.ndarray, strike_prices: list, S: float | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices, y=payoff, mode='lines', name='Payoff', line=dict(color='#111AC7', width=3)))
    for k in strike_prices:
        fig.add_shape(
            type='line',
            x0=k, x1=k, y0=0, y1=1,
            xref='x', yref='paper',
            line=dict(color='red', width=2, dash='dot')
        )
    if S is not None:
        fig.add_shape(
            type='line',
            x0=S, x1=S, y0=0, y1=1,
            xref='x', yref='paper',
            line=dict(color='black', width=2, dash='dash')
        )
    fig.update_layout(title='Options Payoff Diagram',
                      xaxis_title='Underlying Price',
                      yaxis_title='Profit / Loss',
                      showlegend=True)
    return fig


def plotly_theo_vs_payoff(prices: np.ndarray, K: float, T: float, r: float, sigma: float, premium: float,
                          option_type: str = "Call", position: str = "Buy", S: float | None = None) -> tuple[go.Figure, list, np.ndarray]:
    theo_values = [black_scholes(p, K, T, r, sigma, option_type) for p in prices]
    payoff = single_option_payoff(prices, K, premium, option_type, position)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices, y=theo_values, mode='lines', name='Theoretical Value', line=dict(color='green', width=3)))
    fig.add_trace(go.Scatter(x=prices, y=payoff, mode='lines', name='Payoff at Expiry', line=dict(color='#111AC7', dash='dash')))
    fig.add_shape(type='line', x0=K, x1=K, y0=0, y1=1, xref='x', yref='paper', line=dict(color='red', width=2, dash='dot'))
    if S is not None:
        fig.add_shape(type='line', x0=S, x1=S, y0=0, y1=1, xref='x', yref='paper', line=dict(color='black', width=2, dash='dash'))
    fig.add_shape(type='line', x0=min(prices), x1=max(prices), y0=0, y1=0, line=dict(color='black', width=2, dash='dash'))
    fig.update_layout(title='Theoretical Value vs Payoff',
                      xaxis_title='Underlying Price',
                      yaxis_title='Value / P&L',
                      showlegend=True)
    return fig, theo_values, payoff


def plotly_spread_theo_vs_payoff(prices: np.ndarray, K1: float, K2: float, T: float, r: float, sigma: float,
                                 premium1: float, premium2: float, option_type: str, position1: str, position2: str,
                                 S: float | None = None) -> tuple[go.Figure, np.ndarray, np.ndarray]:
    theo_values = []
    for p in prices:
        theo_spread_price = get_spread_price_for_iv(sigma, p, K1, K2, T, r, option_type, position1, position2)
        theo_values.append(theo_spread_price)

    payoff = spread_payoff(prices, K1, K2, premium1, premium2, option_type, position1, position2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices, y=theo_values, mode='lines', name='Theoretical Value', line=dict(color='green', width=3)))
    fig.add_trace(go.Scatter(x=prices, y=payoff, mode='lines', name='Payoff at Expiry', line=dict(color='#111AC7', dash='dash', width=3)))
    fig.add_shape(type='line', x0=K1, x1=K1, y0=0, y1=1, xref='x', yref='paper', line=dict(color='red', width=2, dash='dot'))
    fig.add_shape(type='line', x0=K2, x1=K2, y0=0, y1=1, xref='x', yref='paper', line=dict(color='purple', width=2, dash='dot'))
    if S is not None:
        fig.add_shape(type='line', x0=S, x1=S, y0=0, y1=1, xref='x', yref='paper', line=dict(color='black', width=2, dash='dash'))
    fig.add_shape(type='line', x0=min(prices), x1=max(prices), y0=0, y1=0, line=dict(color='black', width=2, dash='dash'))
    fig.update_layout(title='Spread: Theoretical Value vs Payoff',
                      xaxis_title='Underlying Price',
                      yaxis_title='Value / P&L',
                      showlegend=True)
    return fig, np.array(theo_values), payoff


def plotly_greek_sensitivity(spot_range: np.ndarray, combined_values: list, leg1_values: list, leg2_values: list,
                             S: float, greek_choice: str, K1: float, K2: float,
                             position1: str, position2: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spot_range, y=combined_values, mode='lines', name=f"Combined {greek_choice}",
                             line=dict(color=color_map.get(greek_choice, 'blue'), width=3)))
    fig.add_trace(go.Scatter(x=spot_range, y=leg1_values, mode='lines', name=f"{position1} leg @ {K1}",
                             line=dict(color='gray', dash='dash', width=2)))
    fig.add_trace(go.Scatter(x=spot_range, y=leg2_values, mode='lines', name=f"{position2} leg @ {K2}",
                             line=dict(color='black', dash='dash', width=2)))
    fig.add_shape(type='line', x0=S, x1=S, y0=0, y1=1, xref='x', yref='paper', line=dict(color='red', width=2, dash='dash'))
    fig.update_layout(title=f"{greek_choice} vs Spot Price (Spread)",
                      xaxis_title="Underlying Price",
                      yaxis_title=f"{greek_choice} (theta per day; vega/rho per 1%)",
                      showlegend=True)
    return fig


def plotly_volatility_smile(strikes: np.ndarray, ivs: list, colors: list) -> go.Figure:
    fig = go.Figure()
    for k, iv, color in zip(strikes, ivs, colors):
        fig.add_trace(go.Scatter(
            x=[k], y=[iv],
            mode='markers',
            marker=dict(color=color, size=8),
            showlegend=False
        ))
    fig.add_trace(go.Scatter(
        x=strikes, y=ivs,
        mode='lines',
        name='IV Smile',
        line=dict(color='gray', width=2, dash='dash')
    ))
    fig.update_layout(
        title="Volatility Smile (Synthetic, Colored by Moneyness)",
        xaxis_title="Strike Price",
        yaxis_title="Implied Volatility",
        yaxis_tickformat=".2%",
        showlegend=False
    )
    return fig


# Streamlit App
st.set_page_config(page_title="Options Pricer & Strategy Visualizer", layout="centered")
st.title("Options Pricing and Strategy Visualizer")
st.sidebar.header("Inputs")

strategy = st.sidebar.selectbox("Strategy Type", ["Single Option", "Bull Call Spread", "Bull Put Spread"])

# Unified volatility input mode logic for both single and spread
vol_mode = st.sidebar.selectbox("Volatility Input Mode", ["Manual", "Implied Volatility"])

view_mode = st.sidebar.selectbox("View Mode", ["Per Share", "Per 100 Shares"])
multiplier = 100 if view_mode == "Per 100 Shares" else 1

if strategy == "Single Option":
    option_type = st.sidebar.radio("Option Type", ["Call", "Put"])
    position = st.sidebar.radio("Position", ["Buy", "Sell"])
else:
    if strategy == "Bull Call Spread":
        option_type = "Call"
    elif strategy == "Bull Put Spread":
        option_type = "Put"
    position = "Spread"  # display-only

# --- Market Price Input ---
typed_market_price = st.sidebar.number_input("Enter Market Price (S)", value=100.0, step=1.0, format="%.2f")

# Dynamically define spot price slider range
spot_range_width = typed_market_price * 0.5
min_spot_slider = max(1.0, typed_market_price - spot_range_width)
max_spot_slider = typed_market_price + spot_range_width

# Final market price slider (centered around input)
S = st.sidebar.slider("Adjust Market Price (S)",
                      min_value=float(round(min_spot_slider, 2)),
                      max_value=float(round(max_spot_slider, 2)),
                      value=float(round(typed_market_price, 2)))

# --- Strategy-dependent Strike Inputs with Dynamic Range ---
if strategy == "Single Option":
    typed_K = st.sidebar.number_input("Enter Strike Price (K)", value=float(round(S, 2)), step=1.0, format="%.2f")
    strike_slider_width = typed_K * 0.3
    min_K = max(1.0, typed_K - strike_slider_width)
    max_K = typed_K + strike_slider_width

    K = st.sidebar.slider("Adjust Strike Price (K)",
                          min_value=float(round(min_K, 2)),
                          max_value=float(round(max_K, 2)),
                          value=float(round(typed_K, 2)))

else:  # Spread Strategies
    # Ensure K1 < K2 for bull spreads
    default_K1 = float(round(S * 0.9, 2))
    default_K2 = float(round(S * 1.1, 2))
    if default_K1 >= default_K2:
        default_K1 = max(1.0, S - 10)
        default_K2 = S + 10

    typed_K1 = st.sidebar.number_input("Enter Lower Strike (Buy K1)", value=default_K1, step=1.0, format="%.2f")
    typed_K2 = st.sidebar.number_input("Enter Upper Strike (Sell K2)", value=default_K2, step=1.0, format="%.2f")

    K1_slider_width = typed_K1 * 0.3
    K2_slider_width = typed_K2 * 0.3

    min_K1 = max(1.0, typed_K1 - K1_slider_width)
    max_K1 = typed_K1 + K1_slider_width

    K1 = st.sidebar.slider("Adjust Lower Strike (K1)",
                           min_value=float(round(min_K1, 2)),
                           max_value=float(round(max_K1, 2)),
                           value=float(round(typed_K1, 2)))

    # K2 must be greater than K1
    min_K2_slider = K1 + 0.01
    max_K2_slider = typed_K2 + K2_slider_width
    slider_K2_value = float(round(typed_K2, 2))
    if slider_K2_value <= K1:
        slider_K2_value = min_K2_slider

    K2 = st.sidebar.slider("Adjust Upper Strike (K2)",
                           min_value=float(round(min_K2_slider, 2)),
                           max_value=float(round(max_K2_slider, 2)),
                           value=slider_K2_value)

# --- Time to Expiry Input ---
use_days = st.sidebar.checkbox("Input Time to Expiry in Days", value=True)

if use_days:
    days_to_expiry = st.sidebar.number_input("Days to Expiry", value=30, min_value=1)
    T = days_to_expiry / 365
    if T < 1e-6: T = 1e-6
else:
    T = st.sidebar.slider("Time to Expiry (Years)", 0.01, 2.0, 0.25, step=0.01)
    if T < 1e-6: T = 1e-6

r = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0) / 100

vol = 0.20  # Initialize vol with a default value

# --- Volatility Input Logic ---
if vol_mode == "Manual":
    vol = st.sidebar.slider("Volatility (%)", 1.0, 100.0, 20.0) / 100
else:  # Implied Volatility mode
    st.sidebar.markdown("---")
    st.sidebar.subheader("Implied Volatility Calculation")

    if strategy == "Single Option":
        market_price_for_iv = st.sidebar.number_input("Market Premium (for IV calculation)", value=1.00, step=0.01, format="%.2f")
        calculated_vol = implied_volatility(S, K, T, r, market_price_for_iv, option_type)
        if not np.isnan(calculated_vol):
            vol = calculated_vol
            st.sidebar.markdown(f"**Estimated Implied Volatility:** `{vol:.2%}`")
        else:
            vol = st.sidebar.slider("Volatility (%) (Fallback, IV failed)", 1.0, 100.0, 20.0) / 100
            st.sidebar.warning("Could not estimate implied volatility. This can happen with very short expiry, zero premium, or very far OTM options. Using manual volatility.")
    else:
        st.sidebar.markdown("For spreads, the 'Implied Volatility' refers to a single volatility that would price the *entire spread* at the given market net premium.")

    market_net_premium_for_iv = st.sidebar.number_input("Market Net Premium (for IV calculation)", value=1.00, step=0.01, format="%.2f")

    if strategy == "Bull Call Spread":
        iv_option_type = "Call"
        iv_K1, iv_K2 = K1, K2
        iv_position1, iv_position2 = "Buy", "Sell"
        st.sidebar.write(f"{iv_position1} Call @ {iv_K1:.2f}, {iv_position2} Call @ {iv_K2:.2f}")

    elif strategy == "Bull Put Spread":
        iv_option_type = "Put"
        iv_K1, iv_K2 = K1, K2
        # FIX: long lower (K1), short higher (K2)
        iv_position1, iv_position2 = "Buy", "Sell"
        st.sidebar.write(f"{iv_position1} Put @ {iv_K1:.2f}, {iv_position2} Put @ {iv_K2:.2f}")

    else:
        iv_option_type = "Call"
        iv_K1, iv_K2 = K1, K2
        iv_position1, iv_position2 = "Buy", "Sell"

    # FIX: no abs() on market premium
    calculated_vol_spread = implied_volatility_spread(
        S, iv_K1, iv_K2, T, r, market_net_premium_for_iv,
        iv_option_type, iv_position1, iv_position2
    )

    if not np.isnan(calculated_vol_spread):
        vol = calculated_vol_spread
        st.sidebar.markdown(f"**Estimated Implied Volatility (for Spread):** `{vol:.2%}`")
    else:
        vol = st.sidebar.slider("Volatility (%) (Fallback, IV failed for spread)", 1.0, 100.0, 20.0) / 100
        st.sidebar.warning("Could not estimate implied volatility for the spread. This can happen with very short expiry or if the market premium is unrealistic. Using manual volatility.")

    st.sidebar.markdown("---")

prices = np.linspace(max(1.0, 0.5 * S), 1.5 * S, 100)

if strategy == "Single Option":
    use_market_price = st.sidebar.checkbox("Use Market Price (Override BSM)", value=False)

    premium = black_scholes(S, K, T, r, vol, option_type)
    if np.isnan(premium):
        st.error("Cannot calculate theoretical premium due to invalid inputs (e.g., Time to Expiry too short, Volatility too low/high, or Spot/Strike issues). Please adjust inputs.")
        premium = 0.0
    else:
        if use_market_price:
            premium = st.sidebar.number_input(f"Market Premium ({position} {option_type.capitalize()} @ {K:.2f})({view_mode})",
                                              value=float(premium), step=0.01, format="%.2f")
            # Re-solve IV from entered market premium so Greeks/curves match
            new_iv = implied_volatility(S, K, T, r, premium, option_type)
            if not np.isnan(new_iv):
                vol = new_iv

    payoff = single_option_payoff(prices, K, premium, option_type, position)
    fig = plotly_payoff(prices, payoff, [K], S=S)

    st.subheader(f"{strategy} Price & Payoff")
    st.markdown(f"**Premium ({position}):** ${premium * multiplier:.2f}")
    st.plotly_chart(fig, use_container_width=True)

    greeks_vals, d2 = greeks(S, K, T, r, vol, option_type)
    st.subheader("Strategy Summary (Single Option)")

    if np.isnan(premium) or np.isnan(greeks_vals["delta"]):
        st.warning("Cannot calculate summary and Greeks due to invalid inputs. Please check input parameters like Time to Expiry or Volatility.")
    else:
        if option_type == 'Call':
            if position == "Buy":
                breakeven = K + premium
                max_profit = float('inf')
                max_loss = premium
            else:
                breakeven = K + premium
                max_profit = premium
                max_loss = float('inf')
        else:
            if position == "Buy":
                breakeven = K - premium
                max_profit = K - premium
                max_loss = premium
            else:
                breakeven = K - premium
                max_profit = premium
                max_loss = K - premium

        prob_itm = norm.cdf(d2) if option_type == 'Call' else norm.cdf(-d2) if not np.isnan(d2) else np.nan

        st.markdown(f"**Breakeven Price:** ${breakeven:.2f}")
        st.markdown(f"**Max Potential Profit:** {'Unlimited' if max_profit == float('inf') else f'${max_profit * multiplier:.2f}'}")
        st.markdown(f"**Max Potential Loss:** {'Unlimited' if max_loss == float('inf') else f'${max_loss * multiplier:.2f}'}")
        st.markdown(f"**Estimated Probability ITM:** {prob_itm * 100:.2f}%" if not np.isnan(prob_itm) else "N/A")

        st.subheader("Greeks (per share)")
        for key, val in greeks_vals.items():
            if not np.isnan(val):
                unit = " (per day)" if key == "theta" else " (per 1%)" if key in ("vega", "rho") else ""
                st.markdown(f"{key.capitalize()}: {val:.4f}{unit}")
            else:
                st.markdown(f"{key.capitalize()}: N/A (Invalid Inputs)")

        st.subheader("Greek Sensitivity vs Spot Price")
        greek_choice = st.selectbox("Select Greek to Plot (Single Option)", ["Delta", "Gamma", "Theta", "Vega"], key="single_greek")
        fig_single_greek = plotly_single_greek_sensitivity(prices, K, T, r, vol, option_type, greek_choice, S, position)
        st.plotly_chart(fig_single_greek, use_container_width=True)

        st.subheader("Theoretical Value vs Payoff")
        fig2, theo_values, payoff_values = plotly_theo_vs_payoff(prices, K, T, r, vol, premium, option_type, position, S=S)
        st.plotly_chart(fig2, use_container_width=True)
        df = pd.DataFrame({"Price": prices, "Theoretical Value": theo_values, "Payoff": payoff_values})
        st.download_button("Download Data (CSV)", df.to_csv(index=False).encode('utf-8'), file_name="option_data.csv", mime="text/csv")

elif strategy in ["Bull Put Spread", "Bull Call Spread"]:
    use_market_price = st.sidebar.checkbox("Use Market Prices for Legs", value=False)

    if strategy == "Bull Put Spread":
        option_type = "Put"
        K_buy = K1   # Long Put at Lower Strike
        K_sell = K2  # Short Put at Higher Strike
        position1 = "Buy"
        position2 = "Sell"

        premium_buy = black_scholes(S, K_buy, T, r, vol, option_type)
        premium_sell = black_scholes(S, K_sell, T, r, vol, option_type)

        if np.isnan(premium_buy) or np.isnan(premium_sell):
            st.error("Cannot calculate theoretical premiums for spread due to invalid inputs (e.g., Time to Expiry too short, Volatility too low/high). Please adjust inputs.")
            premium_buy = 0.0
            premium_sell = 0.0
        else:
            st.sidebar.markdown("---")
            st.sidebar.subheader("Leg Premiums")
            if use_market_price:
                premium_buy = st.sidebar.number_input(f"Market Premium (Long Put @ {K_buy:.2f})", value=float(premium_buy), step=0.01, format="%.2f")
                premium_sell = st.sidebar.number_input(f"Market Premium (Short Put @ {K_sell:.2f})", value=float(premium_sell), step=0.01, format="%.2f")
            else:
                st.sidebar.markdown(f"**Theoretical Premium (Long Put @ {K_buy:.2f}):** ${premium_buy * multiplier:.2f}")
                st.sidebar.markdown(f"**Theoretical Premium (Short Put @ {K_sell:.2f}):** ${premium_sell * multiplier:.2f}")

        net_premium = premium_sell - premium_buy  # Credit received
        payoff = spread_payoff(prices, K_buy, K_sell, premium_buy, premium_sell, option_type, position1, position2)

        st.subheader("Bull Put Spread Payoff")
        st.markdown(f"**Net Premium Received (Credit):** ${net_premium * multiplier:.2f}")

        fig = plotly_payoff(prices, payoff, sorted([K_buy, K_sell]), S=S)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Strategy Summary (Bull Put Spread)")
        breakeven = K_sell - net_premium
        max_profit = net_premium
        max_loss = (K_sell - K_buy) - net_premium

        # Probability for Max Profit (S_T >= K_sell)
        _, d2_K_sell = greeks(S, K_sell, T, r, vol, option_type)
        prob_max_profit = norm.cdf(d2_K_sell)

        st.markdown(f"**Width:** {K_sell - K_buy:.2f}")
        st.markdown(f"**Breakeven Price:** ${breakeven:.2f}")
        st.markdown(f"**Max Potential Profit:** ${max_profit * multiplier:.2f}")
        st.markdown(f"**Max Potential Loss:** ${max_loss * multiplier:.2f}")
        st.markdown(f"**Estimated Probability of Max Profit (S ≥ {K_sell:.2f}):** {prob_max_profit * 100:.2f}%" if not np.isnan(prob_max_profit) else "N/A")

    elif strategy == "Bull Call Spread":
        option_type = "Call"
        K_buy = K1   # Long Call at Lower Strike
        K_sell = K2  # Short Call at Higher Strike
        position1 = "Buy"
        position2 = "Sell"

        premium_buy = black_scholes(S, K_buy, T, r, vol, option_type)
        premium_sell = black_scholes(S, K_sell, T, r, vol, option_type)

        if np.isnan(premium_buy) or np.isnan(premium_sell):
            st.error("Cannot calculate theoretical premiums for spread due to invalid inputs (e.g., Time to Expiry too short, Volatility too low/high). Please adjust inputs.")
            premium_buy = 0.0
            premium_sell = 0.0
        else:
            st.sidebar.markdown("---")
            st.sidebar.subheader("Leg Premiums")
            if use_market_price:
                premium_buy = st.sidebar.number_input(f"Market Premium (Long Call @ {K_buy:.2f})", value=float(premium_buy), step=0.01, format="%.2f")
                premium_sell = st.sidebar.number_input(f"Market Premium (Short Call @ {K_sell:.2f})", value=float(premium_sell), step=0.01, format="%.2f")
            else:
                st.sidebar.markdown(f"**Theoretical Premium (Long Call @ {K_buy:.2f}):** ${premium_buy * multiplier:.2f}")
                st.sidebar.markdown(f"**Theoretical Premium (Short Call @ {K_sell:.2f}):** ${premium_sell * multiplier:.2f}")

        net_premium = premium_buy - premium_sell  # Debit paid
        payoff = spread_payoff(prices, K_buy, K_sell, premium_buy, premium_sell, option_type, position1, position2)

        st.subheader("Bull Call Spread Payoff")
        st.markdown(f"**Net Premium Paid (Debit):** ${net_premium * multiplier:.2f}")

        fig = plotly_payoff(prices, payoff, sorted([K_buy, K_sell]), S=S)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Strategy Summary (Bull Call Spread)")
        breakeven = K_buy + net_premium
        max_profit = (K_sell - K_buy) - net_premium
        max_loss = net_premium

        # Probability for Max Profit (S_T ≥ K_sell)
        _, d2_K_sell = greeks(S, K_sell, T, r, vol, option_type)
        prob_max_profit = norm.cdf(d2_K_sell) if not np.isnan(d2_K_sell) else np.nan

        st.markdown(f"**Width:** {K_sell - K_buy:.2f}")
        st.markdown(f"**Breakeven Price:** ${breakeven:.2f}")
        st.markdown(f"**Max Potential Profit:** ${max_profit * multiplier:.2f}")
        st.markdown(f"**Max Potential Loss:** ${max_loss * multiplier:.2f}")
        st.markdown(f"**Estimated Probability of Max Profit (S ≥ {K_sell:.2f}):** {prob_max_profit * 100:.2f}%" if not np.isnan(prob_max_profit) else "N/A")

    # Greeks for each leg
    greeks_leg_buy, _ = greeks(S, K_buy, T, r, vol, option_type)
    greeks_leg_sell, _ = greeks(S, K_sell, T, r, vol, option_type)

    if any(np.isnan(val) for val in greeks_leg_buy.values()) or any(np.isnan(val) for val in greeks_leg_sell.values()):
        st.warning("Cannot calculate combined Greeks due to invalid inputs for one or both legs. Please check input parameters like Time to Expiry or Volatility.")
        total_greeks = {greek: np.nan for greek in greeks_leg_buy}
    else:
        sign1 = 1   # long K_buy
        sign2 = -1  # short K_sell
        total_greeks = {
            greek: greeks_leg_buy[greek] * sign1 + greeks_leg_sell[greek] * sign2
            for greek in greeks_leg_buy
        }

    st.subheader("Combined Greeks (Strategy-Level, per share)")
    for key, val in total_greeks.items():
        if not np.isnan(val):
            unit = " (per day)" if key == "theta" else " (per 1%)" if key in ("vega", "rho") else ""
            st.markdown(f"{key.capitalize()}: {val:.4f}{unit}")
        else:
            st.markdown(f"{key.capitalize()}: N/A (Invalid Inputs)")

    # Greek Sensitivity vs Spot Price Plot for Spreads
    st.subheader("Greek Sensitivity vs Spot Price")
    greek_choice = st.selectbox("Select Greek to Plot", ["Delta", "Gamma", "Theta", "Vega"], key="spread_greek_plot")

    spot_range = np.linspace(max(1.0, 0.5 * S), 1.5 * S, 100)
    combined_values, leg1_values, leg2_values = [], [], []

    sign1, sign2 = 1, -1  # long lower, short higher
    for spot in spot_range:
        g1, _ = greeks(spot, K_buy, T, r, vol, option_type)
        g2, _ = greeks(spot, K_sell, T, r, vol, option_type)
        val1 = g1.get(greek_choice.lower(), np.nan) * sign1
        val2 = g2.get(greek_choice.lower(), np.nan) * sign2
        leg1_values.append(val1)
        leg2_values.append(val2)
        combined_values.append(val1 + val2)

    fig_greek = plotly_greek_sensitivity(
        spot_range, combined_values, leg1_values, leg2_values,
        S, greek_choice, K_buy, K_sell, position1, position2
    )
    st.plotly_chart(fig_greek, use_container_width=True)

    st.subheader("Theoretical Value vs Payoff")
    fig2, theo_values, payoff_values = plotly_spread_theo_vs_payoff(
        prices, K_buy, K_sell, T, r, vol, premium_buy, premium_sell, option_type, position1, position2, S=S
    )
    st.plotly_chart(fig2, use_container_width=True)

    df = pd.DataFrame({"Price": prices, "Theoretical Value": theo_values, "Payoff": payoff_values})
    st.download_button("Download Data (CSV)", df.to_csv(index=False).encode('utf-8'), file_name="spread_data.csv", mime="text/csv")

# --- Volatility Smile (Synthetic Skew) ---
st.subheader("Volatility Smile")

# Toggle: Color code smile by moneyness
color_smile = st.checkbox("Color code by moneyness", value=True)

# Define a realistic range around spot
smile_strikes = np.linspace(max(1.0, S * 0.7), S * 1.3, 31)

# Synthetic smile generation
def synthetic_vol(S: float, K: float, base_vol: float = 0.20, skew_strength: float = 0.0005, atm_shift: float = 0.0) -> float:
    shifted_K = K - (S * atm_shift)
    curve = skew_strength * ((shifted_K - S) / S) ** 2
    return max(0.05, min(1.0, base_vol + curve))

smile_iv = []
for K_smile in smile_strikes:
    base_for_smile = vol if 'vol' in locals() and not np.isnan(vol) else 0.20
    vol_used = synthetic_vol(S, K_smile, base_vol=base_for_smile)
    price_for_iv = black_scholes(S, K_smile, T, r, vol_used, 'Call')
    if not np.isnan(price_for_iv) and price_for_iv > 0:
        iv = implied_volatility(S, K_smile, T, r, price_for_iv, 'Call')
        smile_iv.append(iv)
    else:
        smile_iv.append(np.nan)

# Filter clean data
plot_strikes = [s for s, iv_val in zip(smile_strikes, smile_iv) if not np.isnan(iv_val)]
plot_ivs = [iv_val for iv_val in smile_iv if not np.isnan(iv_val)]
any_failed_iv = any(np.isnan(iv_val) for iv_val in smile_iv)

# Plot if enough data
if len(plot_strikes) > 1:
    if color_smile:
        moneyness = []
        colors = []
        for k in plot_strikes:
            if k < S:
                moneyness.append("ITM Call / OTM Put")
                colors.append("green")
            elif k > S:
                moneyness.append("OTM Call / ITM Put")
                colors.append("red")
            else:
                moneyness.append("ATM")
                colors.append("blue")

        fig = go.Figure()
        for k, iv, c in zip(plot_strikes, plot_ivs, colors):
            fig.add_trace(go.Scatter(
                x=[k], y=[iv],
                mode='markers',
                marker=dict(color=c, size=8),
                name="",
                showlegend=False
            ))
        fig.add_trace(go.Scatter(
            x=plot_strikes, y=plot_ivs,
            mode='lines',
            name='IV Smile',
            line=dict(color='gray', dash='dash', width=2)
        ))
        fig.update_layout(
            title="Volatility Smile (Colored by Moneyness)",
            xaxis_title="Strike Price",
            yaxis_title="Implied Volatility",
            yaxis_tickformat=".2%",
            showlegend=False
        )
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_strikes, y=plot_ivs,
            mode='lines+markers',
            name='IV Smile',
            line=dict(color='blue', width=2)
        ))
        fig.update_layout(
            title="Volatility Smile (Simulated)",
            xaxis_title="Strike Price",
            yaxis_title="Implied Volatility",
            yaxis_tickformat=".2%",
            showlegend=False
        )

    if any_failed_iv:
        st.warning("⚠️ Some implied volatilities could not be calculated. Plot may include fallback synthetic vols.")
    else:
        st.markdown("✅ All implied volatilities were successfully backsolved from synthetic prices.")

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Not enough valid data points to plot the Volatility Smile. Please check inputs.")
