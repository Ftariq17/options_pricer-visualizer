
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import streamlit as st
from scipy.stats import norm
from scipy.optimize import brentq

st.set_page_config(page_title="Options Pricer & Strategy Visualizer", layout="wide")

# -----------------------------
# Utilities
# -----------------------------

@dataclass
class OptionInputs:
    S: float         # Underlying price
    K: float         # Strike
    T: float         # Time to expiry (years)
    r: float         # Risk-free rate (annualized, decimal)
    q: float         # Dividend yield (annualized, decimal)
    sigma: float     # Volatility (annualized, decimal)
    is_call: bool    # True for Call, False for Put

def _ensure_positive(x: float, floor: float = 1e-12) -> float:
    return max(x, floor)

# -----------------------------
# Black–Scholes, Greeks, IV
# -----------------------------

def d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> Tuple[float, float]:
    S = _ensure_positive(S)
    K = _ensure_positive(K)
    T = max(T, 1e-8)
    sigma = max(sigma, 1e-8)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2

def bs_price(inp: OptionInputs) -> float:
    d1, d2 = d1_d2(inp.S, inp.K, inp.T, inp.r, inp.q, inp.sigma)
    disc_r = math.exp(-inp.r * inp.T)
    disc_q = math.exp(-inp.q * inp.T)
    if inp.is_call:
        return disc_q * inp.S * norm.cdf(d1) - disc_r * inp.K * norm.cdf(d2)
    else:
        return disc_r * inp.K * norm.cdf(-d2) - disc_q * inp.S * norm.cdf(-d1)

def bs_greeks(inp: OptionInputs) -> Dict[str, float]:
    d1, d2 = d1_d2(inp.S, inp.K, inp.T, inp.r, inp.q, inp.sigma)
    disc_r = math.exp(-inp.r * inp.T)
    disc_q = math.exp(-inp.q * inp.T)
    pdf_d1 = norm.pdf(d1)

    delta = disc_q * (norm.cdf(d1) if inp.is_call else (norm.cdf(d1) - 1))
    gamma = disc_q * pdf_d1 / (inp.S * inp.sigma * math.sqrt(inp.T))
    vega  = inp.S * disc_q * pdf_d1 * math.sqrt(inp.T)
    theta_call = (
        - (inp.S * disc_q * pdf_d1 * inp.sigma) / (2 * math.sqrt(inp.T))
        - inp.r * inp.K * disc_r * norm.cdf(d2)
        + inp.q * inp.S * disc_q * norm.cdf(d1)
    )
    theta_put = (
        - (inp.S * disc_q * pdf_d1 * inp.sigma) / (2 * math.sqrt(inp.T))
        + inp.r * inp.K * disc_r * norm.cdf(-d2)
        - inp.q * inp.S * disc_q * norm.cdf(-d1)
    )
    theta = theta_call if inp.is_call else theta_put
    rho = inp.K * inp.T * disc_r * (norm.cdf(d2) if inp.is_call else -norm.cdf(-d2))

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega":  float(vega / 100.0),   # per 1% vol
        "theta": float(theta / 365.0),  # per day
        "rho":   float(rho / 100.0),    # per 1% rate
        "d1": float(d1),
        "d2": float(d2),
    }

def implied_vol(target_price: float, base: OptionInputs, low: float = 1e-4, high: float = 5.0) -> Optional[float]:
    # Root find on price(sigma) - target_price = 0
    def f(sig):
        inp = OptionInputs(S=base.S, K=base.K, T=base.T, r=base.r, q=base.q, sigma=max(sig, 1e-8), is_call=base.is_call)
        return bs_price(inp) - target_price

    try:
        # Ensure bracket
        f_low, f_high = f(low), f(high)
        if f_low * f_high > 0:
            # Try widening the bracket
            for h in [10.0, 20.0, 40.0]:
                f_high = f(h)
                if f_low * f_high <= 0:
                    high = h
                    break
        root = brentq(f, low, high, maxiter=200, xtol=1e-10)
        return float(root)
    except Exception:
        return None

# -----------------------------
# Robust strike capture (prevents NameError)
# -----------------------------

def get_strategy_strikes(strategy: str) -> Tuple[Optional[float], Optional[float]]:
    """Return (K1, K2) based on strategy; ensures variables exist before use."""
    if strategy == "Single Option":
        K1 = st.number_input("Strike (K)", value=100.0, step=1.0, key="single_K1")
        K2 = None
    elif strategy == "Bull Call Spread":
        K1 = st.number_input("Long Call Strike (K1)", value=100.0, step=1.0, key="bcs_K1")
        K2 = st.number_input("Short Call Strike (K2)", value=105.0, step=1.0, key="bcs_K2")
    elif strategy == "Bull Put Spread":
        K1 = st.number_input("Short Put Strike (K1)", value=100.0, step=1.0, key="bps_K1")
        K2 = st.number_input("Long Put Strike (K2)", value=95.0, step=1.0, key="bps_K2")
    else:
        K1, K2 = None, None
    return K1, K2

# -----------------------------
# UI Layout
# -----------------------------

st.title("Options Pricer & Strategy Visualizer")

col_a, col_b, col_c = st.columns([1, 1, 1])

with col_a:
    S = st.number_input("Underlying Price (S)", value=100.0, step=0.5, format="%.4f")
    r = st.number_input("Risk-free Rate r (dec.)", value=0.02, step=0.001, format="%.4f")
with col_b:
    q = st.number_input("Dividend Yield q (dec.)", value=0.00, step=0.001, format="%.4f")
    T = st.number_input("Time to Expiry (years)", value=0.5, step=0.01, format="%.4f")
with col_c:
    sigma_default = st.number_input("Volatility σ (dec.)", value=0.20, step=0.01, format="%.4f")
    strategy = st.selectbox("Strategy", ["Single Option", "Bull Call Spread", "Bull Put Spread"])

# Capture strikes EARLY (robust fix)
K1, K2 = get_strategy_strikes(strategy)
iv_K1, iv_K2 = K1, K2  # These are now always defined (K2 may be None for Single Option)

st.divider()

# -----------------------------
# Single Option block
# -----------------------------

if strategy == "Single Option":
    is_call = st.radio("Option Type", ["Call", "Put"], horizontal=True) == "Call"
    use_iv = st.toggle("Estimate implied volatility from market price?")

    if use_iv:
        market_price = st.number_input("Observed Market Price", value=5.0, step=0.1, format="%.4f")
        base = OptionInputs(S=S, K=K1, T=T, r=r, q=q, sigma=sigma_default, is_call=is_call)
        iv = implied_vol(market_price, base)
        if iv is not None:
            sigma = iv
            st.success(f"Implied Volatility: {iv:.4f}")
        else:
            sigma = sigma_default
            st.warning("Could not solve for implied volatility; falling back to input σ.")
    else:
        sigma = sigma_default

    inp = OptionInputs(S=S, K=K1, T=T, r=r, q=q, sigma=sigma, is_call=is_call)
    price = bs_price(inp)
    greeks = bs_greeks(inp)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Price")
        st.metric("Theoretical Price", f"{price:.4f}")
    with c2:
        st.subheader("Greeks")
        st.write({k: round(v, 6) for k, v in greeks.items()})

# -----------------------------
# Bull Call Spread
# -----------------------------

elif strategy == "Bull Call Spread":
    if K2 is None:
        st.error("Please enter both strikes for the spread.")
    else:
        long_sigma = sigma_default
        short_sigma = sigma_default

        # Long Call at K1
        long_inp = OptionInputs(S=S, K=K1, T=T, r=r, q=q, sigma=long_sigma, is_call=True)
        long_price = bs_price(long_inp)

        # Short Call at K2
        short_inp = OptionInputs(S=S, K=K2, T=T, r=r, q=q, sigma=short_sigma, is_call=True)
        short_price = bs_price(short_inp)

        net_debit = long_price - short_price
        st.subheader("Pricing")
        st.write({
            "Long Call (K1)": round(long_price, 6),
            "Short Call (K2)": round(short_price, 6),
            "Net Debit": round(net_debit, 6),
        })

        # Greeks (net)
        long_g = bs_greeks(long_inp)
        short_g = bs_greeks(short_inp)
        net_greeks = {k: long_g[k] - short_g[k] for k in long_g}
        st.subheader("Net Greeks")
        st.write({k: round(v, 6) for k, v in net_greeks.items()})

        # Prob of max profit ~ P(S_T >= K2) under RN measure -> use d2 at K2 for call
        _, d2_k2 = d1_d2(S, K2, T, r, q, sigma_default)
        prob_max_profit = 1 - norm.cdf(d2_k2)
        st.info(f"Approx. P(Max Profit): {prob_max_profit:.4f} (using d2 at K2)")

# -----------------------------
# Bull Put Spread
# -----------------------------

elif strategy == "Bull Put Spread":
    if K2 is None:
        st.error("Please enter both strikes for the spread.")
    else:
        # Convention here: Short higher-strike put (K1), Long lower-strike put (K2)
        short_put_inp = OptionInputs(S=S, K=K1, T=T, r=r, q=q, sigma=sigma_default, is_call=False)
        long_put_inp  = OptionInputs(S=S, K=K2, T=T, r=r, q=q, sigma=sigma_default, is_call=False)

        short_put_price = bs_price(short_put_inp)
        long_put_price  = bs_price(long_put_inp)

        net_credit = short_put_price - long_put_price
        st.subheader("Pricing")
        st.write({
            "Short Put (K1)": round(short_put_price, 6),
            "Long Put (K2)": round(long_put_price, 6),
            "Net Credit": round(net_credit, 6),
        })

        # Net Greeks
        g_short = bs_greeks(short_put_inp)
        g_long  = bs_greeks(long_put_inp)
        net_greeks = {k: g_short[k] - g_long[k] for k in g_short}
        st.subheader("Net Greeks")
        st.write({k: round(v, 6) for k, v in net_greeks.items()})

        # Prob of max profit for Bull Put (credit): want S_T >= K1
        # Use put's d2 at strike K1; P(S_T >= K1) = 1 - N(d2_put_strike_K1) under RN dynamics.
        # For puts, d2 is computed the same; we use 1 - N(d2_K1).
        _, d2_k1 = d1_d2(S, K1, T, r, q, sigma_default)
        prob_max_profit = 1 - norm.cdf(d2_k1)
        st.info(f"Approx. P(Max Profit): {prob_max_profit:.4f} (using d2 at K1)")

# -----------------------------
# Footer / Notes
# -----------------------------

st.caption("Educational tool. Not investment advice. Black–Scholes with continuous dividend yield.")
