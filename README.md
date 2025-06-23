# 📈 Options Pricer & Strategy Visualizer

An interactive **Streamlit web app** for pricing European options, analyzing Greeks, estimating implied volatility, and visualizing the payoff of common single-leg and multi-leg options strategies including **Bull Call Spreads** and **Bull Put Spreads**.

## 🔍 Features

- 🧠 **Black-Scholes Pricing** for calls and puts  
- 🧮 **Greeks** calculation: Delta, Gamma, Theta, Vega, Rho  
- 🎯 **Implied Volatility Estimation** via Newton-Raphson  
- 💸 Strategy payoff visualization for:
  - Single Option (Buy/Sell Call or Put)
  - Bull Call Spread
  - Bull Put Spread
- 📉 Realistic **Theoretical vs Payoff** plots  
- 📊 **Greek Sensitivity** plots across spot prices  
- 📤 CSV download of computed values  
- 🌀 **Volatility Smile** simulation with synthetic skew  


## 📦 Installation

### Requirements

- Python 3.8+
- `streamlit`
- `numpy`
- `pandas`
- `plotly`
- `scipy`

### Setup

```bash
git clone https://github.com/yourusername/options-pricer-visualizer.git
cd options-pricer-visualizer
pip install -r requirements.txt
streamlit run app.py
