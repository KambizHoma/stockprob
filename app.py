import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import levy_stable
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stock Hit Probability | Nippotica",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Nippotica branding
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2em;
    }
    .main-header p {
        color: #e0e7ff;
        margin: 5px 0 0 0;
    }
    .stAlert {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# Header with Nippotica branding
st.markdown("""
<div class="main-header">
    <h1>🎯 Stock Price Hit Probability</h1>
    <p><strong>Nippotica Corporation</strong> | Algotechniq Business Unit | Fat-Tailed Distribution Analysis</p>
</div>
""", unsafe_allow_html=True)

#############################################
# CORE FUNCTIONS
#############################################

def fit_stable_distribution(prices):
    """Calculate log returns and fit stable distribution using fast quantile method"""
    log_returns = np.diff(np.log(prices))
    
    try:
        from scipy.stats._levy_stable import _fitstart
        alpha, beta, loc, scale = _fitstart(log_returns)
    except (ImportError, AttributeError):
        try:
            alpha, beta, loc, scale = levy_stable.fit(log_returns, method='MLE', optimizer={'maxiter': 20})
        except:
            alpha = 1.8
            beta = 0.0
            loc = np.median(log_returns)
            scale = np.std(log_returns) * 0.5
    
    return alpha, beta, loc, scale, log_returns


def calculate_probability(alpha, beta, loc, scale, last_price, future_price, days_ahead):
    """Calculate probability that price will be above/below future_price in days_ahead"""
    log_return = np.log(future_price / last_price)
    
    time_scale_factor = np.sqrt(days_ahead)
    scaled_scale = scale * time_scale_factor
    scaled_loc = loc * days_ahead
    
    prob_lower = levy_stable.cdf(log_return, alpha, beta, loc=scaled_loc, scale=scaled_scale)
    prob_higher = 1 - prob_lower
    
    return prob_lower, prob_higher


def plot_price_series(prices, last_price, scenarios):
    """Plot price series with scenario markers"""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(prices.index, prices.values, linewidth=2, color='#2E86AB', 
            label='Historical Prices', zorder=3)
    
    ax.scatter([prices.index[-1]], [last_price], color='#1e3a8a', s=150, 
               zorder=5, marker='o', label='Current Price', edgecolors='white', linewidths=2)
    
    colors = ['#10b981', '#f59e0b', '#ef4444']  # green, orange, red
    for i, (pct, price, _, _) in enumerate(scenarios):
        ax.axhline(y=price, color=colors[i % len(colors)], 
                   linestyle='--', alpha=0.7, linewidth=2,
                   label=f'{pct:+.1f}%: ${price:.2f}')
    
    ax.set_xlabel("Date", fontsize=12, fontweight='bold')
    ax.set_ylabel("Price ($)", fontsize=12, fontweight='bold')
    ax.set_title("Price History with Target Scenarios", fontsize=14, fontweight='bold', color='#1e3a8a')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10, framealpha=0.95)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig


def plot_distribution_fit(log_returns, alpha, beta, loc, scale):
    """Plot histogram with fitted distribution"""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.hist(log_returns, bins=50, density=True, alpha=0.6, 
            color='#3b82f6', edgecolor='#1e3a8a', label='Historical Log Returns')
    
    x = np.linspace(log_returns.min(), log_returns.max(), 1000)
    pdf = levy_stable.pdf(x, alpha, beta, loc=loc, scale=scale)
    ax.plot(x, pdf, color='#ef4444', linewidth=3, label='Fitted Stable Distribution')
    
    ax.set_xlabel('Log Returns', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title('Distribution Fit Analysis', fontsize=14, fontweight='bold', color='#1e3a8a')
    ax.legend(fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    return fig


def plot_cdf_analysis(log_returns, alpha, beta, loc, scale, last_price, scenarios, days_ahead):
    """Plot CDF with probability markers"""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    time_scale_factor = np.sqrt(days_ahead)
    scaled_scale = scale * time_scale_factor
    scaled_loc = loc * days_ahead
    
    scenario_prices = [price for _, price, _, _ in scenarios]
    scenario_log_returns = [np.log(price/last_price) for price in scenario_prices]
    
    min_log_return = min(max(-0.4, log_returns.min()), min(scenario_log_returns) - 0.05)
    max_log_return = max(min(0.4, log_returns.max()), max(scenario_log_returns) + 0.05)
    
    min_price = last_price * np.exp(min_log_return)
    max_price = last_price * np.exp(max_log_return)
    
    price_range = np.linspace(min_price, max_price, 1000)
    log_returns_range = np.log(price_range / last_price)
    cdf = levy_stable.cdf(log_returns_range, alpha, beta, loc=scaled_loc, scale=scaled_scale)
    
    ax.plot(price_range, cdf, color='#1e3a8a', linewidth=3, label='Cumulative Distribution')
    
    colors = ['#10b981', '#f59e0b', '#ef4444']
    for i, (pct, price, prob_lower, prob_higher) in enumerate(scenarios):
        ax.axvline(price, color=colors[i % len(colors)], 
                  linestyle='--', linewidth=2, alpha=0.7, label=f'{pct:+.1f}%')
        ax.scatter([price], [prob_lower], color=colors[i % len(colors)], 
                  s=150, zorder=5, edgecolors='white', linewidths=2)
    
    ax.set_xlabel('Future Price ($)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax.set_title('Probability Distribution Function', fontsize=14, fontweight='bold', color='#1e3a8a')
    ax.legend(fontsize=10, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([min_price, max_price])
    plt.tight_layout()
    
    return fig


def create_summary_text(symbol, last_price, alpha, beta, loc, scale, scenarios, days_ahead, historical_days):
    """Generate detailed analysis summary"""
    summary = f"""📊 **Analysis Summary for {symbol}**

**Current Price:** ${last_price:.2f}
**Analysis Period:** {historical_days} days of historical data
**Time Horizon:** {days_ahead} days ahead

---

**📈 Stable Distribution Parameters:**

• **α (Alpha) = {alpha:.4f}**
  - Tail heaviness indicator
  - α = 2.0: Normal distribution (thin tails)
  - α < 2.0: Fat tails (extreme events more likely)
  - Lower α = heavier tails

• **β (Beta) = {beta:.4f}**
  - Skewness parameter
  - β = 0: Symmetric
  - β > 0: Right-skewed (upside bias)
  - β < 0: Left-skewed (downside bias)

• **μ (Location) = {loc:.6f}**
  - Central tendency of daily log returns

• **σ (Scale) = {scale:.6f}**
  - Volatility/dispersion measure

---

**🎯 Target Scenario Probabilities:**

"""
    
    for pct, price, prob_lower, prob_higher in scenarios:
        direction = "📈" if pct > 0 else "📉"
        summary += f"""{direction} **{pct:+.1f}% Change → ${price:.2f}**
   • Probability price will be BELOW this target: {prob_lower*100:.2f}%
   • Probability price will be ABOVE this target: {prob_higher*100:.2f}%

"""
    
    summary += f"""---

**⚠️ Important Notes:**

• This analysis uses stable distributions to capture fat tails and extreme events
• Past performance does not guarantee future results
• Market conditions can change rapidly
• Use this as one input among many for decision-making

**📚 For Educational Purposes Only**
"""
    
    return summary


#############################################
# SIDEBAR CONTROLS
#############################################

st.sidebar.header("🎯 Stock Analysis Controls")

# Analyze button at the top
analyze_button = st.sidebar.button("🎯 Calculate Probabilities", type="primary", use_container_width=True)

st.sidebar.markdown("---")

# Stock symbol input
symbol = st.sidebar.text_input(
    "Stock Symbol",
    value="SPY",
    help="Enter ticker symbol (e.g., SPY, AAPL, BTC-USD)"
)

# End date
end_date = st.sidebar.date_input(
    "Analysis End Date",
    value=pd.to_datetime("2024-12-31"),
    help="The analysis will use 1 year of data up to this date"
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Price Target Scenarios")

# Three scenario sliders
scenario_1 = st.sidebar.slider(
    "Target 1: Upside (%)",
    min_value=-50.0,
    max_value=50.0,
    value=3.0,
    step=0.5,
    help="Positive % change scenario"
)

scenario_2 = st.sidebar.slider(
    "Target 2: Stretch Goal (%)",
    min_value=-50.0,
    max_value=50.0,
    value=5.0,
    step=0.5,
    help="Higher upside scenario"
)

scenario_3 = st.sidebar.slider(
    "Target 3: Downside Risk (%)",
    min_value=-50.0,
    max_value=50.0,
    value=-3.0,
    step=0.5,
    help="Negative % change scenario"
)

st.sidebar.markdown("---")

# Advanced options (collapsed by default)
with st.sidebar.expander("⚙️ Advanced Options"):
    days_ahead = st.slider(
        "Time Horizon (Days)",
        min_value=1,
        max_value=30,
        value=15,
        step=1,
        help="How many days in the future?"
    )

st.sidebar.markdown("---")

# View options
st.sidebar.subheader("📊 Display Options")
show_distribution_fit = st.sidebar.checkbox("Show Distribution Fit", value=True)
show_cdf_analysis = st.sidebar.checkbox("Show CDF Analysis", value=True)
show_statistics = st.sidebar.checkbox("Show Detailed Statistics", value=False)

st.sidebar.markdown("---")

# About section
with st.sidebar.expander("ℹ️ About This Tool"):
    st.markdown("""
**Stock Hit Probability**

This tool calculates the odds that a stock will reach your target price using **Stable Distributions** (Lévy α-stable).

**Why Stable Distributions?**

Traditional models assume normal distributions, but real markets have:
- 📉 Fat tails (extreme events happen more often)
- 📊 Skewness (asymmetric returns)
- ⚡ Higher volatility than bell curves predict

**How It Works:**
1. Downloads historical price data
2. Calculates log returns
3. Fits stable distribution parameters
4. Computes hit probabilities for your targets

**For Educational Use Only**
- Past performance ≠ future results
- Markets can change regime
- Consult professionals for investment advice

---

**Created for Nippotica Corporation**
Algotechniq Business Unit

Inspired by Mandelbrot, Fama, and Nolan's work on heavy-tailed distributions.
""")

#############################################
# MAIN ANALYSIS
#############################################

if analyze_button:
    try:
        with st.spinner(f'Downloading data for {symbol}...'):
            # Calculate date range
            end_date_str = end_date.strftime('%Y-%m-%d')
            start_date = (pd.to_datetime(end_date) - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
            
            # Download data
            data = yf.download(symbol, start=start_date, end=end_date_str, progress=False)
            
            if data.empty:
                st.error(f"❌ No data found for symbol '{symbol}'. Please check the symbol and try again.")
                st.stop()
            
            # Handle both old and new Yahoo Finance column formats
            if 'Adj Close' in data.columns:
                prices = data['Adj Close'].dropna()
            elif 'Close' in data.columns:
                prices = data['Close'].dropna()
            else:
                st.error(f"❌ Unable to find price data for '{symbol}'. Columns available: {list(data.columns)}")
                st.stop()
            
            historical_days = len(prices)
            last_price = prices.iloc[-1]
            
            # Fit distribution
            alpha, beta, loc, scale, log_returns = fit_stable_distribution(prices.values)
            
            # Calculate scenarios
            scenarios = []
            for pct in [scenario_1, scenario_2, scenario_3]:
                future_price = last_price * (1 + pct / 100)
                prob_lower, prob_higher = calculate_probability(
                    alpha, beta, loc, scale, last_price, future_price, days_ahead
                )
                scenarios.append((pct, future_price, prob_lower, prob_higher))
            
        # Success message
        st.success(f"✅ Analysis complete for {symbol}! Analyzed {historical_days} days of data.")
        
        # Display results
        st.markdown("---")
        
        # Price chart
        st.subheader("📈 Price History with Target Scenarios")
        fig_price = plot_price_series(prices, last_price, scenarios)
        st.pyplot(fig_price)
        
        # Distribution fit (optional)
        if show_distribution_fit:
            st.markdown("---")
            st.subheader("📊 Distribution Fit Analysis")
            fig_dist = plot_distribution_fit(log_returns, alpha, beta, loc, scale)
            st.pyplot(fig_dist)
        
        # CDF analysis (optional)
        if show_cdf_analysis:
            st.markdown("---")
            st.subheader("📉 Cumulative Probability Function")
            fig_cdf = plot_cdf_analysis(log_returns, alpha, beta, loc, scale, 
                                       last_price, scenarios, days_ahead)
            st.pyplot(fig_cdf)
        
        # Summary text
        st.markdown("---")
        st.subheader("📊 Detailed Analysis")
        summary = create_summary_text(symbol, last_price, alpha, beta, loc, scale, 
                                     scenarios, days_ahead, historical_days)
        st.markdown(summary)
        
        # Optional statistics table
        if show_statistics:
            st.markdown("---")
            st.subheader("📈 Historical Statistics")
            stats_df = pd.DataFrame({
                'Metric': ['Mean Log Return', 'Std Dev Log Return', 'Min Log Return', 
                          'Max Log Return', 'Skewness (β)', 'Tail Index (α)'],
                'Value': [
                    f"{np.mean(log_returns):.6f}",
                    f"{np.std(log_returns):.6f}",
                    f"{np.min(log_returns):.6f}",
                    f"{np.max(log_returns):.6f}",
                    f"{beta:.4f}",
                    f"{alpha:.4f}"
                ]
            })
            st.table(stats_df)
        
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        st.info("💡 Please check your inputs and try again.")

else:
    # Initial state - show instructions
    st.info("""
    👈 **Get Started:**
    
    1. Enter a stock symbol in the sidebar
    2. Adjust the price target scenarios
    3. Click **"Calculate Probabilities"** to analyze
    
    The tool will download 1 year of historical data and calculate hit probabilities using stable distributions.
    """)
    
    # Example scenarios
    st.markdown("---")
    st.subheader("💡 Quick Examples")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **S&P 500 (SPY)**
        - Moderate volatility
        - Good for learning
        - Typical α ≈ 1.8-1.9
        """)
    
    with col2:
        st.markdown("""
        **Apple (AAPL)**
        - Tech stock volatility
        - Corporate events impact
        - Typical α ≈ 1.7-1.9
        """)
    
    with col3:
        st.markdown("""
        **Bitcoin (BTC-USD)**
        - High volatility
        - Fat tail effects
        - Typical α ≈ 1.4-1.7
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 20px;'>
    <p><strong>Nippotica Corporation</strong> | Algotechniq Business Unit</p>
    <p>Stock Price Hit Probability Tool | Stable Distribution Analysis</p>
    <p style='font-size: 0.9em;'>For educational and research purposes only. Not investment advice.</p>
</div>
""", unsafe_allow_html=True)
