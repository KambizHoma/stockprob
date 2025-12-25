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

# Custom CSS for clean header
st.markdown("""
<style>
    .main-header {
        background: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 2px solid #e5e7eb;
    }
    .main-header h1 {
        color: #111827;
        margin: 0;
        font-size: 2em;
    }
    .main-header p {
        color: #4b5563;
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
    <p><strong>Nippotica Corporation</strong> | Nippofin Business Unit | Fat-Tailed Distribution Analysis</p>
</div>
""", unsafe_allow_html=True)

#############################################
# CORE FUNCTIONS
#############################################

def fit_stable_distribution(prices):
    """Calculate log returns and fit stable distribution using fast quantile method"""
    # Calculate log returns
    log_returns = np.diff(np.log(prices))
    
    # Safety check
    if len(log_returns) == 0:
        raise ValueError("Not enough price data to calculate log returns")
    
    # Try McCulloch's fast method first
    try:
        from scipy.stats._levy_stable import _fitstart
        alpha, beta, loc, scale = _fitstart(log_returns)
    except (ImportError, AttributeError):
        # Fallback to MLE with limited iterations
        try:
            alpha, beta, loc, scale = levy_stable.fit(log_returns, method='MLE', optimizer={'maxiter': 20})
        except Exception:
            # Ultimate fallback: use rough estimates
            if len(log_returns) > 0:
                alpha = 1.8
                beta = 0.0
                loc = np.median(log_returns) if len(log_returns) > 0 else 0.0
                scale = np.std(log_returns) * 0.5 if len(log_returns) > 0 else 0.01
            else:
                raise ValueError("Unable to fit distribution: insufficient data")
    
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
    import matplotlib.dates as mdates
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Convert index to list for plotting
    dates = prices.index.tolist()
    values = prices.values
    
    ax.plot(dates, values, linewidth=2, color='#2E86AB', 
            label='Historical Prices', zorder=3)
    
    ax.scatter([dates[-1]], [last_price], color='#1e3a8a', s=150, 
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
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig


def plot_distribution_fit(log_returns, alpha, beta, loc, scale):
    """Plot histogram with fitted distribution"""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Safety check
    if len(log_returns) == 0:
        ax.text(0.5, 0.5, 'No log returns data available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        return fig
    
    # Plot histogram with robust bins
    n_bins = min(50, max(10, len(log_returns) // 10))
    ax.hist(log_returns, bins=n_bins, density=True, alpha=0.6, 
            color='#3b82f6', edgecolor='#1e3a8a', label='Historical Log Returns')
    
    # Create x range for PDF plot using robust statistics
    try:
        x_min = float(np.percentile(log_returns, 1))
        x_max = float(np.percentile(log_returns, 99))
        
        # Ensure we have a valid range
        if x_min >= x_max:
            x_min = float(np.min(log_returns))
            x_max = float(np.max(log_returns))
        
        # Add some padding
        x_range = x_max - x_min
        x_min -= x_range * 0.1
        x_max += x_range * 0.1
        
        x = np.linspace(x_min, x_max, 1000)
        pdf = levy_stable.pdf(x, alpha, beta, loc=loc, scale=scale)
        ax.plot(x, pdf, color='#ef4444', linewidth=3, label='Fitted Stable Distribution')
    except Exception as e:
        # If PDF plotting fails, just show the histogram
        ax.text(0.5, 0.95, f'Note: Could not plot fitted distribution', 
                ha='center', va='top', transform=ax.transAxes, fontsize=10, color='red')
    
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
    
    # Safety check
    if len(log_returns) == 0:
        ax.text(0.5, 0.5, 'No log returns data available', 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    time_scale_factor = np.sqrt(days_ahead)
    scaled_scale = scale * time_scale_factor
    scaled_loc = loc * days_ahead
    
    scenario_prices = [price for _, price, _, _ in scenarios]
    scenario_log_returns = [np.log(price/last_price) for price in scenario_prices]
    
    # Use percentiles for more robust range
    lr_min = np.percentile(log_returns, 1)
    lr_max = np.percentile(log_returns, 99)
    
    min_log_return = min(max(-0.4, lr_min), min(scenario_log_returns) - 0.05)
    max_log_return = max(min(0.4, lr_max), max(scenario_log_returns) + 0.05)
    
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
    """Generate concise analysis summary with tables"""
    summary = f"""📊 **Analysis Summary for {symbol}**

**Current Price:** ${last_price:.2f}  
**Analysis Period:** {historical_days} days  
**Time Horizon:** {days_ahead} days ahead

---

**📈 Stable Distribution Parameters**

α (Alpha) = {alpha:.4f}  
β (Beta) = {beta:.4f}  
μ (Location) = {loc:.6f}  
σ (Scale) = {scale:.6f}

---

**🎯 Target Scenario Probabilities**
"""
    
    # Create DataFrame for scenarios table
    table_data = []
    for pct, price, prob_lower, prob_higher in scenarios:
        table_data.append({
            'Scenario': f'{pct:+.1f}%',
            'Target Price': f'${price:.2f}',
            'Prob. Below': f'{prob_lower*100:.2f}%',
            'Prob. Above': f'{prob_higher*100:.2f}%'
        })
    
    scenarios_df = pd.DataFrame(table_data)
    
    return summary, scenarios_df


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
Nippofin Business Unit

Inspired by Mandelbrot, Fama, and Nolan's work on heavy-tailed distributions.
""")

#############################################
# MAIN ANALYSIS
#############################################

if analyze_button:
    try:
        with st.spinner(f'Downloading data for {symbol}...'):
            # Calculate date range
            # Convert date object to string safely
            if hasattr(end_date, 'strftime'):
                end_date_str = end_date.strftime('%Y-%m-%d')
            else:
                end_date_str = str(end_date)
            
            # Calculate start date (1 year before)
            end_dt = pd.to_datetime(end_date_str)
            start_dt = end_dt - pd.DateOffset(years=1)
            start_date = start_dt.strftime('%Y-%m-%d')
            
            # Download data with error handling
            try:
                data = yf.download(symbol, start=start_date, end=end_date_str, progress=False)
            except Exception as e:
                st.error(f"❌ Error downloading data: {str(e)}")
                st.stop()
            
            if data is None or data.empty:
                st.error(f"❌ No data found for symbol '{symbol}'. Please check the symbol and try again.")
                st.stop()
            
            # Handle multi-index columns (new yfinance format)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            
            # Handle both old and new Yahoo Finance column formats
            if 'Adj Close' in data.columns:
                prices = data['Adj Close'].dropna()
            elif 'Close' in data.columns:
                prices = data['Close'].dropna()
            else:
                st.error(f"❌ Unable to find price data for '{symbol}'. Columns available: {list(data.columns)}")
                st.stop()
            
            # Check if we have enough data
            if len(prices) < 30:
                st.error(f"❌ Not enough data for '{symbol}'. Only {len(prices)} days found. Need at least 30 days.")
                st.stop()
            
            # Ensure index is datetime and properly formatted
            if not isinstance(prices.index, pd.DatetimeIndex):
                prices.index = pd.to_datetime(prices.index)
            
            historical_days = len(prices)
            last_price = float(prices.iloc[-1])
            
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
        
        # Summary text and table
        st.markdown("---")
        st.subheader("📊 Analysis Summary")
        summary, scenarios_df = create_summary_text(symbol, last_price, alpha, beta, loc, scale, 
                                     scenarios, days_ahead, historical_days)
        st.markdown(summary)
        st.table(scenarios_df)
        
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
    <p><strong>Nippotica Corporation</strong> | Nippofin Business Unit</p>
    <p>Stock Price Hit Probability Tool | Stable Distribution Analysis</p>
    <p style='font-size: 0.9em;'>For educational and research purposes only. Not investment advice.</p>
</div>
""", unsafe_allow_html=True)
