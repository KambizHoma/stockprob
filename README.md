# Stock Price Hit Probability - Streamlit Edition

**Nippotica Corporation | Algotechniq Business Unit**

## Overview

Clean, professional Streamlit implementation of the Stock Hit Probability calculator using stable distributions (Lévy α-stable) to calculate the odds a stock reaches target prices.

## Features

### Core Capabilities
- 📈 **Historical Data Analysis**: Downloads price data from Yahoo Finance
- 🔬 **Automatic Distribution Fitting**: Fast stable distribution fitting using McCulloch's method
- 🎯 **Probability Calculations**: Computes hit probabilities for custom price scenarios
- 📊 **Visual Analysis**: Professional charts with Nippotica branding

### UI Design
- **Clean Sidebar**: All controls organized in left panel like professional finance apps
- **Progressive Disclosure**: Optional displays (distribution fit, CDF, statistics)
- **Smart Defaults**: SPY, +3%, +5%, -3%, 15-day horizon
- **Nippotica Branding**: Corporate blue gradient header and color scheme

## What are Stable Distributions?

Stable distributions capture the **fat tails** and **extreme events** that traditional normal distributions miss.

**Four Parameters:**
- **α (alpha)**: Tail heaviness (0 < α ≤ 2)
  - α = 2: Normal distribution
  - α < 2: Fat tails
  - Lower α = more extreme events

- **β (beta)**: Skewness (-1 ≤ β ≤ 1)
  - β = 0: Symmetric
  - β ≠ 0: Asymmetric

- **μ (mu)**: Location parameter
- **σ (sigma)**: Scale parameter (volatility)

## Installation

### Quick Start
```bash
pip install -r requirements_streamlit.txt
streamlit run stockprob_streamlit.py
```

### Step by Step
```bash
# Clone or download files
cd your-project-folder

# Install dependencies
pip install streamlit numpy pandas matplotlib scipy yfinance

# Run the app
streamlit run stockprob_streamlit.py
```

## How to Use

1. **Enter Stock Symbol** in sidebar (e.g., SPY, AAPL, BTC-USD)
2. **Set Analysis End Date** (app uses 1 year of data before this date)
3. **Adjust Price Targets** using the three scenario sliders
4. **Toggle Advanced Options** to change time horizon (default: 15 days)
5. **Click "Calculate Probabilities"** to run analysis
6. **View Results**:
   - Price chart with target scenarios
   - Distribution fit (optional)
   - CDF analysis (optional)
   - Detailed probability summary
   - Statistics table (optional)

## UI Layout

```
┌─────────────────────────────────────────┐
│  Sidebar                                │
├─────────────────────────────────────────┤
│  🎯 Stock Analysis Controls             │
│  • Stock Symbol                         │
│  • Analysis End Date                    │
│                                         │
│  🎯 Price Target Scenarios              │
│  • Target 1: Upside                     │
│  • Target 2: Stretch Goal               │
│  • Target 3: Downside Risk              │
│                                         │
│  ⚙️ Advanced Options (expandable)       │
│  • Time Horizon (Days)                  │
│                                         │
│  📊 Display Options                     │
│  ☑ Show Distribution Fit                │
│  ☑ Show CDF Analysis                    │
│  ☐ Show Detailed Statistics             │
│                                         │
│  ℹ️ About This Tool (expandable)        │
│                                         │
│  [🎯 Calculate Probabilities]           │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│  Main Area                              │
├─────────────────────────────────────────┤
│  🎯 Stock Price Hit Probability         │
│  Nippotica Corporation Header           │
│                                         │
│  📈 Price History Chart                 │
│                                         │
│  📊 Distribution Fit (if enabled)       │
│                                         │
│  📉 CDF Analysis (if enabled)           │
│                                         │
│  📊 Detailed Analysis Summary           │
│                                         │
│  📈 Statistics Table (if enabled)       │
└─────────────────────────────────────────┘
```

## Example Use Cases

### Conservative Analysis
```
Symbol: SPY
End Date: 2024-12-31
Targets: +2%, +3%, -2%
Days Ahead: 10
```

### Volatile Asset
```
Symbol: BTC-USD
End Date: 2024-12-31
Targets: +10%, +20%, -10%
Days Ahead: 7
```

### Crisis Period Analysis
```
Symbol: ^GSPC
End Date: 2009-12-31
Targets: +5%, +10%, -15%
Days Ahead: 15
```

## Technical Details

### Algorithm
1. Downloads 1 year of historical data ending on specified date
2. Calculates log returns: ln(P_t / P_{t-1})
3. Fits stable distribution using McCulloch's quantile method
4. Scales parameters for time horizon
5. Calculates CDF probabilities for each scenario

### Color Scheme (Nippotica)
- Header gradient: `#1e3a8a` → `#3b82f6`
- Primary: `#1e3a8a` (deep blue)
- Secondary: `#3b82f6` (bright blue)
- Success: `#10b981` (green)
- Warning: `#f59e0b` (orange)
- Danger: `#ef4444` (red)

## Advantages Over Gradio Version

✅ **Cleaner UI**: Sidebar-based controls like professional finance apps
✅ **Better Organization**: Progressive disclosure of advanced features
✅ **Streamlit Native**: No PIL image conversion needed
✅ **Responsive**: Better mobile/tablet support
✅ **Simpler Deployment**: Standard Streamlit hosting options

## Important Disclaimers

⚠️ **For Educational Purposes Only**

- This tool is designed for learning and research
- Past performance does not guarantee future results
- Markets can change regime (parameters are not constant)
- Do not use as sole basis for investment decisions
- Consult financial professionals for investment advice

## References

### Academic Papers
- Mandelbrot, B. (1963). "The Variation of Certain Speculative Prices"
- Fama, E.F. (1965). "The Behavior of Stock-Market Prices"
- Nolan, J.P. (2020). "Univariate Stable Distributions: Models for Heavy Tailed Data"

### Technical Documentation
- SciPy: [`scipy.stats.levy_stable`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levy_stable.html)
- Streamlit: [Documentation](https://docs.streamlit.io/)

## About

**Created for**: Nippotica Corporation - Algotechniq Business Unit

**Purpose**: Educational demonstration of stable distribution applications in quantitative finance

**Modern Implementation**: Clean Streamlit UI with:
- Professional sidebar-based controls
- Nippotica corporate branding
- Progressive disclosure of features
- Optional advanced settings
- Real-time market data integration

## License

MIT License

## Contact

For questions about stable distributions in financial applications, consult:
- Academic literature on heavy-tailed distributions
- Quantitative finance textbooks
- Financial risk management professionals

---

**Nippotica Corporation | Algotechniq Business Unit**
*Fat-Tailed Distribution Analysis for Real Markets*
