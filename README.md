# Projects 🤖

### 1. Electricity Price Spread Forecasting and Trading Strategies ⚡
This is the code of a 10 ECTS Topics project I've done as part of my masters in Economics. I forecast day-ahead electricity price spreads across three borders using persistence and seasonal-naive baselines, Elastic Net, ARIMA, LightGBM (point + quantile q10/q50/q90) and an LSTM. I then evaluate their performance against each other and run a trading strategy on each model to compare their profitability under execution costs and stress tests.

<details>
<summary><b>View Details and Results </b></summary>

#### Objective
Forecasts the cross-border spread $S_t = P_{DK1,t} - P_{B,t}$ between DK1 and three neighbouring bidding zones (DE_LU, DK2, SE3), converts the forecasts into trading signals, and backtests them under progressively harsher execution-cost and position-size assumptions. The question is whether a backtested edge survives realistic costs and the fact that daily P&L is serially correlated.

---
#### 📊 Key Findings
| Border | Robust edge under stress? | HAC-90d Sharpe (LightGBM Point) | Note |
| :--- | :--- | :--- | :--- |
| **DK1–DE_LU** | **Yes** | 8.46 | Profitable under all three stress scenarios (moderate / conservative / severe); net P&L falls 66% from moderate to severe but stays positive |
| **DK1–DK2** | **No** | 1.71 | Negative under conservative stress, and no strategy is profitable under severe stress. Reported as a negative result |
| **DK1–SE3** | **Real, but overstated at short lags** | 7.00 | The 7-day HAC Sharpe is 13.03, but the effective number of independent trading days falls to about 74 of 815 at the 90-day lag, so the 90-day figure is the headline |

* **Most of the Sharpe is the spread, not the model.** The zero-parameter persistence strategy already has a HAC-90d Sharpe of 7.47 at DK1–DE_LU and 5.69 at DK1–SE3. The models mainly add traded volume.
* **Lower forecast error does not mean higher profit.** ARIMA beats persistence on RMSE at every border but has a lower Sharpe than persistence at all three.
* **The borders differ because congestion differs.** The spread is zero in about 60% of hours at DK1–DE_LU and DK1–DK2 but in only 15% at DK1–SE3.

---
#### Data
* **ENTSO-E Transparency Platform:** hourly day-ahead prices, load forecasts, wind/solar forecasts and day-ahead net transfer capacity (NTC) for DK1, DK2, DE_LU and SE3.
* **Yahoo Finance:** TTF gas futures and an EU ETS carbon proxy (KRBN).
* **Period:** 2022-01-01 to 2025-09-30, ending before the October 2025 move to 15-minute market time units. The out-of-sample test period is July 2023 to September 2025 (19,180 hours).

#### Feature Engineering
Lagged spreads (t-24/48/168h), wind-forecast differentials, residual demand per zone, calendar and holiday dummies, and wind×NTC / demand×NTC interaction terms.

#### Models
| Model | Role |
| :--- | :--- |
| Persistence (t-24) / Seasonal Naive (t-168) | Naive baselines |
| Elastic Net | Linear benchmark |
| ARIMA (seasonal ARMA on the spread) | Classical time-series benchmark |
| LightGBM (point forecast) | Main non-linear model |
| LightGBM (quantile q10/q50/q90) | Probabilistic forecast; drives the confidence-weighted strategy and the calibration check |
| LSTM | Sequence model with a 168-hour lookback window |

#### Validation & Statistical Tests
* **Walk-forward validation:** five expanding-window folds with a 168-hour embargo, applied to every model's inner validation split as well as the outer folds.
* **Diebold–Mariano tests:** every model against persistence, with HAC/Newey–West standard errors.
* **Quantile calibration:** empirical coverage of the [q10, q90] interval is measured directly. The worst-calibrated border (DK1–DK2) is also the one where the confidence-weighted strategy loses money.
* **Serial dependence:** HAC Sharpe ratios at six lags (1–90 days) and the Newey–West-implied effective number of independent trading days (Lo, 2002).

#### Backtest
* **Position sizing:** volatility targeting on realized spread volatility, with a ceiling of 100 MWh per hour at every border. The ceilings are stylized and no NTC-based cap is applied.
* **Cost waterfall:** gross P&L → BRP imbalance haircut → exchange fees (Nord Pool's published fee) → volatility-scaled slippage. The haircut is a cost on the absolute spread captured, $q_t S_t - h\,|q_t S_t|$, so it never reduces a loss.
* **Stress tests:** three scenarios with a higher haircut, higher fees and slippage, and lower position ceilings (75 / 50 / 25 MWh).
* **Risk measures:** HAC Sharpe, CVaR95, Profit/CVaR, maximum drawdown and net P&L per MWh traded.

#### Robustness Check on Data Timing
A query against the ENTSO-E API showed that its `createdDateTime` field cannot verify when a forecast was published. Elastic Net and LightGBM Point were therefore re-estimated with every forecast-derived feature shifted a further 24 hours back, and the results did not change materially.

#### Notebook Structure
```
├── Data Pipeline             # ENTSO-E + Yahoo Finance ingestion and cleaning
├── Feature Engineering       # Fundamentals, NTC interactions, calendar, lags
├── Predictive Modeling       # Walk-forward validation of all models, calibration
├── ARIMA Benchmark           # Seasonal ARMA on the same folds
├── Trading Backtest          # Signals, position sizing, cost waterfall
├── Stress Tests              # Moderate / conservative / severe scenarios
├── Serial-Dependence Checks  # Multi-lag HAC Sharpe, effective independent days
├── SHAP                      # Descriptive feature importance per border
└── Lag-Robustness Check      # Forecast features shifted a further 24 hours
```

#### Setup
```bash
pip install entsoe-py pandas numpy scikit-learn lightgbm torch yfinance matplotlib seaborn holidays shap tqdm statsmodels python-dotenv beautifulsoup4 lxml pyarrow
```
Requires a free ENTSO-E API token (register at [transparency.entsoe.eu](https://transparency.entsoe.eu)):
```bash
export ENTSOE_API_KEY="your-token"
```

#### Limitations
The backtest is a stylized execution framework and not a claim that the position sizes were executable in the real auction. No day-ahead NTC series is published for DK1–SE3. The sample includes the 2022 gas crisis, and its influence on the fitted models has not been tested separately.

</details>

### 2. Whale Activity Listener 🐳
This is a bot that monitors activity on Hyperliquid, a decentralized perpetual futures exchange for cryptocurrency. It is inspired by the enormous short position placed on the exchange minutes before Trump's tariff-reveal that resulted in a +$160 million realized profit within 24 hours after the news due to the immediate stock and crypto market crash. The Whale Listener will notify you on Telegram when unusually large short positions are being placed.
<details>
<summary><b>View Technical Details & Infrastructure Architecture</b></summary>
  
#### Objective
Captures order-flow toxicity, institutional positioning, and anomalous block trades on Hyperliquid perpetual swap order books via low-latency streaming.

#### Core Engineering & Resiliency
* **Asynchronous Stream Ingestion:** Developed an asynchronous ingestion pipeline using `asyncio` and `websockets` to handle continuous, low-latency JSON trade ticks concurrently without blocking the main execution thread.
* **Fault-Tolerant Network State:** Implemented exponential backoff mechanics using the `tenacity` framework, actively maintaining WebSocket connection health with customized heartbeats (`ping_interval=15`, `ping_timeout=30`).
* **Hybrid Dynamic Failover:** Engineered an automated fallback module. If the WebSocket layer encounters persistent network disruption and exceeds maximum reconnection parameters (`max_ws_attempts=5`), the pipeline automatically shifts to historical HTTP REST polling to prevent telemetry blind spots.
* **Idempotency & Deduplication:** Tracks and validates unique cryptographic transaction identifiers (`trade_hash`) across an in-memory `set()` state and local append-only storage (`.csv`) to enforce structural data deduplication during multi-channel ingestion.
</details>

### 3. Machine Learning for Crypto Volatility Forecasting 🦾
This is a code i wrote while at Humboldt-Universität zu Berlin. I was interested in trying out different machine learning models to forecast the volatility of BTC and ETH and see which model performed the best. XGBoost slightly outperformed LSTM and Random Forest. Predicting volatility is interesting since it can be used for risk management, hedging strategies and open up for options arbitrage opportunities if our models can predict volatility better than the market (implied volatility). 
<details>
<summary><b>View Model Architecture & Mathematical Features</b></summary>
  
#### Objective
An empirical evaluation of non-linear machine learning architectures against traditional statistical benchmarks for predicting one-step-ahead hourly realized volatility ($RV$) on Bitcoin (BTC-USD).
---

#### 📊 Performance Metrics & Comparison
Models were trained on 14 months of hourly data and evaluated out-of-sample on a 4-month test window.
| Model | Architecture Type | RMSE | MAE | $R^2$ |
| :--- | :--- | :--- | :--- | :--- |
| **Naive Baseline** | Persistence Benchmark ($\hat{\sigma}_{t+1} = \sigma_t$) | 0.0314 | 0.0148 | 0.9216 |
| **Random Forest** | Ensemble Tree (100 estimators, max depth 10) | 0.0277 | 0.0154 | 0.9390 |
| **LSTM** | Recurrent Neural Network (50 hidden units, Dropout 0.2) | 0.0280 | 0.0177 | 0.9374 |
| **XGBoost** | **Gradient Boosting (Learning Rate 0.05, early stopping)** | **0.0258** | **0.0147** | **0.9469** |
---

#### 🛠 Feature Engineering & Mathematical Framework
To avoid look-ahead bias, all features were engineered strictly at time $t$ using vectorized matrix operations. Scalers were fit exclusively on training distributions.
* **Target Variable:** One-step-ahead realized volatility, defined as the rolling annualized standard deviation of log returns over a 30-hour window ($n=30$):
  $$\sigma_t^{RV} = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} r_{t-i}^2} \cdot \sqrt{24 \times 365}$$
* **Garman-Klass Volatility Feature:** Implemented an intra-bar, range-based estimator to capture structural intraday noise omitted by traditional close-to-close calculations:
  $$\sigma_{GK}^2 = 0.5 \cdot \left[\ln\frac{H}{L}\right]^2 - (2\ln 2 - 1) \cdot \left[\ln\frac{C}{O}\right]^2$$
* **Lag Matrix:** Modeled 5-period lags of log returns, squared returns, and rolling realized volatility, alongside 5-hour and 10-hour moving averages.
* **LSTM Input Transformation:** Formatted the 2D tabular feature matrix into scaled 3D temporal tensors `[samples, time-steps, features]` using a 24-hour lookback window.
---

#### 🧠 Key Findings & Empirical Insights
* **XGBoost Superiority:** XGBoost delivered the strongest out-of-sample predictive power, representing an **~18% RMSE improvement over the naive baseline**. Its regularization constraints effectively distributed feature importance across the lag matrix rather than over-relying on $\sigma_t$.
* **The Volatility Floor:** The naive persistence model achieved an $R^2$ of 0.92, confirming strong volatility clustering (ARCH effects) in high-frequency crypto regimes. All machine learning optimization occurred within the narrow margin above this floor.
* **LSTM Tail Underperformance:** Despite its architectural complexity, the LSTM model underperformed on MAE. Training under standard Mean Squared Error (MSE) loss caused the network to make risk-averse predictions near the conditional mean, systematically undershooting volatility spikes.
* **Feature Signality:** MDI feature importance charts confirmed that `vol_lag_1` dominates model decisions, matching classic GARCH(1,1) behavior. The Garman-Klass estimator provided incremental predictive signal at high frequencies.
</details>

### 4. VIX Systematic Signals for S&P 500 🚨
This bot sends you a Telegram message when certain VIX (market fear) index criteria have been met, that historically (backtested) outperforms DCA'ing. It is an oversimplified trading strategy on its own, but extending it and pairing it with other sttategies and models can provide useful signals within a broader execution framework.
<details>
<summary><b>View Strategy Logic & State Management</b></summary>
  
#### Objective
A programmatic systematic mean-reversion execution system designed to capture equity risk premiums during localized macroeconomic market panics.

#### System Architecture
* **Persistent State Management:** Utilizes a custom JSON state serialization engine (`vix_strategy_state.json`) to manage trade lifecycle data across system restarts, ensuring the execution engine strictly honors temporal parameters (e.g., a hardcoded `75-day` cooldown matrix).
* **Vectorized Matrix Slicing:** Employs vectorized rolling window analysis via `pandas` to isolate volatility distributions, calculating 21-day moving averages and rolling minima bounds concurrently to avoid unoptimized row-wise iterative loops.
</details>

### 5. Funding Rate Arbitrage 🏦
This bot tracks the funding rates of 134 different digital assets (tokens) on the GMX perpetual futures exchange and gives daily summaries of top-performing tokens (that has met certain volume criteria) in order to gather data and see which tokens are best suited for this strategy. The idea is then to open a short position of x token with x as colleteral creating a delta-neutral position that accumulates profit on the funding rate alone. Telegram notifications will update you on funding rates turning negative for longer periods of time (and back to positive) as well as the daily summary.
<details>
<summary><b>View Data Pipeline & Arbitrage Mechanics</b></summary>
  
#### Objective
Identifies structural capital inefficiencies across 130+ perpetual asset markets on GMX to generate automated, market-neutral yields.

#### Data Pipeline Architecture
* **Fault-Tolerant ETL:** Features an API engine that dynamically rotates through multiple decentralized fallback endpoints (`gmxinfra.io`) to seamlessly handle network timeouts, HTTP 429 rate limits, and node latency.
* **Delta-Neutral Processing:** Tracks Open Interest (OI) imbalances between long and short sides. The pipeline automatically flags fields where opening short positions yields high continuous funding tracking rates, while calculating spot asset collateral requirements to completely nullify directional price risk.
</details>
---

## 🛠 Portfolio Overview
| Project | Core Tech Stack | Architectural Focus |
| :--- | :--- | :--- |
| **Whale Activity Listener** | Asyncio, WebSockets, Tenacity, Pandas, Python-Telegram-Bot | Event-Driven Streaming, Hybrid Failover ETL, Fault Tolerance |
| **Crypto Volatility Forecasting** | TensorFlow, XGBoost, Scikit-Learn, NumPy, YFinance | Non-Linear Time-Series ML, Multi-Lag Feature Engineering |
| **VIX Systematic Signals** | Pandas, YFinance, Requests, State Serialization (JSON) | Persistent State Management, Rolling Matrix Math, Automation |
| **Funding Rate Arbitrage Bot** | REST APIs, Vectorized Math, Multi-Endpoint Rotators | Delta-Neutral Yield Optimization, Network Redundancy |
| **Electricity Price Spread Forecasting** | ENTSO-E API, LightGBM, PyTorch (LSTM), Scikit-Learn, Statsmodels (ARIMA), SHAP, YFinance | Walk-Forward Time-Series CV, Cross-Border Spread Modeling, Execution-Cost-Aware Backtesting |
---

## 🚀 Installation & Setup
1. **Clone the repository**:
   ```bash
   git clone [https://github.com/oliverhoyrup/crypto-trading-bots.git](https://github.com/oliverhoyrup/crypto-trading-bots.git)
   ```
   
2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3. **Configure Environment Variables**:
   Create a .env file in the root directory and add your credentials:
    ```bash
   TELEGRAM_BOT_TOKEN=your_bot_token_here
   TELEGRAM_CHAT_ID=your_chat_id_here
    ```
