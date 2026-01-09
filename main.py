import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm
from arch import arch_model
import yfinance as yf 

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# ==========================================
# 1. LIVE DATA UTILS
# ==========================================
def fetch_yahoo_data(ticker='SPY', start_date='2020-01-01', end_date=None):
    """
    Fetches daily data from Yahoo Finance and calculates Log Returns.
    """
    print(f"Downloading data for {ticker} from {start_date}...")
    
    # Download data
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if df.empty:
        raise ValueError(f"No data found for {ticker}. Check spelling or date range.")

    # Use Adjusted Close to account for dividends/splits. Adjusted Close numbers accounts for any missed actions that were not accounted for by trade close and allows back-calculations to avoid event volatility.
    
    # Handle multi-index columns if yfinance returns them. This allows for handling datasets stacked by Attribute first and Ticker second.
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df['Adj Close'].columns:
            prices = df['Adj Close'][ticker]
        else:
            prices = df['Adj Close'].iloc[:, 0]
            print(f"Warning: Ticker {ticker} not found in columns, using first available column")
    else:
        prices = df['Adj Close']

    # Calculate Log Returns: ln(Pt / Pt-1)
    # Multiply by 100 to convert to percentage (e.g., 0.01 -> 1.0). 
    # This helps GARCH optimizers converge significantly better.
    lrets = 100 * np.log(prices / prices.shift(1))
    
    # Clean data
    lrets.dropna(inplace=True)
    lrets.name = f"{ticker}_Log_Returns"
    
    print(f" -> Loaded {len(lrets)} trading days.")
    return lrets

# ==========================================
# 2. MODELING PIPELINE
# ==========================================
class VolatilityPipeline:
    def __init__(self, model_name="production_model"):
        self.model_name = model_name
        self.arima_model = None
        self.garch_model = None
        self.garch_res = None
        self.best_garch_order = None 
        self.is_fitted = False
        
    def fit(self, ts: pd.Series, max_p=3, max_q=3, verbose=True):
        if verbose: 
            print(f"[{self.model_name}] Optimizing ARIMA...")
        
        # Stepwise Auto-ARIMA
        self.arima_model = pm.auto_arima(
            ts, start_p=1, start_q=1, max_p=5, max_q=5,
            d=None, seasonal=False, stepwise=True,
            suppress_warnings=True, error_action='ignore'
        )
        
        residuals = self.arima_model.resid()
        
        if verbose: 
            print(f"[{self.model_name}] Optimizing GARCH on residuals...")
        
        best_aic = np.inf
        
        for p in range(1, max_p + 1):
            for q in range(1, max_q + 1):
                try:
                    tmp_mdl = arch_model(residuals, vol='Garch', p=p, o=0, q=q, dist='StudentsT')
                    tmp_res = tmp_mdl.fit(disp='off')
                    if tmp_res.aic < best_aic:
                        best_aic = tmp_res.aic
                        self.best_garch_order = (p, q)
                        self.garch_model = tmp_mdl
                        self.garch_res = tmp_res
                except:
                    continue
        
        # Validate that GARCH model was successfully fitted
        if self.best_garch_order is None:
            raise RuntimeError("Failed to fit any GARCH model. Check your data quality.")
        
        self.is_fitted = True
        if verbose:
            print(f"[{self.model_name}] Best GARCH order: {self.best_garch_order}")
        return self

    # Walk-through Validation Backtest function to allow historical data to pass in
    def backtest(self, ts: pd.Series, start_index: int, refit_interval=0):
        """
        Performs walk-forward validation backtesting.
        
        Args:
            ts: Time series data
            start_index: Index to start backtesting from
            refit_interval: How often to refit models (0 = never refit, just update)
        """
        # Validate inputs
        if start_index >= len(ts):
            raise ValueError(f"start_index ({start_index}) must be less than data length ({len(ts)})")
        if start_index < 50:
            raise ValueError(f"start_index ({start_index}) too small, need at least 50 observations for training")
        
        print(f"[{self.model_name}] Starting Backtest on {len(ts) - start_index} samples...")
        
        predictions = []
        actuals = []
        volatility_preds = []
        
        # Initial training
        train_data = ts.iloc[:start_index]
        self.fit(train_data, verbose=False)
        
        for i in range(start_index, len(ts)):
            # Forecast
            curr_arima_pred = self.arima_model.predict(n_periods=1)[0]
            curr_garch_pred = self.garch_res.forecast(horizon=1, reindex=False)
            curr_vol_pred = np.sqrt(curr_garch_pred.variance.iloc[-1].values[0])
            
            predictions.append(curr_arima_pred)
            volatility_preds.append(curr_vol_pred)
            actuals.append(ts.iloc[i])
            
            # Update State
            true_value = ts.iloc[i]
            self.arima_model.update(pd.Series([true_value]))
            
            # Only refit after the first step and at specified intervals
            should_refit = (refit_interval > 0) and ((i - start_index) > 0) and ((i - start_index) % refit_interval == 0)
            
            if should_refit:
                self.fit(ts.iloc[:i+1], verbose=False)
            else:
                # Update GARCH with new residuals
                new_residuals = self.arima_model.resid()
                p, q = self.best_garch_order
                am = arch_model(new_residuals, vol='Garch', p=p, o=0, q=q, dist='StudentsT')
                self.garch_res = am.fit(disp='off', update_freq=0)
            
            if (i - start_index) % 50 == 0 and (i - start_index) > 0:
                print(f"Step {i - start_index}/{len(ts)-start_index}", end='\r')

        print(f"\nBacktest Complete.")
        
        return pd.DataFrame({
            'Actual': actuals,
            'Pred_Mean': predictions,
            'Pred_Vol': volatility_preds
        }, index=ts.index[start_index:])

# ==========================================
# 3. STRATEGY ENGINE
# ==========================================
class StrategyBacktester:
    def __init__(self, initial_capital=10000, trans_cost_bps=10):
        self.initial_capital = initial_capital
        self.trans_cost = trans_cost_bps / 10000.0
        self.results = None

    def run_strategy(self, backtest_df, target_vol=1.0, conviction_threshold=0.05, max_vol_limit=None):
        """
        Runs trading strategy with volatility targeting.
        
        Note: inputs are now in PERCENTAGE terms because we scaled data by 100.
        - target_vol=1.0 means 1% daily volatility target
        - conviction_threshold=0.05 means 0.05% daily return minimum
        - max_vol_limit: if set, don't trade when predicted volatility exceeds this (e.g., 3.0 = 3%)
        """
        df = backtest_df.copy()
        
        # Base Signal
        df['Signal'] = np.sign(df['Pred_Mean'])
        
        # Filter A: Conviction. This is designed to improve the Sharpe Ratio by trading less often
        mask_low_conviction = df['Pred_Mean'].abs() < conviction_threshold
        df.loc[mask_low_conviction, 'Signal'] = 0
        
        # Filter B: Volatility Safety. This is designed to pause trading when market becomes unstable
        if max_vol_limit is not None:
            mask_too_volatile = df['Pred_Vol'] > max_vol_limit
            df.loc[mask_too_volatile, 'Signal'] = 0

        # Position Sizing with safety check for very small volatilities
        # Replace zero and very small values to avoid division issues
        safe_vol = df['Pred_Vol'].replace(0, np.nan)
        safe_vol = safe_vol.where(safe_vol > 0.01, np.nan)  # Treat <0.01% vol as too small
        
        vol_factor = target_vol / safe_vol
        df['Position_Size'] = vol_factor.clip(upper=2.0).fillna(0)
        
        # Weight & Returns describes the total value allocated to this asset at any given time.
        # We must divide Actual returns by 100 to get back to decimal for PnL calc
        decimal_returns = df['Actual'] / 100.0
        
        df['Weight'] = df['Signal'] * df['Position_Size']
        df['Strat_Ret_Gross'] = df['Weight'].shift(1) * decimal_returns
        df['Weight_Change'] = df['Weight'].diff().abs()
        df['Cost'] = df['Weight_Change'] * self.trans_cost
        df['Strat_Ret_Net'] = df['Strat_Ret_Gross'] - df['Cost']
        
        # Handle NaN values in first row and elsewhere
        df['Strat_Ret_Net'].fillna(0, inplace=True)
        df['Strat_Ret_Gross'].fillna(0, inplace=True)
        df['Cost'].fillna(0, inplace=True)
        
        df['Equity_Curve'] = self.initial_capital * (1 + df['Strat_Ret_Net']).cumprod()
        
        # Benchmark (Buy & Hold SPY - a popular benchmark)
        df['Benchmark_Curve'] = self.initial_capital * (1 + decimal_returns).cumprod()
        
        self.results = df
        return df

    def generate_metric_report(self):
        """Generate performance metrics for the strategy."""
        if self.results is None:
            print("Warning: No results available. Run strategy first.")
            return None
        
        df = self.results
        
        total_ret = (df['Equity_Curve'].iloc[-1] / self.initial_capital) - 1
        bench_ret = (df['Benchmark_Curve'].iloc[-1] / self.initial_capital) - 1
        
        daily_mean = df['Strat_Ret_Net'].mean()
        daily_std = df['Strat_Ret_Net'].std()
        sharpe = np.sqrt(252) * (daily_mean / daily_std) if daily_std != 0 else 0
        
        # Max Drawdown
        rolling_max = df['Equity_Curve'].cummax()
        dd = (df['Equity_Curve'] - rolling_max) / rolling_max
        max_dd = dd.min()

        trades = df[df['Weight_Change'] > 0]
        
        metrics = {
            "Total Days": len(df),
            "Strategy Return": f"{total_ret*100:.2f}%",
            "Benchmark Return": f"{bench_ret*100:.2f}%",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_dd*100:.2f}%",
            "Trades Executed": len(trades)
        }
        return pd.DataFrame(metrics, index=[0])

    def optimize_hyperparameters(self, backtest_df, threshold_range):
        """Find optimal conviction threshold by testing multiple values."""
        best_sharpe = -np.inf
        best_thresh = 0
        stats = []
        print("\n--- Tuning Hyperparameters ---")
        
        for thresh in threshold_range: 
            res = self.run_strategy(backtest_df, conviction_threshold=thresh)
            daily_mean = res['Strat_Ret_Net'].mean()
            daily_std = res['Strat_Ret_Net'].std()
            sharpe = np.sqrt(252) * (daily_mean / daily_std) if daily_std != 0 else 0
            stats.append({'Threshold': thresh, 'Sharpe': sharpe})
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_thresh = thresh
        
        print(f"Optimization Complete. Best Threshold: {best_thresh:.5f} (Sharpe: {best_sharpe:.2f})")
        return best_thresh, pd.DataFrame(stats)

    def plot_equity(self):
        """Plot equity curve comparing strategy to benchmark."""
        if self.results is None:
            print("No results to plot. Run strategy first.")
            return
        
        plt.style.use('bmh')
        fig, ax = plt.subplots(figsize=(10, 6))
        df = self.results
        ax.plot(df.index, df['Benchmark_Curve'], label='Buy & Hold (SPY)', color='grey', alpha=0.5)
        ax.plot(df.index, df['Equity_Curve'], label='Active Strategy', color='blue', linewidth=2)
        ax.set_title("Strategy Equity Curve")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend()
        plt.tight_layout()
        plt.show()

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    # A. FETCH REAL DATA
    # We fetch SPY (S&P 500 ETF) for the last 4 years
    ticker = "SPY"
    TS = fetch_yahoo_data(ticker, start_date='2020-01-01')

    # B. Initialize Pipeline
    pipeline = VolatilityPipeline(f"{ticker}_Model")
    
    # C. Run Walk-Forward Backtest 
    # Testing on last 252 days (approx 1 trading year)
    # Refit every 63 days (approx once per quarter)
    results = pipeline.backtest(TS, start_index=len(TS)-252, refit_interval=63)
    
    # D. Strategy Setup
    trader = StrategyBacktester(initial_capital=100000, trans_cost_bps=5)
    
    # E. Optimize Thresholds (Range 0.0% to 0.15% daily return)
    # Note: data is scaled by 100, so 0.05 = 0.05% return
    thresholds = np.linspace(0, 0.15, 10)
    best_thresh, _ = trader.optimize_hyperparameters(results, thresholds)
    
    # F. Final Run
    final_res = trader.run_strategy(
        results, 
        target_vol=1.0,         # Target 1% daily volatility
        conviction_threshold=best_thresh, 
        max_vol_limit=3.0       # Exit if volatility > 3% (Crash protection)
    )
    
    # G. Report
    print(f"\n--- {ticker} STRATEGY REPORT ---")
    report = trader.generate_metric_report()
    if report is not None:
        print(report.T)
    trader.plot_equity()