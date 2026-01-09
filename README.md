# Volatility-Targeted-Trading-Strategy
A production-ready Python implementation of an adaptive volatility-targeted trading strategy using ARIMA-GARCH modeling for SPY (S&amp;P 500 ETF) and other securities.

## Overview
This system combines time series forecasting with volatility modeling to generate trading signals and dynamically size positions based on predicted market conditions. The strategy aims to maximize risk-adjusted returns while protecting capital during periods of high volatility.

*Adaptive Forecasting*: Auto-optimized ARIMA models for return prediction
*Volatility Modeling*: GARCH models with Student's t-distribution for realistic tail risk
*Walk-Forward Testing*: Realistic backtesting with periodic model refitting
*Risk Management*: Multi-layered filtering system including conviction thresholds and volatility limits
*Position Sizing*: Dynamic volatility-targeted position sizing
*Hyperparameter Optimization*: Automated threshold tuning for Sharpe ratio maximization
*Live Data Integration*: Fetches real-time market data via Yahoo Finance

## Architecture
1. Data Pipeline (fetch_yahoo_data)

Downloads historical price data from Yahoo Finance
Calculates log returns (scaled by 100 for numerical stability)
Handles multi-index dataframes and data validation

2. Volatility Pipeline (VolatilityPipeline)

ARIMA Component: Auto-optimizes order selection for mean prediction
GARCH Component: Grid search for optimal volatility model (Student's t-distribution)
Backtesting Engine: Walk-forward validation with configurable refitting intervals

3. Strategy Engine (StrategyBacktester)

Signal Generation: Directional signals based on predicted returns
Conviction Filtering: Minimum threshold to avoid low-confidence trades
Volatility Filtering: Automatic exit during extreme market conditions
Position Sizing: Volatility-targeted with configurable leverage limits
Transaction Costs: Realistic cost modeling in basis points

# An Individual Thought
I had fun diving deep into automated trading strategies and can immediately visualize how simple application of existing algorithms applied to an application frontend utilizing public APIs from major trend companies can be cash cows in a rapidly changing world. 

For future projects, applying conviction and volatility filters will be necessary. Teams will need to take a good look at hyperparameters like target daily volatility, minimum predicted returns, transaction costs, days between model refits.

This would be a good first step for creating a SAAS application for portfolio management.
