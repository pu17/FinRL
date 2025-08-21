# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview
FinRL is an open-source framework for financial reinforcement learning. It provides a complete ecosystem for automated trading with deep reinforcement learning algorithms, following a train-test-trade pipeline.

## Core Architecture
FinRL follows a three-layer architecture:
- **Applications Layer**: Financial tasks (stock trading, portfolio allocation, cryptocurrency trading)
- **Agents Layer**: DRL algorithms (ElegantRL, Stable Baselines3, RLlib)  
- **Environment Layer**: Market environments and data processing

The main workflow uses three core files:
- `finrl/train.py`: Training DRL models
- `finrl/test.py`: Backtesting trained models
- `finrl/trade.py`: Paper trading and live trading

## Development Commands

### Installation and Setup
```bash
pip install -r requirements.txt
# or for development with Poetry:
poetry install
```

### Running the Framework
```bash
# Main entry point with different modes
python finrl/main.py --mode=train    # Train a model
python finrl/main.py --mode=test     # Test/backtest a model  
python finrl/main.py --mode=trade    # Paper trading

# Alternative approach using individual modules
python -m finrl.train
python -m finrl.test
python -m finrl.trade
```

### Testing
```bash
pytest unit_tests/                   # Run unit tests
pytest unit_tests/environments/      # Test environments only
pytest unit_tests/downloaders/       # Test data downloaders only
```

### Code Quality
```bash
pre-commit run --all-files          # Run all pre-commit hooks
black finrl/                        # Format code with Black
flake8 finrl/                       # Lint with flake8
```

### Docker
```bash
bash docker/bin/build_container.sh  # Build Docker container
bash docker/bin/test.sh             # Run tests in Docker
```

## Key Components

### Data Processing (`finrl/meta/data_processor.py`)
- Handles multiple data sources (Yahoo Finance, Alpaca, Binance, etc.)
- Processes OHLCV data and technical indicators
- Converts data to arrays for RL training

### Environments (`finrl/meta/env_*/`)
- `env_stock_trading/`: Stock trading environments
- `env_portfolio_allocation/`: Portfolio allocation environments  
- `env_cryptocurrency_trading/`: Crypto trading environments
- `env_portfolio_optimization/`: Portfolio optimization environments

### Agents (`finrl/agents/`)
- `elegantrl/`: ElegantRL-based agents
- `stablebaselines3/`: Stable Baselines3 integration
- `rllib/`: Ray RLlib integration
- `portfolio_optimization/`: Portfolio optimization algorithms

### Configuration
- `finrl/config.py`: Global configuration (dates, parameters, API settings)
- `finrl/config_tickers.py`: Stock ticker definitions
- `finrl/config_private.py`: Private API keys (not in git)

## Important Notes

### Data Sources
FinRL supports 15+ data sources including Yahoo Finance (default), Alpaca, Binance, and others. Each has different rate limits and data availability.

### Model Parameters
Default hyperparameters are defined in `config.py` for different algorithms:
- A2C_PARAMS, PPO_PARAMS, DDPG_PARAMS, TD3_PARAMS, SAC_PARAMS
- ERL_PARAMS for ElegantRL
- RLlib_PARAMS for Ray RLlib

### Time Zones
Multiple time zones are supported for different markets (US/Eastern, Asia/Shanghai, Europe/Paris, etc.)

### API Configuration
For live trading, configure API keys in `config_private.py`:
```python
ALPACA_API_KEY = "your_key"
ALPACA_API_SECRET = "your_secret" 
```

### Technical Indicators
Default indicators include: MACD, Bollinger Bands, RSI, CCI, DX, SMA. Additional indicators can be added via the stockstats library.

## Development Workflow
1. Configure data source and parameters in config files
2. Train models using the train pipeline
3. Backtest with historical data using test pipeline
4. Deploy for paper trading using trade pipeline
5. Monitor performance and iterate

## Common Patterns
- Use DataProcessor for all data operations
- Follow the env_config pattern for environment initialization
- Store models in `trained_models/` directory
- Use `if_vix=True` to include VIX volatility index
- Configure different DRL libraries via the `drl_lib` parameter