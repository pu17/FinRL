# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FinRL is an open-source framework for financial reinforcement learning that provides a standardized environment for developing and testing RL algorithms in quantitative finance. It has a three-layer architecture: market environments, agents, and applications.

## Development Commands

### Testing
```bash
# Run unit tests locally
python3 -m unittest discover

# Run tests in Docker
./docker/bin/build_container.sh
./docker/bin/test.sh
```

### Documentation
```bash
# Build documentation (from docs/ directory)
cd docs
make html
```

### Code Quality
```bash
# Install and use pre-commit hooks
pip install pre-commit
pre-commit install

# Format code (if using poetry dev dependencies)
poetry run black .
poetry run isort .
```

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Or using poetry
poetry install
```

### Running FinRL
```bash
# Main training/testing/trading modes
python finrl/main.py --mode=train
python finrl/main.py --mode=test
python finrl/main.py --mode=trade

# POE (Portfolio Optimization Environment) experiments
python poe/examples/run_experiment.py --experiment EIIE
python poe/examples/run_all_experiments.py
```

## Architecture Overview

### Core Structure
- **finrl/**: Main framework with train-test-trade pipeline
  - **applications/**: Financial trading applications (stock trading, crypto, portfolio allocation, etc.)
  - **agents/**: RL algorithm implementations (ElegantRL, Ray RLlib, Stable-baselines3)
  - **meta/**: Market environments and data processing (environments, data processors, preprocessors)
  - **config.py**: Central configuration for dates, technical indicators, model parameters
  - **train.py/test.py/trade.py**: Core pipeline modules

- **poe/**: Portfolio Optimization Environment for advanced portfolio management
  - **config/experiments/**: Experiment configurations (EIIE variants)
  - **models/**: Model training and validation
  - **data/**: Data loading utilities
  - **feature/**: Experiment handling and database integration

### Key Configuration Files
- **finrl/config.py**: Main configuration with training dates, technical indicators, model hyperparameters
- **finrl/config_tickers.py**: Stock ticker lists (DOW_30_TICKER, etc.)
- **poe/config/base_config.py**: POE-specific configuration with data paths

### Data Sources
The framework supports 15+ data sources including:
- YahooFinance, Alpaca, Binance, CCXT (crypto)
- WRDS, Tushare, JoinQuant (professional/academic)
- Real-time and historical OHLCV data with technical indicators

### Environment Types
- **env_stock_trading/**: Stock trading environments with various penalty mechanisms
- **env_portfolio_optimization/**: Portfolio allocation environments
- **env_cryptocurrency_trading/**: Crypto trading environments

## Development Guidelines

### Code Standards
- Use pre-commit hooks for code quality
- Follow PEP format for inline documentation
- Organize code into classes and functions
- Write tests for new functionality
- Reference issues in PRs and tag maintainers

### Branch Strategy
- Create branches from "staging" (not "master")
- Submit PRs to "staging" branch
- Managers merge staging to master every 2-4 weeks

### Model Development
- Models support multiple RL libraries (ElegantRL, Stable-baselines3, Ray RLlib)
- Default technical indicators: ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma']
- Training uses configurable date ranges and data splits
- Support for paper trading via Alpaca API

### POE Experiments
- Use experiment configs in `poe/config/experiments/`
- Models saved as `policy_{name}_{features}_{window}_{id}.pt`
- Database integration for experiment tracking
- Support for EIIE and GPM architectures

## Important Notes
- Requires API keys for live trading (set in config_private.py)
- GPU recommended for training (automatic CUDA detection)
- Time-series data must have consistent date ranges for clean_data method
- Features should not exceed 10% of total data points for optimal performance