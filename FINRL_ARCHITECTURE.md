# FinRL Architecture Reference

## What is FinRL?
FinRL是一个金融强化学习框架，专门用于股票交易、投资组合管理和加密货币交易的AI策略开发。

## Architecture Overview
FinRL采用三层架构设计：
- **Environment Layer (环境层)**: 模拟金融市场，定义状态和动作空间
- **Agent Layer (智能体层)**: 提供深度强化学习算法 
- **Application Layer (应用层)**: 实现具体的金融交易任务

核心流程：**Train** → **Test** → **Trade** (训练 → 回测 → 实盘)

## Environment Layer (环境层)
环境层模拟不同类型的金融市场，每种环境专门针对特定的交易任务设计。

### 1. StockTradingEnv - 股票交易环境
- **用途**: 买卖个股或多股票组合
- **动作空间**: [-1, 1] 连续值，表示买入/卖出/持有信号
- **状态空间**: 现金余额 + 持股数量 + 股价 + 技术指标
- **适用场景**: 日内交易、趋势跟踪、技术分析策略
- **兼容算法**: PPO, SAC, A2C, DDPG, TD3

### 2. StockPortfolioEnv - 投资组合分配环境  
- **用途**: 多资产投资组合的权重分配
- **动作空间**: [0, 1] 权重值，必须总和为1
- **状态空间**: 协方差矩阵 + 技术指标 + 收益率数据
- **适用场景**: 资产配置、风险平价、动态再平衡
- **兼容算法**: A2C, PPO

### 3. PortfolioOptimizationEnv - 深度投资组合优化环境
- **用途**: 基于深度学习的投资组合权重优化
- **动作空间**: [0, 1] 投资组合权重（包含现金）
- **状态空间**: 三维张量 [特征数, 股票数, 时间窗口] 
- **适用场景**: 基于时间序列的复杂投资组合策略
- **兼容算法**: **仅限EIIE**（专用算法）

### 4. CryptocurrencyTradingEnv - 加密货币交易环境
- **用途**: 7×24小时加密货币交易
- **动作空间**: [-1, 1] 买卖信号
- **状态空间**: 现金 + 加密货币持仓 + 价格 + 技术指标
- **适用场景**: 数字货币交易、套利策略
- **兼容算法**: SAC, TD3, PPO

## Agent Layer (智能体层)
智能体层提供各种深度强化学习算法，支持多个RL框架集成。

### Stable Baselines3 算法族
基于成熟的SB3框架，提供标准RL算法实现：

- **PPO (Proximal Policy Optimization)**
  - 类型: On-policy 算法
  - 特点: 稳定、易用、通用性强
  - 最适合: 股票交易、通用任务
  - 参数: n_steps=2048, learning_rate=0.00025

- **SAC (Soft Actor-Critic)**  
  - 类型: Off-policy 算法
  - 特点: 高采样效率、探索能力强
  - 最适合: 股票交易、加密货币交易
  - 参数: buffer_size=100000, learning_rate=0.0001

- **A2C (Advantage Actor-Critic)**
  - 类型: On-policy 算法
  - 特点: 简单、快速、内存友好
  - 最适合: 投资组合分配
  - 参数: n_steps=5, learning_rate=0.0007

- **DDPG/TD3**
  - 类型: Off-policy 连续控制算法
  - 特点: 确定性策略、适合连续动作
  - 最适合: 高频交易、精确控制

### 专用投资组合算法
- **EIIE (Ensemble of Identical Independent Evaluators)**
  - 架构: CNN + LSTM + 全连接层
  - 输入: 时间序列张量 [特征, 股票, 时间窗口]
  - 输出: 投资组合权重分配
  - **专用性**: 只能与PortfolioOptimizationEnv配合使用

### ElegantRL 优化算法
提供GPU加速和内存优化版本的经典算法，训练速度更快。

## ⚠️ 重要兼容性规则
理解这些规则对于避免常见错误至关重要：

### 绝对禁止的组合
- ❌ **EIIE + StockTradingEnv**: 状态空间不匹配（EIIE需要张量输入）
- ❌ **PPO/SAC + PortfolioOptimizationEnv**: 无法处理时间序列张量输入
- ❌ **传统RL算法 + PortfolioOptimizationEnv**: 架构不兼容

### 推荐的最佳组合  
- ✅ **股票交易**: StockTradingEnv + PPO/SAC
- ✅ **投资组合分配**: StockPortfolioEnv + A2C/PPO
- ✅ **投资组合优化**: PortfolioOptimizationEnv + EIIE
- ✅ **加密货币交易**: CryptocurrencyTradingEnv + SAC/TD3

### 兼容性原理
- **状态空间匹配**: 环境输出的状态格式必须与算法输入格式兼容
- **动作空间匹配**: 算法输出的动作必须符合环境预期的格式
- **任务特化**: 某些算法专门为特定任务设计（如EIIE用于投资组合优化）

## Application Layer (应用层)
应用层提供具体的金融任务实现，是用户直接交互的接口：

### 股票交易应用 (`finrl/applications/stock_trading/`)
- **单股票交易**: 专注单一股票的策略
- **多股票交易**: 同时交易多只股票
- **集成交易**: 多算法集成决策
- **滚动窗口交易**: 动态更新训练窗口

### 投资组合应用 (`finrl/applications/portfolio_allocation/`)
- 多资产权重分配策略
- 风险平价和动态再平衡

### 加密货币应用 (`finrl/applications/cryptocurrency_trading/`)
- 24/7交易策略
- 多交易所套利

## 核心文件结构
```
finrl/
├── main.py                                         # 主程序入口
├── config.py                                       # 全局配置文件
├── train.py / test.py / trade.py                   # 核心流程模块
├── meta/                                           # 环境和数据处理层
│   ├── data_processor.py                           # 数据处理核心
│   ├── env_stock_trading/env_stocktrading.py       # 股票交易环境
│   ├── env_portfolio_allocation/env_portfolio.py   # 投资组合分配环境
│   ├── env_portfolio_optimization/                 # 投资组合优化环境
│   └── preprocessor/                               # 数据预处理模块
├── agents/                                         # 智能体算法层
│   ├── stablebaselines3/models.py                 # SB3算法集成
│   ├── elegantrl/models.py                        # ElegantRL集成
│   └── portfolio_optimization/                     # EIIE等专用算法
└── applications/                                   # 应用任务层
    ├── stock_trading/                              # 股票交易应用
    ├── portfolio_allocation/                       # 投资组合应用
    └── cryptocurrency_trading/                     # 加密货币应用
```

## 数据流处理管道
```
原始数据 → DataProcessor → FeatureEngineer → Environment → Agent → 训练/测试
```

### 数据源支持
- **股票数据**: Yahoo Finance, Alpaca, WRDS
- **加密货币数据**: Binance, CCXT
- **技术指标**: MACD, RSI, Bollinger Bands等

## 使用指南

### 新手入门建议
1. **从股票交易开始**: StockTradingEnv + PPO是最稳定的入门组合
2. **逐步增加复杂度**: 先掌握基础交易，再尝试投资组合优化
3. **充分回测**: 在实盘前进行充分的历史数据验证

### 高级用户指南
- **投资组合优化**: 需要深入理解时间序列处理和EIIE架构
- **多算法集成**: 可以结合多种算法的优势
- **自定义环境**: 根据特定需求扩展现有环境

### 性能优化建议
- 使用ElegantRL获得更快训练速度
- GPU加速训练（CUDA支持）
- 合理设置batch_size和buffer_size
- 使用Tensorboard监控训练过程

## 快速启动代码模板

### 股票交易示例
```python
# 1. 数据准备
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
df = YahooDownloader(start_date="2020-01-01", end_date="2023-01-01", 
                     ticker_list=["AAPL", "TSLA"]).fetch_data()

# 2. 环境创建
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
env = StockTradingEnv(df=df, stock_dim=2, initial_amount=100000, 
                      hmax=100, action_space=2, state_space=50)

# 3. 智能体训练
from finrl.agents.stablebaselines3.models import DRLAgent
agent = DRLAgent(env)
model = agent.get_model("ppo")
trained_model = agent.train_model(model, total_timesteps=80000)
```

### 投资组合分配示例
```python
# 环境创建（注意不同的参数）
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
env = StockPortfolioEnv(df=df, stock_dim=30, initial_amount=100000,
                        transaction_cost_pct=0.001)

# 使用A2C算法
agent = DRLAgent(env)
model = agent.get_model("a2c")
```

### 投资组合优化示例
```python
# 特殊的环境设置（需要时间窗口）
from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import PortfolioOptimizationEnv
env = PortfolioOptimizationEnv(df=df, initial_amount=100000, 
                               time_window=4, features=["close","high","low"])

# 必须使用专用算法
from finrl.agents.portfolio_optimization.models import DRLAgent
from finrl.agents.portfolio_optimization.architectures import EIIE

agent = DRLAgent(env)
model = agent.get_model("pg", policy_kwargs={"policy": EIIE, "time_window": 4})
```

## 常见问题解答

### Q: 为什么EIIE不能用于股票交易？
A: EIIE算法期望输入是三维张量[特征,股票,时间]，而StockTradingEnv输出的是一维向量[现金,持股,价格,指标]，数据格式不匹配。

### Q: 如何选择合适的算法？
A: 
- **新手或通用场景**: 选择PPO，稳定可靠
- **需要高探索性**: 选择SAC
- **投资组合分配**: 选择A2C或PPO  
- **投资组合优化**: 必须选择EIIE

### Q: 可以自定义环境吗？
A: 可以，继承相应的基类并实现必要的方法。但要确保状态空间和动作空间与选择的算法兼容。

---
**文档版本**: 2025-08-19  
**适用于**: FinRL框架最新版本  
**维护**: 基于代码库实际架构分析生成