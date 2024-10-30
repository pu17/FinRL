# portfolio_optimization.py
import logging
logging.getLogger('matplotlib.font_manager').disabled = True

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import GroupByScaler
from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import PortfolioOptimizationEnv
from finrl.agents.portfolio_optimization.models import DRLAgent
from finrl.agents.portfolio_optimization.architectures import EIIE

# 设置设备
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 读取CSV文件
df_portfolio = pd.read_csv('/Users/pu17/Documents/stock/FinRL/examples/df_portfolio.csv')
# 定义要过滤的股票代码列表
# target_tics = ['000001.SS', '399001.SZ']

# 使用布尔索引过滤出指定股票代码的数据
# df_portfolio = df_portfolio[df_portfolio['tic'].isin(target_tics)]
# print(df_portfolio.head())

# 提取训练数据
df_portfolio_train = df_portfolio[(df_portfolio["date"] >= "2018-01-01") & (df_portfolio["date"] < "2023-03-01")]
df_portfolio_2023 = df_portfolio[(df_portfolio["date"] >= "2023-03-01") & (df_portfolio["date"] < "2024-01-01")]
df_portfolio_2024 = df_portfolio[(df_portfolio["date"] >= "2024-01-01") & (df_portfolio["date"] <= "2024-10-22")]

# 定义特征列表
features = [
    'open', 'high', 'low', 'close','volume', 'day', 'macd','boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma','close_60_sma',
    'up_down_ratio', 'market_breadth',
    'net_sm_amount','net_md_amount', 'net_lg_amount','net_elg_amount',
    'net_sm_pct', 'net_md_pct', 'net_lg_pct', 'net_elg_pct',
    'spring_festival_pre_holiday', 'spring_festival_post_holiday',
    'labor_day_pre_holiday', 'labor_day_post_holiday',
    'national_day_pre_holiday', 'national_day_post_holiday', 'dayofmonth',
    'dayofyear'
]



# 创建 PortfolioOptimizationEnv 环境
environment = PortfolioOptimizationEnv(
    df_portfolio_train,
    initial_amount=100000,
    comission_fee_pct=0.0025,
    time_window=10,
    features=features,
    normalize_df=None
)

# 设置 PolicyGradient 参数
model_kwargs = {
    "lr": 0.001,
    "policy": EIIE,
}

# 设置 EIIE 的参数
policy_kwargs = {
    "k_size": 3,
    "time_window": 10,
    "initial_features": len(features),  # 使用特征列表的长度
}

# 加载已保存的模型参数



# 创建并训练模型
model = DRLAgent(environment).get_model("pg", device, model_kwargs, policy_kwargs)
# 定义模型参数路径
model_path = "/Users/pu17/Documents/stock/FinRL/examples/policy_EIIE_34_10.pt"

# 加载保存的模型参数
# model.train_policy.load_state_dict(torch.load(model_path))

DRLAgent.train_model(model, episodes=660)

# 保存模型
torch.save(model.train_policy.state_dict(), model_path)