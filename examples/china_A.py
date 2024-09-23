print("Running the correct china_A.py")
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from IPython import display
display.set_matplotlib_formats("svg")

# 确保路径正确，能够正确导入 finrl 模块
import sys
sys.path.append('/Users/pu17/Documents/stock/FinRL')

from finrl.meta.data_processor import DataProcessor
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnvV2
from finrl.agents.stablebaselines3_models import DRLAgent
from finrl import config
from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE
)
# import pyfolio
# from pyfolio import timeseries
# import os

# 确保显示所有列
pd.options.display.max_columns = None

print("ALL Modules have been imported!")

### 创建文件夹
check_and_make_directories(
    [DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR]
)

### 下载数据、清理和特征工程

ticker_list = [
    "600000.SH", "600009.SH", "600016.SH", "600028.SH",
    "600030.SH", "600031.SH", "600036.SH", "600050.SH",
    "600104.SH", "600196.SH", "600276.SH", "600309.SH",
    "600519.SH", "600547.SH", "600570.SH"
]

TRAIN_START_DATE = "2015-01-01"
TRAIN_END_DATE = "2019-08-01"
TRADE_START_DATE = "2019-08-01"
TRADE_END_DATE = "2020-01-03"

TIME_INTERVAL = "1d"
kwargs = {}
kwargs["token"] = "27080ec403c0218f96f388bca1b1d85329d563c91a43672239619ef5"

# 使用最新的 DataProcessor 接口
p = DataProcessor(
    data_source="tushare",
    start_date=TRAIN_START_DATE,
    end_date=TRADE_END_DATE,
    time_interval=TIME_INTERVAL,
    **kwargs
)

# 下载和清理数据
p.download_data(ticker_list=ticker_list)
p.clean_data()
p.fillna()

# 添加技术指标
p.add_technical_indicator(config.INDICATORS)
p.fillna()

print(f"p.dataframe: {p.dataframe}")

### 特征工程
from finrl.meta.data_processors import FeatureEngineer

fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=config.INDICATORS,
    use_vix=False,  # 如果需要加入VIX, 这里可以设置为True
    use_turbulence=True,  # 如果需要加入Turbulence指数，这里可以设置为True
    user_defined_feature=False,
)

# 预处理数据
processed_data = fe.preprocess_data(p.dataframe)
print(f"Processed Data: {processed_data.head()}")

### 拆分训练数据集
train = p.data_split(processed_data, TRAIN_START_DATE, TRAIN_END_DATE)
print(f"len(train.tic.unique()): {len(train.tic.unique())}")
print(f"train.tic.unique(): {train.tic.unique()}")
print(f"train.head(): {train.head()}")
print(f"train.shape: {train.shape}")

# 股票维度和状态空间设置
stock_dimension = len