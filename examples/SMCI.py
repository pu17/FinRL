print("Running the correct 1 china_A.py")
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from IPython import display
display.set_matplotlib_formats("svg")

import sys
sys.path.append('/Users/pu17/Documents/stock/FinRL')

from finrl.meta.data_processor import DataProcessor 
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
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

pd.options.display.max_columns = None

print("ALL Modules have been imported!")

check_and_make_directories(
    [DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR]
)

ticker_list = ["SMCI"]

TRAIN_START_DATE = "2024-01-01"
TRAIN_END_DATE = "2024-01-01"
TRADE_START_DATE = "2024-04-01"
TRADE_END_DATE = "2024-09-03"

TIME_INTERVAL = "1d"

p = DataProcessor(
    data_source="yahoofinance",
    start_date=TRAIN_START_DATE,
    end_date=TRADE_END_DATE,
    time_interval=TIME_INTERVAL
)

data_df = p.download_data(ticker_list=ticker_list,
    start_date=TRAIN_START_DATE,
    end_date=TRADE_END_DATE,
    time_interval=TIME_INTERVAL)

cleaned_data = p.clean_data(data_df)

# 添加技术指标
cleaned_data = p.add_technical_indicator(cleaned_data, config.INDICATORS)
# cleaned_data = p.fillna(cleaned_data)

print(f"p.dataframe: {cleaned_data}")

from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split

# 使用 FeatureEngineer 进行特征工程
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=config.INDICATORS,
    use_vix=True,  # 使用 VIX 指数
    use_turbulence=True,  # 使用 Turbulence 指数
    user_defined_feature=False,
)

# 预处理数据
processed_data = fe.preprocess_data(cleaned_data)
print(f"Processed Data:\n{processed_data.head()}")

# 获取唯一的股票代码列表和日期列表
list_ticker = processed_data["tic"].unique().tolist()  # 获取唯一的股票代码
list_date = list(pd.date_range(processed_data['date'].min(), processed_data['date'].max()).astype(str))  # 获取日期范围

# 生成所有日期和股票代码的组合
combination = list(itertools.product(list_date, list_ticker))

# 将组合转换为 DataFrame，并与原始数据合并
processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(processed_data, on=["date", "tic"], how="left")

# 确保合并后的数据只保留原始数据中的日期
processed_full = processed_full[processed_full['date'].isin(processed_data['date'])]

# 根据日期和股票代码排序
processed_full = processed_full.sort_values(['date', 'tic'])

# 填充缺失值，用 0 进行填充
processed_full = processed_full.fillna(0)

# 打印处理后的数据前几行
print(f"Processed Full Data:\n{processed_full.head()}")

# 将数据切分为训练集和交易集
train = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
trade = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)

# 打印训练集和交易集的长度
print(f"Training data length: {len(train)}")
print(f"Trading data length: {len(trade)}")

# 保存训练集和交易集为 CSV 文件
train.to_csv('train_data.csv', index=False)
trade.to_csv('trade_data.csv', index=False)

print("Training and trading datasets have been saved as CSV files.")