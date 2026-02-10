import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import tushare as ts
import itertools
from stable_baselines3.common.logger import configure
import matplotlib.pyplot as plt
from stable_baselines3 import A2C, DDPG, PPO, TD3, SAC

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl import config_tickers
from finrl.config import INDICATORS
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import TRAINED_MODEL_DIR, RESULTS_DIR
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

# 设置日期常量
TRAIN_START_DATE = '2019-01-04'
TRAIN_END_DATE = '2022-01-01'
TRADE_START_DATE = '2022-01-01'
TRADE_END_DATE = '2024-10-09'

# 设置文件路径常量
MARKET_BREADTH_FILE_PATH = '/Users/pu17/Documents/stock/stock_price_prediction/data/processed/market_breadth_all_exchanges.csv'

# 下载数据
def download_data():
    sz_df_finrl = YahooDownloader(
        start_date=TRAIN_START_DATE,
        end_date=TRADE_END_DATE,
        ticker_list=['000001.SS', '399001.SZ', '603000.SS']
    ).fetch_data()
    return sz_df_finrl

# 预处理数据
def preprocess_data(df):
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=True,
        use_turbulence=True,
        user_defined_feature=False
    )
    processed = fe.preprocess_data(df)
    return processed

# 读取市场宽度数据
def read_market_breadth_data():
    try:
        market_breadth_data = pd.read_csv(MARKET_BREADTH_FILE_PATH)
        market_breadth_data['date'] = pd.to_datetime(market_breadth_data['trade_date']).dt.strftime('%Y-%m-%d')
        return market_breadth_data
    except FileNotFoundError:
        print(f"文件未找到，请检查路径：{MARKET_BREADTH_FILE_PATH}")
        return None
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        return None

# 选择交易所
def select_exchange(row):
    if row['tic'] == '399001.SZ':
        return 'SZSE'
    elif row['tic'] in ['000001.SS', '603000.SS']:
        return 'SSE'
    return None

# 合并数据
def merge_data(processed, market_breadth_data):
    processed['exchange'] = processed.apply(select_exchange, axis=1)
    merged_df = pd.merge(processed, market_breadth_data, how='left', on=['date', 'exchange'])
    merged_df.drop(columns=['trade_date', 'exchange'], inplace=True)
    return merged_df

# 填充数据
def fill_missing_data(processed):
    list_ticker = processed["tic"].unique().tolist()
    list_date = list(pd.date_range(processed['date'].min(), processed['date'].max()).astype(str))
    combination = list(itertools.product(list_date, list_ticker))

    processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(processed, on=["date", "tic"], how="left")
    processed_full = processed_full[processed_full['date'].isin(processed['date'])]
    processed_full = processed_full.sort_values(['date', 'tic'])
    processed_full = processed_full.fillna(0)
    return processed_full

# 构建训练环境
def build_train_environment(train):
    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }

    e_train_gym = StockTradingEnv(df=train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    return env_train

# 训练模型
def train_models(env_train, total_timesteps=50000):
    agent = DRLAgent(env=env_train)
    
    models = ["a2c", "ppo", "ddpg", "td3", "sac"]
    trained_models = {}

    for model_name in models:
        print(f"开始训练 {model_name.upper()} 模型...")
        model = agent.get_model(model_name)

        tmp_path = RESULTS_DIR + f'/{model_name}'
        new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        model.set_logger(new_logger)

        trained_model = agent.train_model(model=model, 
                                          tb_log_name=model_name,
                                          total_timesteps=total_timesteps)
        
        trained_models[model_name] = trained_model
        print(f"{model_name.upper()} 模型训练完成。")

    return trained_models

# 回测函数
def backtest(trade_data, trained_models):
    # 构建回测环境
    stock_dimension = len(trade_data.tic.unique())
    state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
    
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": [0] * stock_dimension,
        "buy_cost_pct": [0.001] * stock_dimension,
        "sell_cost_pct": [0.001] * stock_dimension,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }

    e_trade_gym = StockTradingEnv(df=trade_data, **env_kwargs)

    results = {}
    for model_name, model in trained_models.items():
        print(f"开始回测 {model_name.upper()} 模型...")
        df_account_value, df_actions = DRLAgent.DRL_prediction(model=model, environment=e_trade_gym)
        results[model_name] = df_account_value

    return results

# 绘制回测结果
def plot_backtest_results(results):
    plt.figure(figsize=(10, 6))
    for model_name, df_account_value in results.items():
        plt.plot(df_account_value['date'], df_account_value['account_value'], label=model_name.upper())
    
    plt.title('Backtest Results')
    plt.xlabel('Date')
    plt.ylabel('Account Value')
    plt.legend()
    plt.savefig(RESULTS_DIR + '/backtest_results.png')
    plt.close()

# 主函数
def main():
    try:
        # 下载数据
        sz_df_finrl = download_data()
        
        # 预处理数据
        processed = preprocess_data(sz_df_finrl)
        
        # 读取市场宽度数据
        market_breadth_data = read_market_breadth_data()
        
        if market_breadth_data is not None:
            # 合并数据
            processed = merge_data(processed, market_breadth_data)
        else:
            print("无法读取市场宽度数据，将继续处理没有市场宽度数据的股票数据。")
        
        # 填充缺失数据
        processed_full = fill_missing_data(processed)
        processed_full.head(10)
        
        # 分割数据
        train = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
        trade = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)
        
        # 保存数据
        train.to_csv('train_data.csv', index=False)
        trade.to_csv('trade_data.csv', index=False)
        
        print("数据处理完成，已保存为 CSV 文件。")

        # 构建训练环境
        env_train = build_train_environment(train)

        # 训练模型
        trained_models = train_models(env_train)

        # 保存训练好的模型
        current_date = datetime.now().strftime("%Y%m%d")
        for model_name, model in trained_models.items():
            model_path = f"{TRAINED_MODEL_DIR}/agent_{model_name}_{current_date}"
            model.save(model_path)
        print(f"所有模型训练完成并保存。保存日期: {current_date}")

        # 执行回测
        backtest_results = backtest(trade, trained_models)

        # 绘制回测结果
        plot_backtest_results(backtest_results)
        print("回测完成，结果已保存为图片。")

    except Exception as e:
        print(f"处理过程中发生错误：{e}")

if __name__ == "__main__":
    check_and_make_directories([TRAINED_MODEL_DIR, RESULTS_DIR])
    main()
