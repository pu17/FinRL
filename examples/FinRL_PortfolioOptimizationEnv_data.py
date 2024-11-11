import sys
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import GroupByScaler, FeatureEngineer
from finrl.config import INDICATORS

def setup_environment():
    # 添加环境路径
    sys.path.append('/Users/pu17/Documents/stock/stock_price_prediction')

def collect_data():
    # 数据收集逻辑
    TOP_BRL = ["IAU", "YINN", "NVDA", 'GOOGL', 'AAPL']
    portfolio_raw_df = YahooDownloader(
        start_date='2016-01-01',
        end_date='2024-10-25',
        ticker_list=TOP_BRL
    ).fetch_data()
    print("Shape of DataFrame: ", portfolio_raw_df.shape)
    return portfolio_raw_df

def preprocess_data(portfolio_raw_df):
    # 数据预处理逻辑
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=True,
        use_turbulence=True,
        user_defined_feature=False
    )
    processed = fe.preprocess_data(portfolio_raw_df)
    return processed

def preprocess_with_custom_feature_engineer(processed):
    from feature.ch_feature_engineer import ChFeatureEngineer
    ch_fe = ChFeatureEngineer()
    # 连接到数据库并预处理数据
    portfolio_raw_df = ch_fe.preprocess_data(processed)
    print("Head of the DataFrame after custom preprocessing:")
    print(portfolio_raw_df.head())
    return portfolio_raw_df

def fill_missing_values(portfolio_raw_df):
    # 打印每列中缺失值的数量
    print("Missing values before filling:")
    print(portfolio_raw_df.isnull().sum())
    
    # 填充缺失值
    portfolio_raw_df = portfolio_raw_df.fillna(0)
    
    # 打印填充后每列中缺失值的数量
    print("Missing values after filling:")
    print(portfolio_raw_df.isnull().sum())
    return portfolio_raw_df

def normalize_data(processed):
    # 数据归一化逻辑
    portfolio_norm_df = GroupByScaler(by="tic", scaler=MaxAbsScaler).fit_transform(processed)
    portfolio_norm_df['date'] = portfolio_norm_df['date'].astype(str)
    return portfolio_norm_df

def save_to_csv(df_portfolio):
    # 保存数据到 CSV 文件
    df_portfolio.to_csv('df_portfolio.csv', index=False)
    print("Data saved to df_portfolio.csv")

if __name__ == "__main__":
    setup_environment()
    portfolio_raw_df = collect_data()
    processed = preprocess_data(portfolio_raw_df)
    # processed = preprocess_with_custom_feature_engineer(processed)
    portfolio_raw_df = fill_missing_values(processed)
    df_portfolio = normalize_data(portfolio_raw_df)
    save_to_csv(df_portfolio)