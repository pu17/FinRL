import sys
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.tusharedownloader import TushareDownloader
from finrl.meta.preprocessor.preprocessors import GroupByScaler, FeatureEngineer
from finrl.config import INDICATORS
from datetime import datetime

def setup_environment():
    # 添加环境路径
    sys.path.append('/Users/pu17/Documents/stock/stock_price_prediction')

def collect_data():
    # 数据收集逻辑
    TOP_BRL = ['000001.SS', '399001.SZ', '603000.SS', '000035.SZ', '002261.SZ', '000938.SZ', '600547.SS', '600756.SS', '601899.SS', '601988.SS']
    
    # 获取今天的日期作为默认的结束日期
    end_date = datetime.today().strftime('%Y%m%d')
    print(end_date)
    
    portfolio_raw_df = TushareDownloader(
        start_date='20150101',
        end_date=end_date,
        ticker_list=TOP_BRL
    ).fetch_data()
    
    # 检查最大日期
    max_date = portfolio_raw_df['date'].max()
    print("Maximum date in DataFrame: ", max_date)
    
    # 检查重复项
    duplicate_rows = portfolio_raw_df.duplicated()
    num_duplicates = duplicate_rows.sum()
    print(f"Number of duplicate rows: {num_duplicates}")
    
    if num_duplicates > 0:
        print("Duplicate rows:")
        print(portfolio_raw_df[duplicate_rows])
    
    print("Shape of DataFrame: ", portfolio_raw_df.shape)
    return portfolio_raw_df

def preprocess_data(portfolio_raw_df):
    # 数据预处理逻辑
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=False,
        use_turbulence=False,
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
    processed = preprocess_with_custom_feature_engineer(processed)
    portfolio_raw_df = fill_missing_values(processed)
    df_portfolio = normalize_data(portfolio_raw_df)
    save_to_csv(df_portfolio)