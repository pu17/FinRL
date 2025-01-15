import pandas as pd
import os
from datetime import datetime

def load_data(data_dir, start_date, end_date, ticker_list):
    """
    加载并划分数据集为训练集和测试集。
    
    参数:
        data_dir (str): 数据目录。
        start_date (str): 开始日期。
        end_date (str): 结束日期。
        ticker_list (list): 股票代码列表。
    
    返回:
        tuple: (df_train, df_test)
    """
    df_portfolio = pd.read_csv(data_dir)

    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    # 确保日期格式为 '%Y-%m-%d'
    start_date = datetime.strptime(start_date, '%Y%m%d').strftime('%Y-%m-%d') if isinstance(start_date, str) else start_date
    end_date = end_date.strftime('%Y-%m-%d') if isinstance(end_date, datetime) else end_date
    
    ticker_list = [tic.replace('.SS', '.SH') if tic.endswith('.SS') else tic for tic in ticker_list]
    
    # 根据日期和股票代码过滤数据
    df_filtered = df_portfolio[
        (df_portfolio["date"] >= start_date) & 
        (df_portfolio["date"] <= end_date) & 
        (df_portfolio["tic"].isin(ticker_list))
    ]
    print("过滤后后:", df_filtered["tic"].unique())

    # 划分训练集和测试集
    df_train = df_filtered[df_filtered["date"] < "2024-03-01"]
    df_test = df_filtered[df_filtered["date"] >= "2024-03-01"]
    
    return df_train, df_test

def get_unique_stocks(df_train):
    """
    获取训练数据中的唯一股票代码。
    
    参数:
        df_train (pd.DataFrame): 训练数据。
    
    返回:
        list: 唯一股票代码列表。
    """
    return df_train['tic'].unique().tolist()