import pandas as pd
import os

def load_data(data_dir):
    """
    加载并划分数据集为训练集和测试集。
    
    参数:
        config (dict): 配置字典。
    
    返回:
        tuple: (df_train, df_test)
    """
    DATA_PATH = os.path.join(data_dir, 'df_portfolio.csv')
    df_portfolio = pd.read_csv(DATA_PATH)
    
    
    df_train = df_portfolio[
        (df_portfolio["date"] >= "2018-01-01") & (df_portfolio["date"] < "2024-03-01")
    ]
    df_test = df_portfolio[
        (df_portfolio["date"] >= "2024-03-01") & (df_portfolio["date"] <= "2024-11-18")
    ]
    
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