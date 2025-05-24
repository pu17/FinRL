"""Contains methods and classes to collect data from
tushare API
"""

from __future__ import annotations

import pandas as pd
import tushare as ts
from tqdm import tqdm
import logging


class TushareDownloader:
    """Provides methods for retrieving daily stock data from
    tushare API
    Attributes
    ----------
        start_date : str
            start date of the data (modified from config.py)
        end_date : str
            end date of the data (modified from config.py)
        ticker_list : list
            a list of stock tickers (modified from config.py)
    Methods
    -------
    fetch_data()
        Fetches data from tushare API
    date: date
    Open: opening price
    High: the highest price
    Close: closing price
    Low: lowest price
    Volume: volume
    Price_change: price change
    P_change: fluctuation
    ma5: 5-day average price
    Ma10: 10 average daily price
    Ma20:20 average daily price
    V_ma5:5 daily average
    V_ma10:10 daily average
    V_ma20:20 daily average
    """

    def __init__(self, start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list
        self.ticker_list = self._add_stock_suffix(self.ticker_list)
        self.api_token = 'dcd771227705b0513e5a9cd4903ec0e0aedf182d7308af30afc3388d'
        ts.set_token(self.api_token)
        self.pro = ts.pro_api()

    def fetch_data(self) -> pd.DataFrame:
        """Fetches data from Tushare
        Returns
        -------
        `pd.DataFrame`
            Columns: date, open, high, low, close, volume, tic
        """
        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        for tic in tqdm(self.ticker_list, total=len(self.ticker_list)):
            # 获取 ts_code 的资金流数据
            if tic.endswith('.SS'):
                tic = tic.replace('.SS', '.SH')

            temp_df = self.pro.daily(ts_code=tic, start_date=self.start_date, end_date=self.end_date)
            print(temp_df.head())
            temp_df["tic"] = tic
            # Select and rename necessary columns
            temp_df = temp_df.rename(columns={
                "trade_date": "date",
                "vol": "volume"
            })

            if temp_df.empty:
                logging.warning(f"temp_df 为空，跳过此数据块。股票代码: {tic}")
                continue

            temp_df = temp_df[["date", "open", "high", "low", "close", "volume", "tic"]]

            # Concatenate data
            data_df = pd.concat([data_df, temp_df], axis=0, ignore_index=True)

        # Convert date to datetime and format it as YYYY-MM-DD
        data_df["date"] = pd.to_datetime(data_df["date"], format='%Y%m%d')
        data_df["date"] = data_df["date"].dt.strftime("%Y-%m-%d")

        # Sort data
        data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)

        # Create day of the week column (Monday = 0)
        data_df["day"] = pd.to_datetime(data_df["date"]).dt.dayofweek

        # Drop missing data
        data_df = data_df.dropna()

        # rank desc
        data_df = data_df.sort_index(axis=0, ascending=False)
        # convert date to standard string format, easy to filter

        print("Shape of DataFrame: ", data_df.shape)
        print(data_df.tail())

        return data_df
    def get_all_a_stocks(self):
        """获取所有A股股票代码列表（上交所和深交所）
        
        Returns:
        -------
        list
            包含所有A股股票代码的列表，格式为 'XXXXXX.SH' 或 'XXXXXX.SZ'
        """
        
        # 获取基础信息
        data = self.pro.stock_basic(exchange='', list_status='L')
        
        # 过滤A股市场（SH和SZ）
        a_stocks = data[data['market'].isin(['主板', '中小板', '创业板', '科创板'])]
        
        # 获取股票代码列表
        a_stock_list = a_stocks['ts_code'].tolist()
        
        print(f"获取到 {len(a_stock_list)} 只A股股票")
        return a_stock_list

    def select_equal_rows_stock(self, df):
        df_check = df.tic.value_counts()
        df_check = pd.DataFrame(df_check).reset_index()
        df_check.columns = ["tic", "counts"]
        mean_df = df_check.counts.mean()
        equal_list = list(df.tic.value_counts() >= mean_df)
        names = df.tic.value_counts().index
        select_stocks_list = list(names[equal_list])
        df = df[df.tic.isin(select_stocks_list)]
        return df
    
    def _add_stock_suffix(self, tickers: list) -> list:
        """
        自动为股票代码添加交易所后缀
        规则：
        - 6开头：沪市A股 → .SS
        - 0/3开头：深市 → .SZ
        - 已包含后缀的代码保持不变
        """
        processed = []
        for tic in tickers:
            # 分离基础代码和已有后缀
            base_code = tic.split('.')[0]  # 补足6位
            
            # 判断交易所
            if base_code.startswith(('6', '9')):
                suffix = '.SH'
            elif base_code.startswith(('0', '3')):
                suffix = '.SZ'
            else:
                raise ValueError(f"无法识别的股票代码格式: {tic}")
            
            # 保留原始格式（如有后缀则保持）
            if '.' in tic:
                processed.append(tic)
            else:
                processed.append(f"{base_code}{suffix}")
        return processed
