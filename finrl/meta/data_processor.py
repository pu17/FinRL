from __future__ import annotations

import numpy as np
import pandas as pd

from finrl.meta.data_processors.processor_alpaca import AlpacaProcessor as Alpaca
from finrl.meta.data_processors.processor_wrds import WrdsProcessor as Wrds
from finrl.meta.data_processors.processor_yahoofinance import (
    YahooFinanceProcessor as YahooFinance,
)
from finrl.meta.data_processors.processor_tushare import TushareProcessor  


class DataProcessor:
    def __init__(self, data_source, tech_indicator=None, vix=None, **kwargs):
        # 调试输出
        print(f"Data source provided: {data_source}")
        if data_source == "alpaca":
            API_KEY = kwargs.get("API_KEY")
            API_SECRET = kwargs.get("API_SECRET")
            API_BASE_URL = kwargs.get("API_BASE_URL")
            self.processor = Alpaca(API_KEY, API_SECRET, API_BASE_URL)

        elif data_source == "wrds":
            self.processor = Wrds()

        elif data_source == "yahoofinance":
            self.processor = YahooFinance()

        elif data_source == "tushare":
            print("tushare")
            API_TOKEN = kwargs.get("token")
            if not API_TOKEN:
                raise ValueError("Please provide a valid Tushare API token!")
            self.processor = TushareProcessor(API_TOKEN)

        else:
            raise ValueError("Data source input is NOT supported yet.")

        self.tech_indicator_list = tech_indicator
        self.vix = vix

    def download_data(
        self, ticker_list, start_date, end_date, time_interval
    ) -> pd.DataFrame:
        df = self.processor.download_data(
            ticker_list=ticker_list,
            start_date=start_date,
            end_date=end_date,
            time_interval=time_interval,
        )
        return df

    def clean_data(self, df) -> pd.DataFrame:
        df = self.processor.clean_data(df)

        return df

    def add_technical_indicator(self, df, tech_indicator_list) -> pd.DataFrame:
        self.tech_indicator_list = tech_indicator_list
        df = self.processor.add_technical_indicator(df, tech_indicator_list)

        return df

    def add_turbulence(self, df) -> pd.DataFrame:
        df = self.processor.add_turbulence(df)

        return df

    def add_vix(self, df) -> pd.DataFrame:
        df = self.processor.add_vix(df)

        return df

    def add_turbulence(self, df) -> pd.DataFrame:
        df = self.processor.add_turbulence(df)

        return df

    def add_vix(self, df) -> pd.DataFrame:
        df = self.processor.add_vix(df)

        return df

    def add_vixor(self, df) -> pd.DataFrame:
        df = self.processor.add_vixor(df)

        return df

    def df_to_array(self, df, if_vix) -> np.array:
        price_array, tech_array, turbulence_array = self.processor.df_to_array(
            df, self.tech_indicator_list, if_vix
        )
        # fill nan and inf values with 0 for technical indicators
        tech_nan_positions = np.isnan(tech_array)
        tech_array[tech_nan_positions] = 0
        tech_inf_positions = np.isinf(tech_array)
        tech_array[tech_inf_positions] = 0

        return price_array, tech_array, turbulence_array
