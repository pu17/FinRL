import tushare as ts
import pandas as pd
from tqdm import tqdm

class TushareProcessor:
    def __init__(self, api_token: str):
        # 初始化 Tushare API
        ts.set_token(api_token)
        self.pro = ts.pro_api()

    def download_data(self, ticker_list, start_date, end_date, time_interval="D") -> pd.DataFrame:
        """从 Tushare 下载股票数据"""
        data_df = pd.DataFrame()
        for tic in tqdm(ticker_list, total=len(ticker_list)):
            try:
                # 使用 ts.pro_bar 获取历史数据
                temp_df = ts.pro_bar(ts_code=tic, start_date=start_date, end_date=end_date, freq=time_interval)
                if temp_df is not None and not temp_df.empty:
                    temp_df["tic"] = tic
                    data_df = pd.concat([data_df, temp_df], ignore_index=True)
                else:
                    print(f"No data returned for {tic}: Possible data issue.")
            except Exception as e:
                print(f"Error downloading data for {tic}: {e}")

        # 检查是否有数据
        if data_df.empty:
            print("No data downloaded for the provided tickers.")
            return pd.DataFrame()  # 返回空的 DataFrame，避免后续操作出错

        # 检查 'trade_date' 列是否存在
        if 'trade_date' in data_df.columns:
            data_df.rename(columns={'trade_date': 'timestamp'}, inplace=True)
        else:
            print("No 'trade_date' (or 'timestamp') column found in the downloaded data!")
            return pd.DataFrame()  # 返回空的 DataFrame，避免后续操作出错

        # 将 'timestamp' 列设置为 datetime 类型
        data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
        data_df = data_df.reset_index(drop=True)

        # 保留与 YahooFinanceProcessor 一致的列
        data_df = data_df[['timestamp', 'open', 'high', 'low', 'close', 'vol', 'tic']]
        data_df.rename(columns={'vol': 'volume'}, inplace=True)

        return data_df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理股票数据"""
        # 删除不需要的列
        df = df.drop(
            [
                "price_change",
                "p_change",
                "ma5",
                "ma10",
                "ma20",
                "v_ma5",
                "v_ma10",
                "v_ma20",
            ],
            axis=1, errors='ignore'
        )
        
        # 设置时间戳格式为字符串YYYY-MM-DD
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d")
        df = df.sort_values(by=["timestamp", "tic"]).reset_index(drop=True)
        
        return df

    def add_technical_indicator(self, df: pd.DataFrame, tech_indicator_list: list) -> pd.DataFrame:
        """添加技术指标到数据中"""
        # Example: 使用滚动窗口计算简单移动均线
        for indicator in tech_indicator_list:
            if indicator == "ma5":
                df["ma5"] = df["close"].rolling(window=5).mean()
            # 根据需要添加更多的技术指标
        return df

    def df_to_array(self, df: pd.DataFrame, tech_indicator_list: list, if_vix: bool) -> tuple:
        """将 DataFrame 转换为 numpy 数组"""
        price_array = df.pivot(index="timestamp", columns="tic", values="close").values
        tech_array = df[tech_indicator_list].values
        turbulence_array = df["turbulence"].values if "turbulence" in df.columns else None
        return price_array, tech_array, turbulence_array