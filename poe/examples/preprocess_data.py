import os
import logging
import pandas as pd
import importlib.util
from poe.utils.logger import setup_logger
from poe.config.load_config import load_config
from finrl.meta.preprocessor.tusharedownloader import TushareDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, GroupByScaler
from sklearn.preprocessing import MaxAbsScaler
from datetime import datetime
from poe.config.base_config import BASE_PATH

def create_experiment_directories(experiment_name):
    data_dir = os.path.join(BASE_PATH, 'data', experiment_name)
    model_dir = os.path.join(BASE_PATH, 'models', experiment_name)
    log_dir = os.path.join(BASE_PATH, 'logs', experiment_name)
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    return data_dir, model_dir, log_dir

def collect_data(config):
    logging.info("开始数据收集。")
    data_params = config['data_params']
    
    # 设置默认的结束日期为今天
    end_date = data_params.get('end_date')
    if end_date is None:
        end_date = datetime.today().strftime('%Y%m%d')
    
    # 确保日期格式为 '%Y%m%d'
    start_date = data_params['start_date'].strftime('%Y%m%d') if isinstance(data_params['start_date'], datetime) else data_params['start_date']
    end_date = end_date.strftime('%Y%m%d') if isinstance(end_date, datetime) else end_date
    
    print(start_date,end_date)
    portfolio_raw_df = TushareDownloader(
        start_date=start_date,
        end_date=end_date,
        ticker_list=data_params['ticker_list']
    ).fetch_data()
    
    if portfolio_raw_df.empty:
        logging.error("收集到的 portfolio_raw_df 为空。")
        return portfolio_raw_df
    
    # 打印最大日期和 DataFrame 的形状
    max_date = portfolio_raw_df['date'].max()
    logging.info(f"DataFrame 中的最大日期: {max_date}")
    logging.info(f"DataFrame 的形状: {portfolio_raw_df.shape}")
    
    logging.info("数据收集完成。")
    return portfolio_raw_df

def preprocess_data(portfolio_raw_df):
    logging.info("开始数据预处理。")
    from finrl.config import INDICATORS
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False
    )
    processed = fe.preprocess_data(portfolio_raw_df)
    logging.info("数据预处理完成。")
    return processed

def custom_preprocess(processed):
    logging.info("开始自定义特征工程。")
    from feature.ch_feature_engineer import ChFeatureEngineer
    ch_fe = ChFeatureEngineer()
    portfolio_processed = ch_fe.preprocess_data(processed)
    logging.info("自定义特征工程完成。")
    return portfolio_processed

def fill_missing(portfolio_processed):
    logging.info("检查并填充缺失值。")
    logging.info("缺失值填充前：")
    logging.info(portfolio_processed.isnull().sum())
    
    portfolio_filled = portfolio_processed.fillna(0)
    
    logging.info("缺失值填充后：")
    logging.info(portfolio_filled.isnull().sum())
    return portfolio_filled

def normalize(portfolio_filled):
    logging.info("开始数据归一化。")
    portfolio_norm = GroupByScaler(by="tic", scaler=MaxAbsScaler).fit_transform(portfolio_filled)
    portfolio_norm['date'] = portfolio_norm['date'].astype(str)
    logging.info("数据归一化完成。")
    return portfolio_norm

def save(portfolio_norm, output_path):
    portfolio_norm.to_csv(output_path, index=False)
    logging.info(f"数据已保存至 {output_path}")


def main(experiment):
    
    # 加载实验配置
    experiment_config = load_config(experiment)
    
    # 合并配置，实验配置覆盖基础配置
    config = {**experiment_config}
    
    # 创建实验目录
    data_dir, model_dir, log_dir = create_experiment_directories(
        experiment
    )
    
    # 设置日志
    config['LOG_FILE'] = os.path.join(log_dir, 'experiment.log')
    setup_logger(config)
    logging.info(f"开始数据预处理实验：{experiment}")
    
    # 数据收集和预处理
    portfolio_raw_df = collect_data(config)
    if portfolio_raw_df.empty:
        logging.error("portfolio_raw_df 为空，终止预处理过程。")
        return
    
    processed = preprocess_data(portfolio_raw_df)
    processed = custom_preprocess(processed)
    filled = fill_missing(processed)
    normalized = normalize(filled)
    
    # 保存数据
    output_path = os.path.join(data_dir, 'df_portfolio.csv')
    save(normalized, output_path)
    
    logging.info("数据预处理实验完成。")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='运行指定的预处理实验。')
    parser.add_argument('--experiment', type=str, required=True, help='实验名称，对应 config/experiments/ 下的 Python 配置文件名称（不需要后缀）。')
    args = parser.parse_args()
    
    main(args.experiment)