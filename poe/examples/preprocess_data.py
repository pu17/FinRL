import logging
from poe.data.data_loader import load_data
from poe.utils.logger import setup_logger
from poe.config.load_config import load_config
from feature.ch_feature_engineer import ChFeatureEngineer
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.preprocessor.tusharedownloader import TushareDownloader
from finrl.meta.preprocessor.preprocessors import GroupByScaler
from sklearn.preprocessing import MaxAbsScaler
from datetime import datetime

def main(experiment_name):
    # 加载配置
    config = load_config(experiment_name)
    
    # 设置日志
    setup_logger(config)
    logging.info(f"开始数据预处理实验：{experiment_name}")
    
    # 加载数据
    df_train, df_test = load_data()
    
    # 数据预处理
    from finrl.config import INDICATORS
    
    def preprocess(portfolio_raw_df):
        logging.info("开始基础数据预处理。")
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=INDICATORS,
            use_vix=False,
            use_turbulence=False,
            user_defined_feature=False
        )
        processed = fe.preprocess_data(portfolio_raw_df)
        logging.info("基础数据预处理完成。")
        return processed

    def custom_preprocess(processed):
        logging.info("开始自定义特征工程。")
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

    def save(portfolio_norm):
        output_path = config['DATA_PATH']
        portfolio_norm.to_csv(output_path, index=False)
        logging.info(f"数据已保存至 {output_path}")
    
    # 运行预处理流程
    processed = preprocess(df_train)
    processed = custom_preprocess(processed)
    filled = fill_missing(processed)
    normalized = normalize(filled)
    save(normalized)
    
    logging.info("数据预处理实验完成。")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='运行指定的预处理实验。')
    parser.add_argument('--experiment', type=str, required=True, help='实验名称，对应 config/experiments/ 下的 YAML 配置文件名称。')
    args = parser.parse_args()
    
    main(args.experiment)