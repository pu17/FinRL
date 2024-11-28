import os
import sys
import pandas as pd
import logging
import json
from datetime import datetime
import argparse

from poe.config.load_config import load_config
from poe.utils.logger import setup_logger
from poe.utils.utils import convert_to_float
from poe.data.data_loader import load_data, get_unique_stocks
from poe.models.model_training import train_model
from poe.models.validation import validate_model, parse_metrics
from finrl.agents.portfolio_optimization.models import DRLAgent
from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import PortfolioOptimizationEnv
from finrl.agents.portfolio_optimization.architectures import EIIE, GPM
import torch

from poe.config.base_config import BASE_PATH

def create_experiment_directories(experiment_name,name):
    model_dir = os.path.join(BASE_PATH, 'models', experiment_name)
    log_dir = os.path.join(BASE_PATH, 'logs', experiment_name)
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    return model_dir, log_dir

def main(experiment_name):
    # 添加环境路径
    DB_PATH = '/Users/pu17/Documents/stock/stock_price_prediction'
    sys.path.append(DB_PATH)
    from feature.experimenthandler import ExperimentHandler
  # 加载配置
    config = load_config(experiment_name)

        # 初始化参数
    training_params = config['TRAINING_PARAMS']

    time_window = training_params["time_window"]
    data_params= config['data_params']
    features = config['FEATURES']
    initial_features = len(features)
    training_params["policy_kwargs"]["initial_features"] = initial_features
    episodes = training_params["episodes"]
    policy_kwargs = training_params["policy_kwargs"]
    model_kwargs = training_params["model_kwargs"]
    experiment_id = config['EXPERIMENT_INFO'].get('experiment_id')
    name = config['EXPERIMENT_INFO'].get('name')

  
    # 创建实验目录
    model_dir, log_dir = create_experiment_directories(experiment_name,name)
    # 设置日志
    config['LOG_FILE'] = os.path.join(log_dir, 'experiment.log')
    setup_logger(config)
    logging.info(f"开始实验：{experiment_name}")
    
    print(data_params['start_date'],data_params['end_date'],data_params['ticker_list'])
    # 加载数据
    df_train, df_test = load_data(
        data_params['data_file_path'],
        start_date=data_params['start_date'],
        end_date=data_params['end_date'],
        ticker_list=data_params['ticker_list']
    )
    if df_train.empty or df_test.empty:
        logging.error("训练数据或测试数据为空，终止实验。")
        return
    unique_stocks = get_unique_stocks(df_train)
    logging.info(f"训练数据中的股票代码: {unique_stocks}")
    logging.info(f"测试数据中的股票代码: {df_test['tic'].unique().tolist()}")
    

    
    policy_name = model_kwargs.get("policy")
    if not policy_name:
        logging.error("未指定策略名称（policy）。")
        return

    base_model_path = os.path.join(
        model_dir,
        f"policy_{policy_name}_{initial_features}_{time_window}_{experiment_id}.pt"
    )  

    # 设置设备
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'    

    # 选择模型架构
    if policy_name == "EIIE":
        policy = EIIE(time_window=time_window, initial_features=initial_features, device=device)
        model_kwargs['policy'] = EIIE
    elif policy_name == "GPM":
        policy = GPM(**model_kwargs)
    else:
        logging.error(f"未知的策略名称: {policy_name}")
        return

        # 创建训练环境
    env = PortfolioOptimizationEnv(
        df=df_train,
        initial_amount=training_params["initial_amount"],
        comission_fee_pct=training_params["comission_fee_pct"],
        time_window=training_params["time_window"],
        features=features,
        normalize_df=None
    )
    
    # 创建并训练模型
    model = DRLAgent(env).get_model("pg", device, model_kwargs, policy_kwargs)
    # 实例化 ExperimentHandler
    DB_PATH = '/Users/pu17/Documents/stock/stock_price_prediction'
    # 添加环境路径
    sys.path.append(DB_PATH)
    from feature.experimenthandler import ExperimentHandler

    # 实例化 ExperimentHandler
    experiment_handler = ExperimentHandler()
    
    # 检查模型文件是否存在
    if os.path.exists(base_model_path):
        experiment_id = experiment_handler.get_experiment_id_by_model_path(base_model_path)
        if experiment_id:
            existing_experiment = experiment_handler.get_experiment_by_id(experiment_id)
            if existing_experiment:
                try:
                    model.train_policy.load_state_dict(torch.load(base_model_path))

                    # 定义模型参数路径
                    policy_name = model_kwargs["policy"].__name__

                    # 训练模型
                    training_output = train_model(model, episodes, policy_name)

                    # 更新参数
                    parameters = json.loads(existing_experiment['parameters'])
                    existing_episodes = parameters.get('episodes', 0)
                    training_params["episodes"] += existing_episodes
                    logging.info(f"找到现有实验 ID: {experiment_id}，更新后的 episodes: {training_params['episodes']}")
                except json.JSONDecodeError as e:
                    logging.error(f"解析 parameters 时出错: {e}")
                    experiment_id = None
            else:
                experiment_id = None
                # 训练模型
                training_output = train_model(model, episodes, policy_name)
        else:
            experiment_id = None
            # 训练模型
            training_output = train_model(model, episodes, policy_name)
    else:
        experiment_id = None
        # 训练模型
        training_output = train_model(model, episodes, policy_name)


    
    # 保存模型路径
    model_path = base_model_path
    
    # 验证模型
    env_test = PortfolioOptimizationEnv(
        df=df_test,
        initial_amount=training_params["initial_amount"],
        comission_fee_pct=training_params["comission_fee_pct"],
        time_window=training_params["time_window"],
        features=features,
        normalize_df=None
    )
    validation_output = validate_model(model, env_test, policy)
    
    # 解析验证输出
    test_metrics = parse_metrics(validation_output, env, env_test, 'test')
    train_metrics = parse_metrics(training_output, env, env_test, 'training')
    
    # 准备存储数据
    training_parameters = training_params.copy()
    
    # 转换 metrics 类型
    testing_metrics = convert_to_float({
        "training": train_metrics,
        "test": test_metrics
    })
    
    # 更新实验备注
    experiment_notes = config['EXPERIMENT_INFO'].get('notes', '').format(
        num_assets=len(unique_stocks), 
        num_features=initial_features
    )
       # 选择模型架构
    if policy_name == "EIIE":
        policy = EIIE(time_window=time_window, initial_features=initial_features, device=device)
        model_kwargs['policy'] = "EIIE"
    elif policy_name == "GPM":
        policy = GPM(**model_kwargs)
    else:
        logging.error(f"未知的策略名称: {policy_name}")
        return
    
    # 记录训练结束时间
    end_time = datetime.now()
    
    if experiment_id:
        # 更新现有实验数据
        experiment_handler.update_experiment(
            experiment_id=experiment_id,
            parameters=training_parameters,
            training_metrics=testing_metrics["training"],
            testing_metrics=testing_metrics["test"],
            end_time=end_time
        )
        logging.info(f"实验 ID {experiment_id} 更新成功。")
    else:
        # 插入新的实验数据
        experiment_id = experiment_handler.insert_experiment_with_stocks(
            name=config['EXPERIMENT_INFO']['name'],
            notes=experiment_notes,
            parameters=training_parameters,
            training_metrics=testing_metrics["training"],
            testing_metrics=testing_metrics["test"],
            model_type="PortfolioOptimizationEnv",
            architecture_layers="128,64",
            architecture_activation="relu",
            start_time=datetime.now(),
            end_time=end_time,
            model_path=model_path
        )
        logging.info(f"插入新的实验数据，实验 ID: {experiment_id}")
    
    if experiment_id and (not os.path.exists(model_path)):
        # 如果创建了新的 experiment_id，插入相关股票代码
        for stock_code in unique_stocks:
            experiment_handler.insert_stock_code(experiment_id, stock_code)
    
    # 保存模型
    torch.save(model.train_policy.state_dict(), model_path)
    logging.info(f"模型已保存到 {model_path}")
    
    # 断开数据库连接
    experiment_handler.disconnect()
    
    print(f"Assets in training data: {df_train['tic'].unique()}")
    print(f"Assets in testing data: {df_test['tic'].unique()}")
    logging.info("实验结束。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='运行指定的实验。')
    parser.add_argument('--experiment', type=str, required=True, help='实验名称，对应 config/experiments/ 下的 Python 配置文件名称（不需要后缀）。')
    args = parser.parse_args()
    
    main(args.experiment)