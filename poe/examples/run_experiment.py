import os
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
from feature.experimenthandler import ExperimentHandler
from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import PortfolioOptimizationEnv
from finrl.agents.portfolio_optimization.architectures import EIIE, GPM
import torch

def main(experiment_name):
    # 加载配置
    config = load_config(experiment_name)
    
    # 设置日志
    setup_logger(config)
    logging.info(f"开始实验：{experiment_name}")
    
    # 加载数据
    df_train, df_test = load_data(config)
    unique_stocks = get_unique_stocks(df_train)
    logging.info(f"训练数据中的股票代码: {unique_stocks}")
    logging.info(f"测试数据中的股票代码: {df_test['tic'].unique().tolist()}")
    
    # 初始化参数
    training_params = config['TRAINING_PARAMS']
    time_window = training_params["time_window"]
    initial_features = training_params["initial_features"]
    features = training_params["features"]
    episodes = training_params["episodes"]
    policy_kwargs = training_params["policy_kwargs"]
    model_kwargs = training_params["model_kwargs"]
    
    policy_name = model_kwargs["policy"]
    model_dir = config['MODEL_DIR']
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(
        model_dir,
        f"policy_{policy_name}_{initial_features}_{time_window}.pt"
    )
    
    # 实例化 ExperimentHandler
    experiment_handler = ExperimentHandler(
        host=config['DATABASE']['host'],
        user=config['DATABASE']['user'],
        password=config['DATABASE']['password'],
        database=config['DATABASE']['database']
    )
    
    # 检查模型文件是否存在
    if os.path.exists(model_path):
        experiment_id = experiment_handler.get_experiment_id_by_model_path(model_path)
        if experiment_id:
            existing_experiment = experiment_handler.get_experiment_by_id(experiment_id)
            if existing_experiment:
                try:
                    parameters = json.loads(existing_experiment['parameters'])
                    existing_episodes = parameters.get('episodes', 0)
                    episodes += existing_episodes  # 更新 episodes
                    # 更新模型路径后缀
                    model_path = os.path.join(
                        model_dir,
                        f"policy_{policy_name}_{initial_features}_{time_window}_{experiment_id}.pt"
                    )
                    logging.info(f"找到现有实验 ID: {experiment_id}，更新后的 episodes: {episodes}")
                except json.JSONDecodeError as e:
                    logging.error(f"解析 parameters 时出错: {e}")
                    existing_experiment = None
            else:
                experiment_id = None  # 未找到现有实验
        else:
            experiment_id = None  # 未找到相关实验
    else:
        experiment_id = None  # 模型文件不存在
    
    # 创建训练环境
    env = PortfolioOptimizationEnv(df=df_train, **training_params)
    
    # 选择模型架构
    if policy_name == "EIIE":
        model = EIIE(**model_kwargs)
    elif policy_name == "GPM":
        model = GPM(**model_kwargs)
    else:
        logging.error(f"未知的策略名称: {policy_name}")
        return
    
    # 训练模型
    training_output = train_model(model, env, episodes, policy_name)
    
    # 验证模型
    env_test = PortfolioOptimizationEnv(df_test, **training_params)
    validation_output = validate_model(model, env_test, agent)
    
    # 解析验证输出
    test_metrics = {}  # 根据实际情况解析 validation_output
    # 应根据具体的验证输出格式实现解析逻辑，例如：
    # test_metrics = parse_metrics(validation_output, env, env_test, 'test')
    
    # 准备存储数据
    training_parameters = training_params.copy()
    
    # 转换 metrics 类型
    testing_metrics = convert_to_float({
        "training": {},  # 根据需要解析 training_output
        "test": test_metrics
    })
    
    # 格式化实验备注
    experiment_notes = config['EXPERIMENT_INFO']['notes'].format(
        num_assets=len(config['TRAINING_PARAMS']["portfolio_size"]), 
        num_features=len(features)
    )
    
    # 记录训练结束时间
    end_time = datetime.now()
    
    if experiment_id:
        # 更新现有实验数据
        experiment_handler.update_experiment(
            experiment_id=experiment_id,
            parameters=training_parameters,
            training_metrics=testing_metrics["training"],
            testing_metrics=testing_metrics["test"]
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
    
    if experiment_id and (not os.path.exists(model_path)):
        # 如果创建了新的 experiment_id，插入相关股票代码
        for stock_code in unique_stocks:
            experiment_handler.insert_stock_code(experiment_id, stock_code)
    
    # 保存模型
    torch.save(model.state_dict(), model_path)
    logging.info(f"模型已保存到 {model_path}")
    
    # 断开数据库连接
    experiment_handler.disconnect()
    
    print(f"Assets in training data: {df_train['tic'].unique()}")
    print(f"Assets in testing data: {df_test['tic'].unique()}")
    logging.info("实验结束。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='运行指定的实验。')
    parser.add_argument('--experiment', type=str, required=True, help='实验名称，对应 config/experiments/ 下的 YAML 配置文件名称。')
    args = parser.parse_args()
    
    main(args.experiment)