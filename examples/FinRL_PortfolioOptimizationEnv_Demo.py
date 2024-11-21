# portfolio_optimization.py
import logging
logging.getLogger('matplotlib.font_manager').disabled = True

import torch
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import MaxAbsScaler
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import GroupByScaler
from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import PortfolioOptimizationEnv
from finrl.agents.portfolio_optimization.models import DRLAgent
from finrl.agents.portfolio_optimization.architectures import EIIE, GPM
import mysql.connector
from mysql.connector import Error

import io
import sys
import re
from datetime import datetime
import os

# 定义捕获输出的函数
def capture_output(func, *args, **kwargs):
    """
    捕获函数执行期间的标准输出。

    参数:
        func (callable): 要执行的函数。
        *args: 传递给函数的非关键字参数。
        **kwargs: 传递给函数的关键字参数。

    返回:
        str: 被捕获的输出内容。
    """
    captured_output = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = captured_output

    try:
        func(*args, **kwargs)
    finally:
        sys.stdout = sys_stdout

    return captured_output.getvalue()

def parse_validation_output(output):
    """
    解析验证输出内容并返回一个包含所有指标块的列表。

    参数:
        output (str): 验证输出的字符串。

    返回:
        list of dict: 包含所有解析后的指标块。
    """
    # 使用分隔符分割输出
    blocks = output.split('=================================')
    # 移除空白块并去除前后空白字符
    blocks = [block.strip() for block in blocks if block.strip()]
    metrics_list = []
    
    for block in blocks:
        print("解析的块内容：", block)  # 打印每个块的内容
        metrics = {}
        try:
            metrics["initial_portfolio_value"] = float(re.search(r"Initial portfolio value:([\d\.]+)", block).group(1))
            metrics["final_portfolio_value"] = float(re.search(r"Final portfolio value:\s*([\d\.]+)", block).group(1))
            metrics["final_accumulative_portfolio_value"] = float(re.search(r"Final accumulative portfolio value:\s*([\d\.]+)", block).group(1))
            metrics["max_drawdown"] = float(re.search(r"Maximum DrawDown:\s*(-?[\d\.]+)", block).group(1))
            metrics["sharpe_ratio"] = float(re.search(r"Sharpe ratio:\s*(-?[\d\.]+)", block).group(1))
            metrics_list.append(metrics)
        except AttributeError as e:
            logging.error(f"解析输出时出错: {e}")
    return metrics_list

# 设置设备
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 数据收集逻辑
TOP_BRL = [
    '000001.SS', '399001.SZ', '603000.SS', '000035.SZ',
    '002261.SZ', '000938.SZ', '600547.SS', '600756.SS',
    '601899.SS', '601988.SS'
]

num_assets = len(TOP_BRL)  # 资产数量

# 读取CSV文件
df_portfolio = pd.read_csv('/Users/pu17/Documents/stock/FinRL/examples/df_portfolio.csv')

# 提取训练数据
df_portfolio_train = df_portfolio[
    (df_portfolio["date"] >= "2018-01-01") & (df_portfolio["date"] < "2024-03-01")
]
df_portfolio_2024 = df_portfolio[
    (df_portfolio["date"] >= "2024-03-01") & (df_portfolio["date"] <= "2024-11-18")
]

# 定义特征列表
features = [
    'open', 'high', 'low', 'close', 'volume', 'day', 'macd', 'boll_ub',
    'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma',
    'up_down_ratio', 'market_breadth', 'net_sm_amount', 'net_md_amount',
    'net_lg_amount', 'net_elg_amount', 'net_sm_pct', 'net_md_pct',
    'net_lg_pct', 'net_elg_pct', 'spring_festival_pre_holiday',
    'spring_festival_post_holiday', 'labor_day_pre_holiday',
    'labor_day_post_holiday', 'national_day_pre_holiday',
    'national_day_post_holiday', 'dayofmonth', 'dayofyear'
]

# 定义参数
time_window = 10
initial_features = len(features)

# 创建 PortfolioOptimizationEnv 环境
environment = PortfolioOptimizationEnv(
    df_portfolio_train,
    initial_amount=100000,
    comission_fee_pct=0.0025,
    time_window=time_window,
    features=features,
    normalize_df=None
)

# 设置 PolicyGradient 参数
model_kwargs = {
    "lr": 0.001,
    "policy": EIIE,
}

# 设置 EIIE 的参数
policy_kwargs = {
    "k_size": 3,
    "time_window": time_window,        # 使用变量形式
    "initial_features": initial_features,  # 使用变量形式
}

# 创建并训练模型
model = DRLAgent(environment).get_model("pg", device, model_kwargs, policy_kwargs)

# 定义模型参数路径
policy_name = model_kwargs["policy"].__name__
# 初始 experiment_id 为 None
experiment_id = None
model_path = f"/Users/pu17/Documents/stock/FinRL/examples/models/policy_{policy_name}_{initial_features}_{time_window}_1.pt"

# 定义训练轮数
episodes = 10

# 初始化 testing_metrics 字典，移除 UBAH 相关部分
testing_metrics = {
    "training": {},
    "test": {}
}

DB_PATH = '/Users/pu17/Documents/stock/stock_price_prediction'
# 添加环境路径
sys.path.append(DB_PATH)
from feature.experimenthandler import ExperimentHandler

# 实例化 ExperimentHandler
experiment_handler = ExperimentHandler()



# 检查数据库连接
if experiment_handler.connection and experiment_handler.connection.is_connected():
    logging.info("成功连接到数据库。")

    # 记录训练开始时间
    start_time = datetime.now()

    # 捕捉并解析训练阶段的输出
    training_output = capture_output(DRLAgent.train_model, model, episodes=episodes)
    parsed_training_metrics_list = parse_validation_output(training_output)

    if parsed_training_metrics_list:
        parsed_training_metrics = parsed_training_metrics_list[-1]
        parsed_training_metrics["value"] = environment._asset_memory["final"]
        testing_metrics["training"] = parsed_training_metrics
        logging.info("成功解析训练阶段的指标。")
        logging.info(f"训练阶段的指标：{parsed_training_metrics}")
    else:
        logging.error("未能解析训练阶段的指标。")

    # 检查模型文件是否存在
    if os.path.exists(model_path):
        experiment_id = experiment_handler.get_experiment_id_by_model_path(model_path)
        if experiment_id:
            existing_experiment = experiment_handler.get_experiment_by_id(experiment_id)
            if existing_experiment:
                # 从 parameters 中提取 episodes
                parameters = json.loads(existing_experiment['parameters'])
                existing_episodes = parameters.get('episodes', 0)
                episodes += existing_episodes  # 更新 episodes
                # 更新模型路径后缀
                model_path = f"/Users/pu17/Documents/stock/FinRL/examples/models/policy_{policy_name}_{initial_features}_{time_window}_{experiment_id}.pt"
    else:
        experiment_id = None  # 如果模型不存在，则创建新实验

    # 保存模型参数
    torch.save(model.train_policy.state_dict(), model_path)

    # 创建测试环境并加载策略
    environment_test = PortfolioOptimizationEnv(
        df_portfolio_2024,
        initial_amount=100000,
        comission_fee_pct=0.0025,
        time_window=time_window,
        features=features,
        normalize_df=None
    )
    policy = EIIE(time_window=time_window, initial_features=len(features), device=device)

    # 捕捉并解析测试阶段的输出
    validation_output_2024 = capture_output(DRLAgent.DRL_validation, model, environment_test, policy=policy)
    parsed_test_metrics_list = parse_validation_output(validation_output_2024)

    if parsed_test_metrics_list:
        parsed_test_metrics = parsed_test_metrics_list[-1]
        parsed_test_metrics["value"] = environment_test._asset_memory["final"]
        testing_metrics["test"] = parsed_test_metrics
        logging.info("成功解析测试阶段的指标。")
        logging.info(f"测试阶段的指标：{parsed_test_metrics}")
    else:
        logging.error("未能解析测试阶段的指标。")

    # 定义 training_parameters 字典
    training_parameters = {
        "time_window": time_window,
        "initial_features": initial_features,
        "features": features,
        "initial_amount": 100000,
        "comission_fee_pct": 0.0025,
        "model_kwargs": {
            "lr": model_kwargs["lr"],
            "policy": model_kwargs["policy"].__name__,  # 使用 __name__ 获取类名
        },
        "policy_kwargs": policy_kwargs,
        "episodes": episodes,  # 将 episodes 作为 training_parameters 的一部分
        "portfolio_size": len(TOP_BRL)
    }

    # 在打印最终的 testing_metrics 字典之前，确保所有的 float32 类型被转换为 float
    def convert_to_float(obj):
        if isinstance(obj, dict):
            return {k: convert_to_float(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_float(i) for i in obj]
        elif isinstance(obj, np.float32):
            return float(obj)
        else:
            return obj

    # 转换 testing_metrics 中的所有 float32 类型
    testing_metrics = convert_to_float(testing_metrics)

    # 打印最终的 testing_metrics 字典
    print("最终的测试指标结构：")
    print(json.dumps(testing_metrics, indent=4, ensure_ascii=False))

    # 定义实验名称和备注
    experiment_name = "EIIE_Portfolio_Optimization"
    experiment_notes = f"训练 EIIE 模型进行投资组合优化，资产数量: {num_assets}，特征数量: {len(features)}。"

    experiment_handler.drop_table('experiments')
    experiment_handler.create_table('experiments')
    # 记录训练结束时间
    end_time = datetime.now()

    if experiment_id:
        # 可能需要从 existing_experiment 中提取所有信息，但这里假设要更新 parameters
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
            name=experiment_name,
            notes=experiment_notes,
            parameters=training_parameters,
            training_metrics=testing_metrics["training"],
            testing_metrics=testing_metrics["test"],
            model_type="PortfolioOptimizationEnv",
            architecture_layers="128,64",
            architecture_activation="relu",
            start_time=start_time,
            end_time=end_time,
            model_path=model_path
        )
    print(f"Experiment ID: {experiment_id}")
    print(f"Model Path Exists: {os.path.exists(model_path)}")

    if experiment_id and (not os.path.exists(model_path)):
        print("111")
        experiment_handler.create_table('experiment_stocks')
        unique_stocks = df_portfolio_train['tic'].unique()
        for stock_code in unique_stocks:
            experiment_handler.insert_stock_code(experiment_id, stock_code)

else:
    logging.error("无法连接到数据库。")

# 断开数据库连接
experiment_handler.disconnect()

print(f"Assets in training data: {df_portfolio_train['tic'].unique()}")
print(f"Assets in testing data: {df_portfolio_2024['tic'].unique()}")