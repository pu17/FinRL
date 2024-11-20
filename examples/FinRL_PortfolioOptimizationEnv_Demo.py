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
from finrl.agents.portfolio_optimization.architectures import EIIE
import mysql.connector
from mysql.connector import Error

import io
import sys
import re

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
model_path = "/Users/pu17/Documents/stock/FinRL/examples/policy_EIIE_34_10.pt"

# 加载保存的模型参数（如果有）
model.train_policy.load_state_dict(torch.load(model_path))

# 定义训练轮数
episodes = 5

# 初始化 testing_metrics 字典，按照新的结构组织
testing_metrics = {
    "model": {
        "training": {},
        "test": {}
    },
    "UBAH": {
        "training": {},
        "test": {}
    }
}

# 定义实验名称和备注
experiment_name = "EIIE_Portfolio_Optimization"
experiment_notes = f"训练 EIIE 模型进行投资组合优化，资产数量: {num_assets}，特征数量: {len(features)}。"

DB_PATH = '/Users/pu17/Documents/stock/stock_price_prediction'
# 添加环境路径
import sys
sys.path.append(DB_PATH)
from feature.mysqlhandler import MySQLHandler  # 假设 MySQLHandler 类保存在 mysql_handler.py 中
# 实例化 MySQLHandler
mysql_handler = MySQLHandler()

# 检查数据库连接
if mysql_handler.connection and mysql_handler.connection.is_connected():
    logging.info("成功连接到数据库。")

    # 捕捉并解析训练阶段的输出
    training_output = capture_output(DRLAgent.train_model, model, episodes=episodes)
    parsed_training_metrics_list = parse_validation_output(training_output)
    
    if parsed_training_metrics_list:
        parsed_training_metrics = parsed_training_metrics_list[-1]
        testing_metrics["model"]["training"] = parsed_training_metrics
        logging.info("成功解析训练阶段的指标。")
        logging.info(f"训练阶段的指标：{parsed_training_metrics}")
    else:
        logging.error("未能解析训练阶段的指标。")
    
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
    policy.load_state_dict(torch.load(model_path))
    
    # 捕捉并解析测试阶段的输出
    validation_output_2024 = capture_output(DRLAgent.DRL_validation, model, environment_test, policy=policy)
    parsed_test_metrics_list = parse_validation_output(validation_output_2024)
    
    if parsed_test_metrics_list:
        parsed_test_metrics = parsed_test_metrics_list[-1]
        testing_metrics["model"]["test"] = parsed_test_metrics
        logging.info("成功解析测试阶段的指标。")
        logging.info(f"测试阶段的指标：{parsed_test_metrics}")
    else:
        logging.error("未能解析测试阶段的指标。")
    
    # 定义 UBAH 策略的运行函数
    PORTFOLIO_SIZE = len(TOP_BRL)-3

    def run_UBAH(environment, portfolio_size):
        """
        运行 Uniform Buy and Hold (UBAH) 策略并收集性能指标。

        参数:
            environment (PortfolioOptimizationEnv): 投资组合优化环境。
            portfolio_size (int): 资产数量。

        返回:
            dict: 包含 UBAH 策略的性能指标。
        """
        terminated = False
        environment.reset()
        while not terminated:
            # 定义 UBAH 策略动作：持有所有资产的等权重
            action = [0] + [1 / portfolio_size] * portfolio_size
            _, _, terminated, _ = environment.step(action)
        # 捕捉 calculate_metrics 的输出
        metrics_output = capture_output(environment.calculate_metrics)
        # 解析指标
        parsed_metrics_list = parse_validation_output(metrics_output)
        if parsed_metrics_list:
            return parsed_metrics_list[-1]  # 获取最后一个指标块
        else:
            logging.error("未能解析 UBAH 策略的指标。")
            return {}

    # 运行并解析 UBAH 策略的训练阶段指标
    # UBAH_results_training = run_UBAH(environment, PORTFOLIO_SIZE)
    # testing_metrics["UBAH"]["training"] = UBAH_results_training
    # logging.info("成功解析 UBAH 策略训练阶段的指标。")
    # logging.info(f"UBAH 训练阶段的指标：{UBAH_results_training}")

    # # 运行并解析 UBAH 策略的测试阶段指标
    # UBAH_results_test = run_UBAH(environment_test, PORTFOLIO_SIZE)
    # testing_metrics["UBAH"]["test"] = UBAH_results_test
    # logging.info("成功解析 UBAH 策略测试阶段的指标。")
    # logging.info(f"UBAH 测试阶段的指标：{UBAH_results_test}")

    # # 打印最终的 testing_metrics 字典
    # print("最终的测试指标结构：")
    # print(json.dumps(testing_metrics, indent=4, ensure_ascii=False))

#     # 插入完整的实验数据
#     experiment_id = mysql_handler.insert_experiment_with_stocks(
#         name=experiment_name,
#         notes=experiment_notes,
#         parameters=training_parameters,
#         training_metrics=testing_metrics["model"]["training"],
#         testing_metrics={
#             "test": testing_metrics["model"]["test"],
#             "UBAH": testing_metrics["UBAH"]
#         },
#         stock_codes=TOP_BRL,
#         model_type="PolicyGradient",
#         architecture_layers="128,64",
#         architecture_activation="relu"
#     )

#     if experiment_id:
#         # 进行后续操作，如模型验证等
#         logging.info("所有实验数据已成功存储。")
#     else:
#         logging.error("实验数据插入失败。")

# else:
#     logging.error("无法连接到数据库。")

# 断开数据库连接
# mysql_handler.disconnect()

print(f"Assets in training data: {df_portfolio_train['tic'].unique()}")
print(f"Assets in testing data: {df_portfolio_2024['tic'].unique()}")