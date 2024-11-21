import logging
from finrl.agents.portfolio_optimization.models import DRLAgent
from poe.utils.utils import capture_output

def train_model(model, environment, episodes, policy_name):
    """
    训练模型并返回训练输出。
    
    参数:
        model: 要训练的模型。
        environment: 训练环境。
        episodes (int): 训练轮数。
        policy_name (str): 策略名称。
    
    返回:
        str: 捕获的训练输出。
    """
    logging.info(f"开始训练模型: {policy_name}，轮数: {episodes}")
    training_output = capture_output(DRLAgent.train_model, model, env=environment, episodes=episodes)
    logging.info("模型训练完成。")
    return training_output