import logging
from finrl.agents.portfolio_optimization.models import DRLAgent
from poe.utils.utils import capture_output, parse_validation_output

def validate_model(model, environment_test, policy):
    """
    验证模型并返回测试输出。
    
    参数:
        model: 要验证的模型。
        environment_test: 测试环境。
        policy: 策略实例。
    
    返回:
        str: 捕获的测试输出。
    """
    logging.info("开始验证模型。")
    validation_output = capture_output(DRLAgent.DRL_validation, model, environment_test, policy=policy)
    logging.info("模型验证完成。")
    return validation_output

def parse_metrics(output, environment, environment_test, metric_type):
    """
    解析训练或测试输出并返回最新的指标。
    
    参数:
        output (str): 捕获的输出内容。
        environment: 当前环境（训练或测试环境）。
        environment_test: 测试环境。
        metric_type (str): 指标类型，'training' 或 'test'。
    
    返回:
        dict: 最新的指标。
    """
    metrics_list = parse_validation_output(output)
    if metrics_list:
        metrics = metrics_list[-1]
        if metric_type == "training":
            metrics["value"] = environment._asset_memory["final"]
        elif metric_type == "test":
            metrics["value"] = environment_test._asset_memory["final"]
        return metrics
    else:
        logging.error(f"未能解析{metric_type}阶段的指标。")
        return {}