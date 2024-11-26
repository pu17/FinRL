import io
import sys
import json
import logging
import numpy as np
import re

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
    blocks = output.split('=================================')
    blocks = [block.strip() for block in blocks if block.strip()]
    metrics_list = []

    for block in blocks:
        logging.debug(f"解析的块内容：{block}")
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

def convert_to_float(obj):
    """
    将对象中的 float32 类型转换为 float。
    
    参数:
        obj: 需要转换的对象。
    
    返回:
        转换后的对象。
    """
    if isinstance(obj, dict):
        return {k: convert_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_float(i) for i in obj]
    elif isinstance(obj, np.float32):
        return float(obj)
    else:
        return obj