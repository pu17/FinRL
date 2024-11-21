import yaml
import os

def load_config(experiment_name):
    """
    加载基础配置并合并实验特定配置。

    参数:
        experiment_name (str): 实验名称，对应 config/experiments/ 下的 YAML 配置文件名称。

    返回:
        dict: 合并后的配置字典。
    """
    base_config_path = os.path.join(os.path.dirname(__file__), 'base_config.yaml')
    experiment_config_path = os.path.join(os.path.dirname(__file__), 'experiments', f'{experiment_name}.yaml')
    
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    if os.path.exists(experiment_config_path):
        with open(experiment_config_path, 'r') as f:
            experiment_config = yaml.safe_load(f)
        # 合并基础配置和实验特定配置
        for key, value in experiment_config.items():
            if isinstance(value, dict) and key in base_config:
                base_config[key].update(value)
            else:
                base_config[key] = value
    else:
        raise FileNotFoundError(f"实验配置文件 {experiment_config_path} 不存在。")
    
    return base_config