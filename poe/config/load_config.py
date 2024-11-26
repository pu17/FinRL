# config/load_config.py

import importlib.util
import os

def load_config(experiment_name):
    config_dir = os.path.join(os.path.dirname(__file__), 'experiments')
    config_path = os.path.join(config_dir, f"{experiment_name}.py")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到：{config_path}")
    
    spec = importlib.util.spec_from_file_location(experiment_name, config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    config = {attr: getattr(config_module, attr) for attr in dir(config_module) if not attr.startswith("__")}
    
    return config