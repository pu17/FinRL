import logging
from logging.handlers import RotatingFileHandler

def setup_logger(config):
    """
    设置日志配置。

    参数:
        config (dict): 配置字典。
    """
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, config['LOG_LEVEL'].upper(), logging.INFO))
    
    # 创建旋转日志处理器
    handler = RotatingFileHandler(config['LOG_FILE'], maxBytes=10**7, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    # 禁用matplotlib的字体管理器日志
    logging.getLogger('matplotlib.font_manager').disabled = True