# config/experiments/gpm.py

TRAINING_PARAMS = {
    'episodes': 200,
    'model_kwargs': {
        'lr': 0.0005,
        'policy': 'GPM'
    }
}

EXPERIMENT_INFO = {
    'name': 'GPM_Portfolio_Optimization',
    'notes': '训练 GPM 模型进行投资组合优化，资产数量: {num_assets}，特征数量: {num_features}。'
}

MODEL_DIR = '/Users/pu17/Documents/stock/FinRL/examples/models/GPM_Portfolio_Optimization/'