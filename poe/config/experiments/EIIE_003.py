# config/experiments/EIIE.py

EXPERIMENT_INFO = {
    'experiment_id': 5,
    'name': 'EIIE 003',
    'notes': '训练 EIIE 模型进行投资组合优化，资产数量: {num_assets}，特征数量: {num_features}。'
}

data_params = {
    'start_date': '20161001',  # 数据收集的开始日期
    'end_date': None,          # 数据收集的结束日期，None 表示使用今天的日期
    'ticker_list': [           # 股票代码列表
        '603000.SS',
        '002261.SZ',
        '600547.SS',
        '600756.SS',
        '601899.SS',
        '601988.SS'
    ],
    'data_file_path': '/Users/pu17/Documents/stock/FinRL/poe/data/EIIE/df_portfolio.csv'  # 新增的数据文件路径
}

FEATURES = [
    'open',
    'high',
    'low',
    'close',
    'volume',
    'day',
    'macd',
    'boll_ub',
    'boll_lb',
    'rsi_30',
    'cci_30',
    'dx_30',
    'close_30_sma',
    'close_60_sma',
    'up_down_ratio',
    'market_breadth',
    'net_sm_amount',
    'net_md_amount',
    'net_lg_amount',
    'net_elg_amount',
    'net_sm_pct',
    'net_md_pct',
    'net_lg_pct',
    'net_elg_pct',
    'spring_festival_pre_holiday',
    'spring_festival_post_holiday',
    'labor_day_pre_holiday',
    'labor_day_post_holiday',
    'national_day_pre_holiday',
    'national_day_post_holiday',
    'dayofmonth',
    'dayofyear'
]

TRAINING_PARAMS = {
    'time_window': 10,
    'initial_amount': 100000,
    'comission_fee_pct': 0.0025,
    'model_kwargs': {
        'lr': 0.001,
        'policy': 'EIIE'
    },
    'policy_kwargs': {
        "k_size": 3,
        "time_window": 10,             # 使用变量形式
        "initial_features": len(FEATURES)       # 需要在代码中传入具体值
    },
    'episodes': 100,
    'portfolio_size': 10
}

LOG_FILE = 'experiment.log'
LOG_LEVEL = 'INFO'