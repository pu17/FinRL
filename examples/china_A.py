import os
from stable_baselines3 import A2C, DDPG, PPO, TD3, SAC
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
import pandas as pd
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.config import INDICATORS
import datetime
import itertools
import pytz

# 设置参数
if_using_a2c = True
if_using_ddpg = True
if_using_ppo = True
if_using_td3 = True
if_using_sac = True
TRAINED_MODEL_DIR = '/Users/pu17/Documents/stock/FinRL/trained_models'

# 设置文件路径常量
MARKET_BREADTH_FILE_PATH = '/Users/pu17/Documents/stock/stock_price_prediction/data/processed/market_breadth_all_exchanges.csv'
PROCESSED_DATA_DIR = '/Users/pu17/Documents/stock/FinRL/processed_data'

# 1. 加载训练好的模型
def load_trained_models():
    trained_models = {}
    if if_using_a2c:
        trained_models['a2c'] = A2C.load(os.path.join(TRAINED_MODEL_DIR, "agent_a2c"))
    if if_using_ddpg:
        trained_models['ddpg'] = DDPG.load(os.path.join(TRAINED_MODEL_DIR, "agent_ddpg"))
    if if_using_ppo:
        trained_models['ppo'] = PPO.load(os.path.join(TRAINED_MODEL_DIR, "agent_ppo"))
    if if_using_td3:
        trained_models['td3'] = TD3.load(os.path.join(TRAINED_MODEL_DIR, "agent_td3"))
    if if_using_sac:
        trained_models['sac'] = SAC.load(os.path.join(TRAINED_MODEL_DIR, "agent_sac"))
    return trained_models

# 2. 准备新的交易数据
def prepare_trade_data(start_date, end_date):
    print(f"尝试下载数据，开始日期: {start_date}, 结束日期: {end_date}")
    
    # 下载数据
    sz_df_finrl = YahooDownloader(
        start_date=start_date,
        end_date=end_date,
        ticker_list=['000001.SS', '399001.SZ', '603000.SS']
    ).fetch_data()

    print(f"下载的数据日期范围: {sz_df_finrl['date'].min()} 到 {sz_df_finrl['date'].max()}")
    print(f"下载的数据形状: {sz_df_finrl.shape}")

    # 预处理数据
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=True,
        use_turbulence=True,
        user_defined_feature=False
    )

    processed = fe.preprocess_data(sz_df_finrl)

    # 读取市场宽度数据
    market_breadth_data = pd.read_csv(MARKET_BREADTH_FILE_PATH)
    market_breadth_data['date'] = pd.to_datetime(market_breadth_data['trade_date']).dt.strftime('%Y-%m-%d')

    # 选择交易所
    def select_exchange(row):
        if row['tic'] == '399001.SZ':
            return 'SZSE'
        elif row['tic'] in ['000001.SS', '603000.SS']:
            return 'SSE'
        return None

    processed['exchange'] = processed.apply(select_exchange, axis=1)

    # 合并数据
    merged_df = pd.merge(processed, market_breadth_data, how='left', on=['date', 'exchange'])
    merged_df.drop(columns=['trade_date', 'exchange'], inplace=True)

    # 在合并市场宽度数据后打印信息
    print(f"合并市场宽度数据后的日期范围: {merged_df['date'].min()} 到 {merged_df['date'].max()}")
    print(f"合并市场宽度数据后的数据形状: {merged_df.shape}")

    # 填充缺失数据
    list_ticker = merged_df["tic"].unique().tolist()
    list_date = list(pd.date_range(merged_df['date'].min(), merged_df['date'].max()).astype(str))
    combination = list(itertools.product(list_date, list_ticker))

    processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(merged_df, on=["date", "tic"], how="left")
    processed_full = processed_full[processed_full['date'].isin(merged_df['date'])]
    processed_full = processed_full.sort_values(['date', 'tic'])
    
    # 在填充前后打印信息
    print(f"填充前的日期范围: {processed_full['date'].min()} 到 {processed_full['date'].max()}")
    print(f"填充前的数据形状: {processed_full.shape}")
    
    processed_full = processed_full.fillna(method='ffill').fillna(0)
    
    print(f"填充后的日期范围: {processed_full['date'].min()} 到 {processed_full['date'].max()}")
    print(f"填充后的数据形状: {processed_full.shape}")

    # 保存处理后的数据
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    processed_full.to_csv(os.path.join(PROCESSED_DATA_DIR, 'latest_trade_data.csv'), index=False)
    print(f"处理后的数据已保存到 {os.path.join(PROCESSED_DATA_DIR, 'latest_trade_data.csv')}")

    return processed_full

# 3. 使用模型进行预测
def predict_with_models(models, trade_data):
    print(f"trade_data shape: {trade_data.shape}")
    print(f"trade_data columns: {trade_data.columns}")
    print(f"Sample of 'close' column: {trade_data['close'].head()}")

    # 确保数据按日期和股票代码排序
    trade_data = trade_data.sort_values(['date', 'tic'])

    # 设置环境参数
    stock_dimension = len(trade_data.tic.unique())
    state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
    
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }

    # 创建环境
    e_trade_gym = StockTradingEnv(df=trade_data, turbulence_threshold=70, risk_indicator_col='vix', **env_kwargs)

    results = {}
    for model_name, model in models.items():
        print(f"使用 {model_name.upper()} 模型进行预测...")
        try:
            df_account_value, df_actions = DRLAgent.DRL_prediction(model=model, environment=e_trade_gym)
            
            # 处理动作数据
            df_actions['date'] = trade_data['date'].unique()
            df_actions = df_actions.melt(id_vars=['date'], var_name='tic', value_name='action')
            df_actions = df_actions.sort_values(['date', 'tic'])
            
            results[model_name] = {
                "account_value": df_account_value, 
                "actions": df_actions
            }
        except Exception as e:
            print(f"{model_name.upper()} 模型预测失败: {str(e)}")
            continue
    
    return results

# 4. 分析预测结果 (这部分保持不变)
def analyze_results(results):
    # 实现与之前相同
    pass

# 主函数
def main():
    # 获取当前日期（考虑时区）
    china_tz = pytz.timezone('Asia/Shanghai')
    current_date = datetime.datetime.now(china_tz).date()
    
    # 设置结束日期为今天
    end_date = current_date.strftime("%Y-%m-%d")
    
    # 设置开始日期为500天前
    start_date = (current_date - datetime.timedelta(days=500)).strftime("%Y-%m-%d")

    print(f"当前日期: {current_date}")
    print(f"设置的开始日期: {start_date}")
    print(f"设置的结束日期: {end_date}")

    # 加载模型
    loaded_models = load_trained_models()

    # 准备新的交易数据
    new_trade_data = prepare_trade_data(start_date, end_date)

    print(f"处理后的数据日期范围: {new_trade_data['date'].min()} 到 {new_trade_data['date'].max()}")
    print(f"处理后的数据形状: {new_trade_data.shape}")

    # 使用模型进行预测
    # prediction_results = predict_with_models(loaded_models, new_trade_data)

    # # # 分析预测结果
    # # analyze_results(prediction_results)

    # # 输出每个模型的动作
    # for model_name, result in prediction_results.items():
    #     print(f"\n{model_name.upper()} 模型的预测动作:")
    #     actions_df = result['actions']
        
    #     # 获取最新的日期
    #     latest_date = actions_df['date'].max()
    #     latest_actions = actions_df[actions_df['date'] == latest_date]
        
    #     print(f"最新日期 {latest_date} 的动作:")
    #     for _, row in latest_actions.iterrows():
    #         action = row['action']
    #         if action > 0:
    #             action_type = "买入"
    #         elif action < 0:
    #             action_type = "卖出"
    #         else:
    #             action_type = "持有"
    #         print(f"股票 {row['tic']}: {action_type} {abs(action)} 股")
        
    #     # 可选：保存完整的动作数据到 CSV 文件
    #     actions_df.to_csv(f"{model_name}_actions.csv", index=False)
    #     print(f"完整的动作数据已保存到 {model_name}_actions.csv")

if __name__ == "__main__":
    main()
