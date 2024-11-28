import os
import glob
import subprocess

def main():
    config_dir = os.path.join(os.path.dirname(__file__), '../config/experiments')
    config_files = glob.glob(os.path.join(config_dir, '*.py'))
    print(f"配置文件目录: {config_dir}")
    
    if not config_files:
        print("没有找到任何配置文件。请检查路径和文件格式。")
        return
    
    for config_file in config_files:
        experiment_name = os.path.splitext(os.path.basename(config_file))[0]
        print(f"开始运行实验：{experiment_name}")
        
        try:
            # 运行数据预处理
            # preprocess_cmd = [
            #     'python', 'preprocess_data.py',
            #     '--experiment', experiment_name
            # ]
            # subprocess.run(preprocess_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # 运行模型训练
            train_cmd = [
                '/Users/pu17/miniconda3/envs/finrobot/bin/python', 'poe/examples/run_experiment.py',
                '--experiment', experiment_name
            ]
            subprocess.run(train_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            print(f"实验 {experiment_name} 运行完成。")
        except subprocess.CalledProcessError as e:
            print(f"运行实验 {experiment_name} 时出错：{e}")
            print(f"标准输出：{e.stdout.decode()}")
            print(f"标准错误：{e.stderr.decode()}")

if __name__ == "__main__":
    main()