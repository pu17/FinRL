import os
import glob
import subprocess

def main():
    config_dir = os.path.join(os.path.dirname(__file__), '../config/experiments')
    config_files = glob.glob(os.path.join(config_dir, '*.yaml'))
    
    for config_file in config_files:
        experiment_name = os.path.splitext(os.path.basename(config_file))[0]
        print(f"开始运行实验：{experiment_name}")
        
        # 运行数据预处理
        preprocess_cmd = [
            'python', 'preprocess_data.py',
            '--experiment', experiment_name
        ]
        subprocess.run(preprocess_cmd, check=True)
        
        # 运行模型训练
        train_cmd = [
            'python', 'run_experiment.py',
            '--experiment', experiment_name
        ]
        subprocess.run(train_cmd, check=True)
        
        print(f"实验 {experiment_name} 运行完成。")

if __name__ == "__main__":
    main()