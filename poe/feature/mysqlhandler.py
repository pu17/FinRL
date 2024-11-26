import os
import subprocess
import mysql.connector
from mysql.connector import Error
import pandas as pd
import logging
import json  
from config import MYSQL_CONFIG, LOCAL_MYSQL_CONFIG, OUTPUT_FILE, BACKUP_DIR
import csv
from datetime import datetime, timedelta

class MySQLHandler:
    def __init__(self):
        self.host = MYSQL_CONFIG['host']
        self.database = MYSQL_CONFIG['database']
        self.user = MYSQL_CONFIG['user']
        self.password = MYSQL_CONFIG['password']
        self.connection = None

        self.local_host = LOCAL_MYSQL_CONFIG['host']
        self.local_user = LOCAL_MYSQL_CONFIG['user']
        self.local_password = LOCAL_MYSQL_CONFIG['password']
        self.local_database = LOCAL_MYSQL_CONFIG['database']

        # 定义建表语句字典
        self.table_creation_queries = {
            'market_breadth_table': """
                CREATE TABLE IF NOT EXISTS market_breadth_table (
                    trade_date DATE,
                    up_count INT,
                    down_count INT,
                    up_down_ratio FLOAT,
                    market_breadth INT,
                    exchange VARCHAR(10),
                    PRIMARY KEY (trade_date, exchange)
                )
            """,
            'market_breadth_industry_table': """
                CREATE TABLE IF NOT EXISTS market_breadth_industry_table (
                    trade_date DATE,
                    up_count INT,
                    down_count INT,
                    up_down_ratio FLOAT,
                    market_breadth INT,
                    exchange VARCHAR(10),
                    l_code VARCHAR(20),
                    PRIMARY KEY (trade_date, exchange, l_code)
                )
            """,
            'moneyflow_summary_table': """
                CREATE TABLE IF NOT EXISTS moneyflow_summary_table (
                    trade_date DATE,
                    exchange VARCHAR(10),
                    buy_sm_vol BIGINT,
                    buy_sm_amount DECIMAL(20, 2),
                    sell_sm_vol BIGINT,
                    sell_sm_amount DECIMAL(20, 2),
                    buy_md_vol BIGINT,
                    buy_md_amount DECIMAL(20, 2),
                    sell_md_vol BIGINT,
                    sell_md_amount DECIMAL(20, 2),
                    buy_lg_vol BIGINT,
                    buy_lg_amount DECIMAL(20, 2),
                    sell_lg_vol BIGINT,
                    sell_lg_amount DECIMAL(20, 2),
                    buy_elg_vol BIGINT,
                    buy_elg_amount DECIMAL(20, 2),
                    sell_elg_vol BIGINT,
                    sell_elg_amount DECIMAL(20, 2),
                    net_mf_vol BIGINT,
                    net_mf_amount DECIMAL(20, 2),
                    net_sm_vol BIGINT,
                    net_sm_amount DECIMAL(20, 2),
                    net_md_vol BIGINT,
                    net_md_amount DECIMAL(20, 2),
                    net_lg_vol BIGINT,
                    net_lg_amount DECIMAL(20, 2),
                    net_elg_vol BIGINT,
                    net_elg_amount DECIMAL(20, 2),
                    net_sm_pct FLOAT,
                    net_md_pct FLOAT,
                    net_lg_pct FLOAT,
                    net_elg_pct FLOAT,
                    PRIMARY KEY (trade_date, exchange)
                )
            """,
            'holiday_features_table': """
                CREATE TABLE IF NOT EXISTS holiday_features_table (
                    trade_date DATE,
                    day INT,
                    spring_festival_pre_holiday INT,
                    spring_festival_post_holiday INT,
                    labor_day_pre_holiday INT,
                    labor_day_post_holiday INT,
                    national_day_pre_holiday INT,
                    national_day_post_holiday INT,
                    dayofmonth INT,
                    dayofyear INT,
                    PRIMARY KEY (trade_date)
                )
            """,
            'index_member_table': """ 
              CREATE TABLE IF NOT EXISTS index_member_table (
              trade_date DATE,
                l1_code VARCHAR(20),
                l1_name VARCHAR(50),
                l2_code VARCHAR(20),
                l2_name VARCHAR(50),
                l3_code VARCHAR(20),
                l3_name VARCHAR(50),
                ts_code VARCHAR(20),
                name VARCHAR(100),
                in_date DATE,
                PRIMARY KEY (ts_code)
                )
            """ ,
            'experiments': """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    start_time DATETIME,
                    end_time DATETIME,
                    notes TEXT,
                    
                    -- 训练参数（使用 JSON 格式存储）
                    parameters JSON NOT NULL,
                    
                    -- 训练集性能指标（使用 JSON 格式存储）
                    training_metrics JSON,
                    
                    -- 测试集性能指标（使用 JSON 格式存储）
                    testing_metrics JSON,
                    
                    -- 模型信息
                    model_type VARCHAR(100),
                    architecture_layers VARCHAR(255),       -- 例如 '128,64'
                    architecture_activation VARCHAR(50),    -- 例如 'relu'
                    
                    -- 模型路径
                    model_path VARCHAR(255)                 -- 新增的 model_path 列
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """,
            'experiment_stocks': """
                CREATE TABLE IF NOT EXISTS experiment_stocks (
                    experiment_id INT NOT NULL,
                    stock_code VARCHAR(20) NOT NULL,
                    PRIMARY KEY (experiment_id, stock_code),
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        }

        # 自动连接到数据库
        self.connect()

    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password
            )
            if self.connection.is_connected():
                logging.info("Successfully connected to MySQL database")
        except Error as e:
            logging.error(f"Error while connecting to MySQL: {e}")

    def disconnect(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logging.info("MySQL connection closed")

    def format_date(self, date_str):
        """将日期字符串格式化为'YYYY-MM-DD'"""
        try:
            if '-' in date_str:
                trade_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            else:
                trade_date = datetime.strptime(date_str, '%Y%m%d').date()
            return trade_date.strftime('%Y-%m-%d')
        except ValueError as e:
            logging.error(f"Error formatting date: {e}")
            return None

    def import_csv_to_mysql_fast(self, file_path, table_name):
        try:
            cursor = self.connection.cursor()
            with open(file_path, 'r') as csvfile:
                csvreader = csv.reader(csvfile)
                columns = next(csvreader)
                columns_str = ', '.join(columns)
                placeholders = ', '.join(['%s'] * len(columns))
                update_str = ', '.join([f"{col} = VALUES({col})" for col in columns if col not in ['trade_date', 'exchange']])
                insert_query = f"""
                INSERT INTO {table_name} ({columns_str})
                VALUES ({placeholders})
                ON DUPLICATE KEY UPDATE {update_str}
                """
                for row in csvreader:
                    formatted_date = self.format_date(row[0])
                    if formatted_date is None:
                        continue
                    converted_row = [formatted_date] + [
                        int(x) if x.isdigit() else float(x) if x.replace('.', '', 1).isdigit() else x
                        for x in row[1:]
                    ]
                    cursor.execute(insert_query, converted_row)
            self.connection.commit()
            logging.info(f"Data from {file_path} successfully imported/updated in table {table_name}")
        except Error as e:
            logging.error(f"Error importing data from CSV: {e}")
        finally:
            if cursor:
                cursor.close()

    def create_table(self, table_name):
        try:
            cursor = self.connection.cursor()
            create_table_query = self.table_creation_queries.get(table_name)
            if create_table_query:
                cursor.execute(create_table_query)
                self.connection.commit()
                logging.info(f"Table {table_name} created successfully")
            else:
                logging.error(f"No creation query found for table {table_name}")
        except Error as e:
            logging.error(f"Error creating table: {e}")
        finally:
            if cursor:
                cursor.close()

    def fetch_data(self, table_name, conditions=None, limit=None):
        try:
            query = f"SELECT * FROM {table_name}"
            if conditions:
                query += f" WHERE {conditions}"
            
            # 添加 ORDER BY 和 LIMIT 子句
            query += " ORDER BY trade_date DESC"  # 假设按 trade_date 排序
            if limit:
                query += f" LIMIT {limit}"
            
            df = pd.read_sql(query, self.connection)
            logging.info(f"Data successfully fetched from table {table_name}")
            return df
        except Error as e:
            logging.error(f"Error fetching data: {e}")
            return pd.DataFrame()

    def execute_query(self, query):
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            self.connection.commit()
            logging.info("Query executed successfully")
        except Error as e:
            logging.error(f"Error executing query: {e}")
        finally:
            if cursor:
                cursor.close()

    def insert_data(self, table_name, data):
        """将DataFrame数据插入到指定的表中"""
        try:
            cursor = self.connection.cursor()
            columns = data.columns.tolist()
            columns_str = ', '.join(columns)
            placeholders = ', '.join(['%s'] * len(columns))
            update_str = ', '.join([f"{col} = VALUES({col})" for col in columns if col not in ['trade_date', 'exchange', 'l_code']])
            insert_query = f"""
            INSERT INTO {table_name} ({columns_str})
            VALUES ({placeholders})
            ON DUPLICATE KEY UPDATE {update_str}
            """
            
            # 确保 trade_date 列是字符串格式
            data['trade_date'] = data['trade_date'].apply(lambda x: x.strftime('%Y-%m-%d') if isinstance(x, pd.Timestamp) else x)

            for _, row in data.iterrows():
                formatted_date = self.format_date(row['trade_date'])
                if formatted_date is None:
                    continue
                converted_row = [formatted_date] + [
                    int(x) if isinstance(x, int) else float(x) if isinstance(x, float) else x
                    for x in row[1:]
                ]
                cursor.execute(insert_query, converted_row)
            self.connection.commit()
            logging.info(f"Data successfully inserted into table {table_name}")
        except Error as e:
            logging.error(f"Error inserting data: {e}")
        finally:
            if cursor:
                cursor.close()

    def fetch_table_as_dataframe(self, table_name, start_date=None, end_date=None):
        """
        Fetches data from the specified table and returns it as a pandas DataFrame.
        
        :param table_name: The name of the table to fetch data from.
        :return: A pandas DataFrame containing the table data.
        """
        try:
            query = f"""
            SELECT * FROM {table_name}
            ORDER BY trade_date DESC
            """
            
            df = pd.read_sql(query, self.connection)
            
            # 计算数据框中的开始和结束日期
            if not df.empty:
                start_date = df['trade_date'].min().strftime('%Y-%m-%d')
                end_date = df['trade_date'].max().strftime('%Y-%m-%d')
            else:
                start_date = end_date = None
            
            logging.info(f"Data successfully fetched from table {table_name} between {start_date} and {end_date}")
            return df
        except Error as e:
            logging.error(f"Error fetching data from table {table_name}: {e}")
            return pd.DataFrame()

    def backup_and_restore(self):
        # 确保备份目录存在
        os.makedirs(BACKUP_DIR, exist_ok=True)

        # 生成备份文件名
        backup_file = os.path.join(BACKUP_DIR, f"{self.database}_backup_{datetime.now().strftime('%Y%m%d')}.sql")

        # 构建远程 mysqldump 命令
        dump_command = [
            "mysqldump",
            f"--host={self.host}",
            f"--user={self.user}",
            f"--password={self.password}",
            self.database,
            "--result-file", backup_file
        ]


        # 构建本地 mysql 导入命令
        restore_command = [
            "mysql",
            f"--host={self.local_host}",
            f"--user={self.local_user}",
            f"--password={self.local_password}",
            self.local_database,
            "--execute", f"source {backup_file}"
        ]

        try:
            # 执行备份命令
            subprocess.run(dump_command, check=True)
            logging.info(f"Database backup successful: {backup_file}")

            # 执行恢复命令
            subprocess.run(restore_command, check=True)
            logging.info(f"Database restored successfully to local database: {self.local_database}")

            # 清理旧的备份文件，只保留最近3个日期的备份
            self.cleanup_old_backups()

        except subprocess.CalledProcessError as e:
            logging.error(f"Error during database backup or restore: {e}")

    def cleanup_old_backups(self):
        # 获取备份目录中的所有文件
        files = os.listdir(BACKUP_DIR)
        # 过滤出符合备份文件命名格式的文件
        backup_files = [f for f in files if f.startswith(f"{self.database}_backup_") and f.endswith(".sql")]

        # 提取日期并排序
        backup_files.sort(key=lambda x: x.split('_')[-1].split('.')[0], reverse=True)

        # 保留最近3个备份文件
        for old_backup in backup_files[3:]:
            os.remove(os.path.join(BACKUP_DIR, old_backup))
            logging.info(f"Removed old backup file: {old_backup}")

    def drop_table(self, table_name):
        """
        Drops the specified table from the database.
        
        :param table_name: The name of the table to drop.
        """
        try:
            cursor = self.connection.cursor()
            drop_table_query = f"DROP TABLE IF EXISTS {table_name}"
            cursor.execute(drop_table_query)
            self.connection.commit()
            logging.info(f"Table {table_name} dropped successfully")
        except Error as e:
            logging.error(f"Error dropping table {table_name}: {e}")
        finally:
            if cursor:
                cursor.close()

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 创建 MySQLHandler 实例
    mysql_handler = MySQLHandler()

    try:
        if mysql_handler.connection and mysql_handler.connection.is_connected():
            logging.info("Successfully connected to the database.")
            
            # 示例：删除表
            table_name = "holiday_features_table"
            mysql_handler.drop_table(table_name)

            # # 创建表
            mysql_handler.create_table(table_name)

            # # 导入 CSV 数据
            # mysql_handler.import_csv_to_mysql_fast(OUTPUT_FILE, table_name)
            # logging.info("CSV import completed.")

            # # 获取并显示数据的前几行
            # fetched_data = mysql_handler.fetch_data(table_name)
            # logging.info(f"Fetched data (first few rows):\n{fetched_data.head()}")

        else:
            logging.error("Failed to connect to the database.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

    finally:
        # 断开连接
        mysql_handler.disconnect()








