import mysql.connector
from mysql.connector import Error
import json
import logging
from feature.mysqlhandler import MySQLHandler

class ExperimentHandler(MySQLHandler):
    def __init__(self, host, user, password, database):
        super().__init__(host, user, password, database)

    def get_experiment_id_by_model_path(self, model_path):
        try:
            cursor = self.connection.cursor(dictionary=True)
            query = "SELECT experiment_id FROM experiments WHERE model_path = %s"
            cursor.execute(query, (model_path,))
            result = cursor.fetchone()
            return result['experiment_id'] if result else None
        except Error as e:
            logging.error(f"Error fetching experiment_id: {e}")
            return None
        finally:
            if cursor:
                cursor.close()

    def get_experiment_by_id(self, experiment_id):
        try:
            cursor = self.connection.cursor(dictionary=True)
            query = "SELECT * FROM experiments WHERE experiment_id = %s"
            cursor.execute(query, (experiment_id,))
            return cursor.fetchone()
        except Error as e:
            logging.error(f"Error fetching experiment by id: {e}")
            return None
        finally:
            if cursor:
                cursor.close()

    def update_experiment(self, experiment_id, parameters, training_metrics, testing_metrics):
        """
        更新实验记录。
        
        参数:
            experiment_id (int): 要更新的实验的唯一标识符。
            parameters (dict): 实验参数。
            training_metrics (dict): 训练集性能指标。
            testing_metrics (dict): 测试集性能指标。
        """
        try:
            cursor = self.connection.cursor()
            query = """
                UPDATE experiments
                SET parameters = %s, training_metrics = %s, testing_metrics = %s
                WHERE experiment_id = %s
            """
            parameters_json = json.dumps(parameters)
            training_metrics_json = json.dumps(training_metrics)
            testing_metrics_json = json.dumps(testing_metrics)
            cursor.execute(query, (parameters_json, training_metrics_json, testing_metrics_json, experiment_id))
            self.connection.commit()
            logging.info(f"成功更新实验 ID: {experiment_id}")
        except Error as e:
            logging.error(f"Error updating experiment: {e}")
        finally:
            if cursor:
                cursor.close()

    def insert_experiment_with_stocks(self, name, notes, parameters, training_metrics, testing_metrics, model_type, architecture_layers, architecture_activation, start_time, end_time, model_path):
        """
        一次性插入实验数据，包括相关股票代码。
        
        参数:
            name (str): 实验名称。
            notes (str): 实验备注。
            parameters (dict): 实验参数。
            training_metrics (dict): 训练集性能指标。
            testing_metrics (dict): 测试集性能指标。
            model_type (str): 模型类型。
            architecture_layers (str): 架构层信息。
            architecture_activation (str): 激活函数信息。
            start_time (datetime): 训练开始时间。
            end_time (datetime): 训练结束时间。
            model_path (str): 模型路径。
        
        返回:
            experiment_id (int): 插入的实验 ID。
        """
        if not self.connection or not self.connection.is_connected():
            logging.error("数据库连接未建立。无法插入实验数据。")
            return None

        try:
            cursor = self.connection.cursor()
            insert_query = """
                INSERT INTO experiments (name, notes, parameters, training_metrics, testing_metrics, model_type, architecture_layers, architecture_activation, start_time, end_time, model_path)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            parameters_json = json.dumps(parameters)
            training_metrics_json = json.dumps(training_metrics)
            testing_metrics_json = json.dumps(testing_metrics)
            cursor.execute(insert_query, (
                name,
                notes,
                parameters_json,
                training_metrics_json,
                testing_metrics_json,
                model_type,
                architecture_layers,
                architecture_activation,
                start_time,
                end_time,
                model_path
            ))
            self.connection.commit()
            experiment_id = cursor.lastrowid
            logging.info(f"成功插入实验数据，实验 ID: {experiment_id}")

            return experiment_id
        except Error as e:
            logging.error(f"插入实验数据失败: {e}")
            return None
        finally:
            if cursor:
                cursor.close()

    def insert_stock_code(self, experiment_id, stock_code):
        """
        插入股票代码到 experiment_stocks 表中。
        
        参数:
            experiment_id (int): 实验 ID。
            stock_code (str): 股票代码。
        """
        try:
            cursor = self.connection.cursor()
            insert_query = """
                INSERT INTO experiment_stocks (experiment_id, stock_code)
                VALUES (%s, %s)
            """
            cursor.execute(insert_query, (experiment_id, stock_code))
            self.connection.commit()
            logging.info(f"成功插入股票代码 {stock_code} 到 experiment_id {experiment_id} 中。")
        except Error as e:
            logging.error(f"插入股票代码时出错: {e}")
        finally:
            if cursor:
                cursor.close()