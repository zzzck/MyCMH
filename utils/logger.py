"""
日志记录模块
"""

import logging
import os
import time
from typing import Dict, Any, Optional
import json


def setup_logger(name: str,
                 log_file: str,
                 level: int = logging.INFO,
                 format_str: Optional[str] = None) -> logging.Logger:
    """
    设置日志记录器

    Args:
        name (str): 日志记录器名称
        log_file (str): 日志文件路径
        level (int): 日志级别
        format_str (str): 日志格式字符串

    Returns:
        logging.Logger: 日志记录器
    """
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # 创建日志目录
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 清除已有的处理器
    logger.handlers.clear()

    # 创建格式化器
    formatter = logging.Formatter(format_str)

    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def log_metrics(logger: logging.Logger,
                metrics: Dict[str, Any],
                step: Optional[int] = None,
                prefix: str = ""):
    """
    记录指标

    Args:
        logger (logging.Logger): 日志记录器
        metrics (Dict[str, Any]): 指标字典
        step (int): 步数
        prefix (str): 前缀
    """
    log_str = f"{prefix}" if prefix else ""

    if step is not None:
        log_str += f"Step {step}: "

    metric_strs = []
    for key, value in metrics.items():
        if isinstance(value, float):
            metric_strs.append(f"{key}: {value:.4f}")
        else:
            metric_strs.append(f"{key}: {value}")

    log_str += ", ".join(metric_strs)

    logger.info(log_str)


class MetricsLogger:
    """
    指标记录器
    """

    def __init__(self, log_dir: str, experiment_name: str = "experiment"):
        """
        初始化指标记录器

        Args:
            log_dir (str): 日志目录
            experiment_name (str): 实验名称
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name

        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)

        # 初始化记录
        self.metrics_history = []
        self.start_time = time.time()

        # 日志文件路径
        self.metrics_file = os.path.join(log_dir, f"{experiment_name}_metrics.json")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None, epoch: Optional[int] = None):
        """
        记录指标

        Args:
            metrics (Dict[str, Any]): 指标字典
            step (int): 步数
            epoch (int): 轮数
        """
        timestamp = time.time()
        elapsed_time = timestamp - self.start_time

        log_entry = {
            'timestamp': timestamp,
            'elapsed_time': elapsed_time,
            'metrics': metrics
        }

        if step is not None:
            log_entry['step'] = step

        if epoch is not None:
            log_entry['epoch'] = epoch

        self.metrics_history.append(log_entry)

        # 保存到文件
        self.save_metrics()

    def save_metrics(self):
        """
        保存指标到文件
        """
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

    def get_best_metric(self, metric_name: str, mode: str = 'max') -> Dict[str, Any]:
        """
        获取最佳指标

        Args:
            metric_name (str): 指标名称
            mode (str): 模式 ('max' 或 'min')

        Returns:
            Dict[str, Any]: 最佳指标记录
        """
        if not self.metrics_history:
            return {}

        valid_entries = [
            entry for entry in self.metrics_history
            if metric_name in entry['metrics']
        ]

        if not valid_entries:
            return {}

        if mode == 'max':
            best_entry = max(valid_entries, key=lambda x: x['metrics'][metric_name])
        else:
            best_entry = min(valid_entries, key=lambda x: x['metrics'][metric_name])

        return best_entry

    def get_metric_history(self, metric_name: str) -> list:
        """
        获取指标历史

        Args:
            metric_name (str): 指标名称

        Returns:
            list: 指标历史值
        """
        history = []
        for entry in self.metrics_history:
            if metric_name in entry['metrics']:
                history.append({
                    'step': entry.get('step'),
                    'epoch': entry.get('epoch'),
                    'value': entry['metrics'][metric_name],
                    'timestamp': entry['timestamp']
                })

        return history


class TrainingLogger:
    """
    训练日志记录器
    """

    def __init__(self, log_dir: str, experiment_name: str = "training"):
        """
        初始化训练日志记录器

        Args:
            log_dir (str): 日志目录
            experiment_name (str): 实验名称
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name

        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)

        # 设置主日志记录器
        log_file = os.path.join(log_dir, f"{experiment_name}.log")
        self.logger = setup_logger(experiment_name, log_file)

        # 指标记录器
        self.metrics_logger = MetricsLogger(log_dir, experiment_name)

        # 训练状态
        self.epoch_start_time = None
        self.training_start_time = time.time()

    def log_training_start(self, config: Dict[str, Any]):
        """
        记录训练开始

        Args:
            config (Dict[str, Any]): 训练配置
        """
        self.logger.info("=" * 60)
        self.logger.info("Training Started")
        self.logger.info("=" * 60)

        # 记录配置
        self.logger.info("Training Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")

        self.logger.info("=" * 60)

    def log_epoch_start(self, epoch: int, total_epochs: int):
        """
        记录epoch开始

        Args:
            epoch (int): 当前epoch
            total_epochs (int): 总epoch数
        """
        self.epoch_start_time = time.time()
        self.logger.info(f"Epoch {epoch + 1}/{total_epochs} started")

    def log_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        """
        记录epoch结束

        Args:
            epoch (int): 当前epoch
            metrics (Dict[str, Any]): epoch指标
        """
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            metrics['epoch_time'] = epoch_time

        # 记录指标
        self.metrics_logger.log(metrics, epoch=epoch)

        # 记录到日志
        self.logger.info(f"Epoch {epoch + 1} completed:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.6f}")
            else:
                self.logger.info(f"  {key}: {value}")

    def log_training_end(self, best_metrics: Dict[str, Any]):
        """
        记录训练结束

        Args:
            best_metrics (Dict[str, Any]): 最佳指标
        """
        total_time = time.time() - self.training_start_time

        self.logger.info("=" * 60)
        self.logger.info("Training Completed")
        self.logger.info(f"Total training time: {total_time:.2f}s")

        self.logger.info("Best metrics:")
        for key, value in best_metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.6f}")
            else:
                self.logger.info(f"  {key}: {value}")

        self.logger.info("=" * 60)

    def log_step(self, step: int, metrics: Dict[str, Any]):
        """
        记录训练步骤

        Args:
            step (int): 步数
            metrics (Dict[str, Any]): 步骤指标
        """
        self.metrics_logger.log(metrics, step=step)

    def log_validation(self, epoch: int, metrics: Dict[str, Any]):
        """
        记录验证结果

        Args:
            epoch (int): epoch
            metrics (Dict[str, Any]): 验证指标
        """
        self.logger.info(f"Validation results for epoch {epoch + 1}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.6f}")
            else:
                self.logger.info(f"  {key}: {value}")

    def log_evaluation(self, metrics: Dict[str, Any], title: str = "Evaluation"):
        """
        记录评估结果

        Args:
            metrics (Dict[str, Any]): 评估指标
            title (str): 标题
        """
        self.logger.info(f"{title} results:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.6f}")
            else:
                self.logger.info(f"  {key}: {value}")