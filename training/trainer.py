"""
跨模态哈希检索训练器
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import os
import time
import logging
from typing import Dict, List, Optional, Tuple
import wandb
from tqdm import tqdm

from models.cross_modal_hash import CrossModalHashModel, CrossModalHashLoss
from evaluation.evaluator import CrossModalEvaluator
from .optimizer import create_optimizer, create_scheduler
from .config import TrainingConfig


class CrossModalHashTrainer:
    """
    跨模态哈希检索训练器
    """

    def __init__(self,
                 config: TrainingConfig,
                 model: CrossModalHashModel,
                 train_dataloader: DataLoader,
                 val_dataloader: Optional[DataLoader] = None,
                 test_dataloader: Optional[DataLoader] = None):
        """
        初始化训练器

        Args:
            config (TrainingConfig): 训练配置
            model (CrossModalHashModel): 跨模态哈希模型
            train_dataloader (DataLoader): 训练数据加载器
            val_dataloader (DataLoader): 验证数据加载器
            test_dataloader (DataLoader): 测试数据加载器
        """
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        # 设备配置
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # 损失函数
        self.criterion = CrossModalHashLoss(
            margin=config.margin,
            lambda_quant=config.lambda_quant,
            lambda_balance=config.lambda_balance,
            temperature=config.contrastive_temperature
        )

        # 优化器和调度器
        self.optimizer = create_optimizer(self.model, config)
        self.scheduler = create_scheduler(
            self.optimizer, config, len(train_dataloader)
        )

        # 混合精度训练
        self.scaler = GradScaler() if config.mixed_precision else None

        # 评估器
        self.evaluator = CrossModalEvaluator(
            k_values=config.eval_k_values,
            device=self.device
        )

        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_map = 0.0
        self.best_epoch = 0

        # 日志设置
        self.setup_logging()

        # 创建保存目录
        os.makedirs(config.save_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

        # 保存配置
        config.save(os.path.join(config.save_dir, 'config.json'))

    def setup_logging(self):
        """
        设置日志
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def train_epoch(self) -> Dict[str, float]:
        """
        训练一个epoch

        Returns:
            Dict[str, float]: 训练指标
        """
        self.model.train()

        total_loss = 0.0
        loss_components = {
            'feature_contrastive': 0.0,
            'hash_contrastive': 0.0,
            'quantization': 0.0,
            'balance': 0.0
        }
        num_batches = len(self.train_dataloader)

        progress_bar = tqdm(self.train_dataloader, desc=f'Epoch {self.current_epoch}')

        for batch_idx, batch in enumerate(progress_bar):
            # 移动数据到设备
            images = batch['images'].to(self.device)
            text_inputs = {k: v.to(self.device) for k, v in batch['text_inputs'].items()}
            labels = batch['labels'].to(self.device)

            # 前向传播
            if self.config.mixed_precision:
                with autocast():
                    outputs = self.model(text_inputs=text_inputs, images=images)
                    losses = self.criterion(outputs['text'], outputs['image'], labels)
            else:
                outputs = self.model(text_inputs=text_inputs, images=images)
                losses = self.criterion(outputs['text'], outputs['image'], labels)

            total_batch_loss = losses['total']

            # 反向传播
            self.optimizer.zero_grad()

            if self.config.mixed_precision:
                self.scaler.scale(total_batch_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_batch_loss.backward()
                self.optimizer.step()

            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step()

            # 累计损失
            total_loss += total_batch_loss.item()
            for key in loss_components:
                if key in losses:
                    loss_components[key] += losses[key].item()

            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{total_batch_loss.item():.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })

            # 日志记录
            if (batch_idx + 1) % self.config.log_interval == 0:
                self.logger.info(
                    f'Epoch {self.current_epoch}, Batch {batch_idx + 1}/{num_batches}, '
                    f'Loss: {total_batch_loss.item():.4f}, '
                    f'LR: {self.optimizer.param_groups[0]["lr"]:.2e}'
                )

            self.global_step += 1

        # 计算平均损失
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches

        metrics = {
            'train_loss': avg_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        metrics.update({f'train_{k}': v for k, v in loss_components.items()})

        return metrics

    def validate(self) -> Dict[str, float]:
        """
        验证模型

        Returns:
            Dict[str, float]: 验证指标
        """
        if self.val_dataloader is None:
            return {}

        self.model.eval()

        total_loss = 0.0
        loss_components = {
            'feature_contrastive': 0.0,
            'hash_contrastive': 0.0,
            'quantization': 0.0,
            'balance': 0.0
        }
        num_batches = len(self.val_dataloader)

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc='Validation'):
                # 移动数据到设备
                images = batch['images'].to(self.device)
                text_inputs = {k: v.to(self.device) for k, v in batch['text_inputs'].items()}
                labels = batch['labels'].to(self.device)

                # 前向传播
                if self.config.mixed_precision:
                    with autocast():
                        outputs = self.model(text_inputs=text_inputs, images=images)
                        losses = self.criterion(outputs['text'], outputs['image'], labels)
                else:
                    outputs = self.model(text_inputs=text_inputs, images=images)
                    losses = self.criterion(outputs['text'], outputs['image'], labels)

                # 累计损失
                total_loss += losses['total'].item()
                for key in loss_components:
                    if key in losses:
                        loss_components[key] += losses[key].item()

        # 计算平均损失
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches

        metrics = {'val_loss': avg_loss}
        metrics.update({f'val_{k}': v for k, v in loss_components.items()})

        return metrics

    def evaluate(self) -> Dict[str, float]:
        """
        评估模型检索性能

        Returns:
            Dict[str, float]: 评估指标
        """
        if self.test_dataloader is None:
            return {}

        self.logger.info("Starting model evaluation...")

        # 使用评估器评估模型
        results = self.evaluator.evaluate_model(self.model, self.test_dataloader)

        # 提取主要指标
        main_metrics = {}
        for metric in ['hamming_Avg_mAP', 'hamming_T2I_mAP', 'hamming_I2T_mAP']:
            if metric in results:
                main_metrics[metric] = results[metric]

        for k in [1, 5, 10]:
            for direction in ['T2I', 'I2T', 'Avg']:
                metric_name = f'hamming_{direction}_Precision@{k}'
                if metric_name in results:
                    main_metrics[metric_name] = results[metric_name]

        return main_metrics

    def save_checkpoint(self, is_best=False, filename=None):
        """
        保存检查点

        Args:
            is_best (bool): 是否为最佳模型
            filename (str): 文件名
        """
        if filename is None:
            filename = f'checkpoint_epoch_{self.current_epoch}.pth'

        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_map': self.best_map,
            'best_epoch': self.best_epoch,
            'config': self.config.to_dict()
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # 保存检查点
        checkpoint_path = os.path.join(self.config.save_dir, filename)
        torch.save(checkpoint, checkpoint_path)

        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.config.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f'Best model saved to {best_path}')

        self.logger.info(f'Checkpoint saved to {checkpoint_path}')

    def load_checkpoint(self, checkpoint_path):
        """
        加载检查点

        Args:
            checkpoint_path (str): 检查点路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_map = checkpoint.get('best_map', 0.0)
        self.best_epoch = checkpoint.get('best_epoch', 0)

        self.logger.info(f'Checkpoint loaded from {checkpoint_path}')

    def train(self):
        """
        完整训练流程
        """
        self.logger.info("Starting training...")
        self.logger.info(f"Training for {self.config.num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Mixed precision: {self.config.mixed_precision}")

        # 从检查点恢复（如果指定）
        if self.config.resume_from:
            self.load_checkpoint(self.config.resume_from)

        start_time = time.time()

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch

            # 训练一个epoch
            train_metrics = self.train_epoch()

            # 验证
            val_metrics = self.validate()

            # 评估（默认每个epoch执行，可按配置降频）
            eval_metrics = {}
            should_eval = self.config.evaluate_every_epoch or ((epoch + 1) % self.config.eval_interval == 0)
            if should_eval:
                eval_metrics = self.evaluate()

                # 检查是否为最佳模型
                current_map = eval_metrics.get('hamming_Avg_mAP', 0.0)
                is_best = current_map > self.best_map
                if is_best:
                    self.best_map = current_map
                    self.best_epoch = epoch
                    self.logger.info(f'New best mAP: {self.best_map:.4f} at epoch {epoch}')

            # 合并所有指标
            all_metrics = {**train_metrics, **val_metrics, **eval_metrics}

            # 日志记录
            self.logger.info(f'Epoch {epoch} completed:')
            for key, value in all_metrics.items():
                self.logger.info(f'  {key}: {value:.4f}')

            # 保存检查点
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint()

            # 保存最佳模型
            if eval_metrics and eval_metrics.get('hamming_Avg_mAP', 0.0) == self.best_map:
                self.save_checkpoint(is_best=True)

        # 训练完成
        total_time = time.time() - start_time
        self.logger.info(f'Training completed in {total_time:.2f}s')
        self.logger.info(f'Best mAP: {self.best_map:.4f} at epoch {self.best_epoch}')

        # 最终评估
        if self.test_dataloader:
            self.logger.info("Final evaluation on test set...")
            final_results = self.evaluate()
            self.evaluator.print_results(final_results, "Final Test Results")

            # 保存最终结果
            results_path = os.path.join(self.config.save_dir, 'final_results.json')
            self.evaluator.save_results(final_results, results_path)


def create_trainer(config: TrainingConfig,
                   model: CrossModalHashModel,
                   train_dataloader: DataLoader,
                   val_dataloader: Optional[DataLoader] = None,
                   test_dataloader: Optional[DataLoader] = None) -> CrossModalHashTrainer:
    """
    创建训练器的便捷函数

    Args:
        config (TrainingConfig): 训练配置
        model (CrossModalHashModel): 模型
        train_dataloader (DataLoader): 训练数据加载器
        val_dataloader (DataLoader): 验证数据加载器
        test_dataloader (DataLoader): 测试数据加载器

    Returns:
        CrossModalHashTrainer: 训练器
    """
    return CrossModalHashTrainer(
        config=config,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader
    )