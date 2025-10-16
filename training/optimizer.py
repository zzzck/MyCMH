"""
优化器和学习率调度器模块
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    MultiStepLR,
    LinearLR,
    SequentialLR
)
import math


def create_optimizer(model, config):
    """
    创建优化器

    Args:
        model: 模型
        config: 训练配置

    Returns:
        torch.optim.Optimizer: 优化器
    """
    # 分组参数（可以为不同模块设置不同的学习率）
    param_groups = []

    # 文本编码器参数
    text_encoder_params = []
    for name, param in model.text_encoder.named_parameters():
        if param.requires_grad:
            text_encoder_params.append(param)

    if text_encoder_params:
        param_groups.append({
            'params': text_encoder_params,
            'lr': config.learning_rate * 0.1,  # 预训练模型使用较小学习率
            'name': 'text_encoder'
        })

    # 图像编码器参数
    image_encoder_params = []
    for name, param in model.image_encoder.named_parameters():
        if param.requires_grad:
            image_encoder_params.append(param)

    if image_encoder_params:
        param_groups.append({
            'params': image_encoder_params,
            'lr': config.learning_rate * 0.1,  # 预训练模型使用较小学习率
            'name': 'image_encoder'
        })

    # 哈希层和投影层参数
    other_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and not any(name.startswith(prefix) for prefix in ['text_encoder', 'image_encoder']):
            other_params.append(param)

    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': config.learning_rate,
            'name': 'hash_layers'
        })

    # 如果没有分组参数，使用所有参数
    if not param_groups:
        param_groups = [{'params': model.parameters(), 'lr': config.learning_rate}]

    # 创建优化器
    if config.optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(
            param_groups,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(
            param_groups,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(
            param_groups,
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=True
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {config.optimizer_type}")

    return optimizer


def create_scheduler(optimizer, config, steps_per_epoch):
    """
    创建学习率调度器

    Args:
        optimizer: 优化器
        config: 训练配置
        steps_per_epoch (int): 每个epoch的步数

    Returns:
        torch.optim.lr_scheduler: 学习率调度器
    """
    total_steps = config.num_epochs * steps_per_epoch
    warmup_steps = config.warmup_epochs * steps_per_epoch

    if config.scheduler_type.lower() == 'cosine':
        # 余弦退火调度器
        if warmup_steps > 0:
            # 带预热的余弦退火
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps
            )

            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=config.learning_rate * 0.01
            )

            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )
        else:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=config.learning_rate * 0.01
            )

    elif config.scheduler_type.lower() == 'step':
        # 阶梯式调度器
        scheduler = StepLR(
            optimizer,
            step_size=config.num_epochs // 3,
            gamma=0.1
        )

    elif config.scheduler_type.lower() == 'multistep':
        # 多阶梯调度器
        milestones = [config.num_epochs // 3, 2 * config.num_epochs // 3]
        scheduler = MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=0.1
        )

    elif config.scheduler_type.lower() == 'linear':
        # 线性调度器
        scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=total_steps
        )

    else:
        # 不使用调度器
        scheduler = None

    return scheduler


class WarmupCosineAnnealingLR:
    """
    自定义预热余弦退火学习率调度器
    """

    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=0, last_epoch=-1):
        """
        初始化调度器

        Args:
            optimizer: 优化器
            warmup_epochs (int): 预热轮数
            max_epochs (int): 总轮数
            eta_min (float): 最小学习率
            last_epoch (int): 上一轮数
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        self.last_epoch = last_epoch

        # 保存初始学习率
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        self.step(last_epoch + 1)

    def get_lr(self):
        """
        计算当前学习率

        Returns:
            list: 各参数组的学习率
        """
        if self.last_epoch < self.warmup_epochs:
            # 预热阶段：线性增长
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # 余弦退火阶段
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [
                self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        """
        更新学习率

        Args:
            epoch (int): 当前轮数
        """
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class GradualWarmupScheduler:
    """
    渐进式预热调度器
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        """
        初始化渐进式预热调度器

        Args:
            optimizer: 优化器
            multiplier (float): 学习率倍数
            total_epoch (int): 预热总轮数
            after_scheduler: 预热后的调度器
        """
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.optimizer = optimizer

        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self, epoch=None):
        """
        更新学习率

        Args:
            epoch (int): 当前轮数
        """
        if epoch is None:
            epoch = self.last_epoch + 1

        if epoch <= self.total_epoch:
            # 预热阶段
            lr_multiplier = (self.multiplier - 1.) * epoch / self.total_epoch + 1.
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * lr_multiplier
        else:
            # 预热结束，使用后续调度器
            if not self.finished:
                self.finished = True
                # 重置基础学习率
                for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                    param_group['lr'] = base_lr * self.multiplier

            if self.after_scheduler:
                self.after_scheduler.step(epoch - self.total_epoch)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """
    获取带预热的余弦调度器

    Args:
        optimizer: 优化器
        num_warmup_steps (int): 预热步数
        num_training_steps (int): 总训练步数
        num_cycles (float): 余弦周期数
        last_epoch (int): 上一步数

    Returns:
        LambdaLR: Lambda学习率调度器
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)