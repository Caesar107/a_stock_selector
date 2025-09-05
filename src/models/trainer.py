# -*- coding: utf-8 -*-
"""
模型训练器
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from .transformer_model import StockTransformer, StockTransformerAdvanced, MultiTaskLoss, FocalLoss

logger = logging.getLogger(__name__)


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 模型配置
        self.model_config = config.get('model', {})
        self.training_config = config.get('training', {})
        
        # 训练参数
        self.batch_size = self.training_config.get('batch_size', 32)
        self.learning_rate = self.training_config.get('learning_rate', 1e-4)
        self.num_epochs = self.training_config.get('num_epochs', 100)
        self.early_stopping_patience = self.training_config.get('early_stopping_patience', 10)
        
        # 模型和优化器
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
    def build_model(self, model_type: str = 'transformer') -> nn.Module:
        """构建模型"""
        try:
            if model_type == 'transformer':
                model = StockTransformer(self.model_config)
            elif model_type == 'advanced_transformer':
                model = StockTransformerAdvanced(self.model_config)
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            model = model.to(self.device)
            
            # 计算模型参数数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            logger.info(f"模型构建完成:")
            logger.info(f"  总参数数: {total_params:,}")
            logger.info(f"  可训练参数数: {trainable_params:,}")
            
            return model
            
        except Exception as e:
            logger.error(f"构建模型失败: {e}")
            raise
    
    def setup_training(self, model: nn.Module):
        """设置训练组件"""
        self.model = model
        
        # 优化器
        optimizer_type = self.training_config.get('optimizer', 'adamw')
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(), 
                lr=self.learning_rate,
                weight_decay=self.training_config.get('weight_decay', 1e-4)
            )
        elif optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                model.parameters(), 
                lr=self.learning_rate,
                weight_decay=self.training_config.get('weight_decay', 1e-4)
            )
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                model.parameters(), 
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.training_config.get('weight_decay', 1e-4)
            )
        
        # 学习率调度器
        scheduler_type = self.training_config.get('scheduler', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.num_epochs
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=20, 
                gamma=0.5
            )
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                patience=5, 
                factor=0.5
            )
        
        # 损失函数
        loss_type = self.training_config.get('loss', 'multitask')
        if loss_type == 'multitask':
            self.criterion = MultiTaskLoss(
                alpha=self.training_config.get('alpha', 0.7),
                beta=self.training_config.get('beta', 0.3)
            )
        elif loss_type == 'focal':
            self.criterion = FocalLoss(
                alpha=self.training_config.get('focal_alpha', 1),
                gamma=self.training_config.get('focal_gamma', 2)
            )
        elif loss_type == 'crossentropy':
            self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"训练设置完成:")
        logger.info(f"  优化器: {optimizer_type}")
        logger.info(f"  学习率调度器: {scheduler_type}")
        logger.info(f"  损失函数: {loss_type}")
    
    def prepare_data_loaders(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> Tuple[DataLoader, DataLoader]:
        """准备数据加载器"""
        try:
            # 转换为PyTorch张量
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.LongTensor(y_train)
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.LongTensor(y_val)
            
            # 创建数据集
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            
            # 创建数据加载器
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True,
                num_workers=0,  # Windows上设置为0避免问题
                pin_memory=True if self.device.type == 'cuda' else False
            )
            
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.batch_size, 
                shuffle=False,
                num_workers=0,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            
            logger.info(f"数据加载器准备完成:")
            logger.info(f"  训练集: {len(train_dataset)} 样本, {len(train_loader)} 批次")
            logger.info(f"  验证集: {len(val_dataset)} 样本, {len(val_loader)} 批次")
            
            return train_loader, val_loader
            
        except Exception as e:
            logger.error(f"准备数据加载器失败: {e}")
            raise
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_class_loss = 0.0
        total_reg_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            if isinstance(self.model, StockTransformerAdvanced):
                outputs = self.model(data)
                # 多输出模型的损失计算
                class_output = outputs['limit_up']
                loss = self.criterion(class_output, target)
            else:
                class_output, reg_output = self.model(data)
                
                if isinstance(self.criterion, MultiTaskLoss):
                    # 生成回归目标（这里简化为分类目标的浮点版本）
                    reg_target = target.float()
                    loss, class_loss, reg_loss = self.criterion(
                        class_output, reg_output, target, reg_target
                    )
                    total_class_loss += class_loss.item()
                    total_reg_loss += reg_loss.item()
                else:
                    loss = self.criterion(class_output, target)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.training_config.get('grad_clip', None):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.training_config['grad_clip']
                )
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_samples += target.size(0)
            
            # 计算准确率
            _, predicted = torch.max(class_output, 1)
            correct_predictions += (predicted == target).sum().item()
            
            if batch_idx % 100 == 0:
                logger.debug(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        metrics = {
            'loss': total_loss / len(train_loader),
            'accuracy': correct_predictions / total_samples
        }
        
        if total_class_loss > 0:
            metrics['class_loss'] = total_class_loss / len(train_loader)
            metrics['reg_loss'] = total_reg_loss / len(train_loader)
        
        return metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # 前向传播
                if isinstance(self.model, StockTransformerAdvanced):
                    outputs = self.model(data)
                    class_output = outputs['limit_up']
                    loss = self.criterion(class_output, target)
                else:
                    class_output, reg_output = self.model(data)
                    
                    if isinstance(self.criterion, MultiTaskLoss):
                        reg_target = target.float()
                        loss, _, _ = self.criterion(
                            class_output, reg_output, target, reg_target
                        )
                    else:
                        loss = self.criterion(class_output, target)
                
                total_loss += loss.item()
                total_samples += target.size(0)
                
                # 预测
                _, predicted = torch.max(class_output, 1)
                correct_predictions += (predicted == target).sum().item()
                
                # 保存用于详细评估
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # 计算详细指标
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted', zero_division=0
        )
        
        metrics = {
            'loss': total_loss / len(val_loader),
            'accuracy': correct_predictions / total_samples,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics, all_predictions, all_targets
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray, 
              model_type: str = 'transformer') -> nn.Module:
        """主训练函数"""
        logger.info("开始模型训练...")
        
        # 构建模型
        model = self.build_model(model_type)
        self.setup_training(model)
        
        # 准备数据
        train_loader, val_loader = self.prepare_data_loaders(X_train, y_train, X_val, y_val)
        
        # 训练历史
        train_history = []
        val_history = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}")
            
            # 训练
            train_metrics = self.train_epoch(train_loader)
            
            # 验证
            val_metrics, val_predictions, val_targets = self.validate_epoch(val_loader)
            
            # 更新学习率
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['loss'])
            else:
                self.scheduler.step()
            
            # 记录历史
            train_history.append(train_metrics)
            val_history.append(val_metrics)
            
            # 日志
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                       f"F1: {val_metrics['f1']:.4f}")
            
            # 早停
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                # 保存最佳模型
                self.save_checkpoint(model, epoch, val_metrics, 'best_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= self.early_stopping_patience:
                logger.info(f"早停触发，在epoch {epoch+1}")
                break
        
        # 保存训练历史
        self._save_training_history(train_history, val_history)
        
        # 生成评估报告
        self._generate_evaluation_report(val_predictions, val_targets)
        
        logger.info("模型训练完成")
        return model
    
    def predict(self, model: nn.Module, X: np.ndarray) -> np.ndarray:
        """预测"""
        model.eval()
        
        # 转换为张量
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch = X_tensor[i:i+self.batch_size]
                
                if isinstance(model, StockTransformerAdvanced):
                    outputs = model(batch)
                    class_output = outputs['limit_up']
                else:
                    class_output, _ = model(batch)
                
                # 获取概率和预测
                probs = torch.softmax(class_output, dim=1)
                _, preds = torch.max(class_output, 1)
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(predictions), np.array(probabilities)
    
    def select_top_stocks(self, predictions: Tuple[np.ndarray, np.ndarray], 
                         stock_codes: List[str], top_n: int = 10) -> List[Dict]:
        """选择top股票"""
        preds, probs = predictions
        
        # 获取涨停概率（假设类别1是涨停）
        limit_up_probs = probs[:, 1]
        
        # 排序并选择top N
        sorted_indices = np.argsort(limit_up_probs)[::-1][:top_n]
        
        selected_stocks = []
        for idx in sorted_indices:
            selected_stocks.append({
                'stock_code': stock_codes[idx],
                'prediction': int(preds[idx]),
                'probability': float(limit_up_probs[idx]),
                'confidence': float(np.max(probs[idx]))
            })
        
        return selected_stocks
    
    def save_model(self, model: nn.Module, filepath: str):
        """保存模型"""
        try:
            save_path = Path(filepath)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': self.model_config,
                'training_config': self.training_config
            }, save_path)
            
            logger.info(f"模型已保存到: {save_path}")
            
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
    
    def load_model(self, filepath: str) -> nn.Module:
        """加载模型"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # 重建模型
            model_config = checkpoint.get('model_config', self.model_config)
            model = self.build_model()
            model.load_state_dict(checkpoint['model_state_dict'])
            
            logger.info(f"模型已从 {filepath} 加载")
            return model
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return None
    
    def save_checkpoint(self, model: nn.Module, epoch: int, metrics: Dict, filename: str):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config
        }
        
        save_path = Path('models_saved') / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, save_path)
    
    def _save_training_history(self, train_history: List, val_history: List):
        """保存训练历史"""
        try:
            history = {
                'train': train_history,
                'validation': val_history
            }
            
            with open('logs/training_history.pkl', 'wb') as f:
                pickle.dump(history, f)
            
            # 绘制训练曲线
            self._plot_training_curves(train_history, val_history)
            
        except Exception as e:
            logger.error(f"保存训练历史失败: {e}")
    
    def _plot_training_curves(self, train_history: List, val_history: List):
        """绘制训练曲线"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            epochs = range(1, len(train_history) + 1)
            
            # Loss曲线
            axes[0, 0].plot(epochs, [h['loss'] for h in train_history], label='Train')
            axes[0, 0].plot(epochs, [h['loss'] for h in val_history], label='Validation')
            axes[0, 0].set_title('Loss')
            axes[0, 0].legend()
            
            # Accuracy曲线
            axes[0, 1].plot(epochs, [h['accuracy'] for h in train_history], label='Train')
            axes[0, 1].plot(epochs, [h['accuracy'] for h in val_history], label='Validation')
            axes[0, 1].set_title('Accuracy')
            axes[0, 1].legend()
            
            # F1曲线
            if 'f1' in val_history[0]:
                axes[1, 0].plot(epochs, [h['f1'] for h in val_history], label='Validation F1')
                axes[1, 0].set_title('F1 Score')
                axes[1, 0].legend()
            
            # 学习率曲线
            if hasattr(self.scheduler, 'get_last_lr'):
                axes[1, 1].plot(epochs, [self.scheduler.get_last_lr()[0]] * len(epochs))
                axes[1, 1].set_title('Learning Rate')
            
            plt.tight_layout()
            plt.savefig('logs/training_curves.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"绘制训练曲线失败: {e}")
    
    def _generate_evaluation_report(self, predictions: List, targets: List):
        """生成评估报告"""
        try:
            # 混淆矩阵
            cm = confusion_matrix(targets, predictions)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig('logs/confusion_matrix.png')
            plt.close()
            
            # 分类报告
            from sklearn.metrics import classification_report
            report = classification_report(targets, predictions)
            
            with open('logs/classification_report.txt', 'w') as f:
                f.write(report)
            
            logger.info("评估报告已生成")
            
        except Exception as e:
            logger.error(f"生成评估报告失败: {e}")
