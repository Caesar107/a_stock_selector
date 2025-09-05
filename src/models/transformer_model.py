# -*- coding: utf-8 -*-
"""
基于Transformer的股票预测模型
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
import math

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class StockTransformer(nn.Module):
    """股票预测Transformer模型"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.input_dim = config.get('input_dim', 100)
        self.d_model = config.get('d_model', 256)
        self.nhead = config.get('nhead', 8)
        self.num_layers = config.get('num_layers', 6)
        self.seq_length = config.get('seq_length', 30)
        self.dropout = config.get('dropout', 0.1)
        self.num_classes = config.get('num_classes', 2)  # 二分类：涨停/不涨停
        
        # 输入投影层
        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=self.seq_length)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # 特征聚合层
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, self.num_classes)
        )
        
        # 回归器（预测具体涨幅）
        self.regressor = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, 1)
        )
        
        # 注意力权重可视化
        self.attention_weights = None
        
    def forward(self, x, return_attention=False):
        """
        前向传播
        Args:
            x: [batch_size, seq_length, input_dim]
            return_attention: 是否返回注意力权重
        """
        batch_size, seq_length, _ = x.shape
        
        # 输入投影
        x = self.input_projection(x)  # [batch_size, seq_length, d_model]
        
        # 添加位置编码
        x = x.transpose(0, 1)  # [seq_length, batch_size, d_model]
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # [batch_size, seq_length, d_model]
        
        # Transformer编码
        if return_attention:
            # 临时保存注意力权重
            attention_weights = []
            for layer in self.transformer_encoder.layers:
                x = layer(x)
                # 这里可以提取注意力权重，但需要修改TransformerEncoderLayer
            encoded = x
        else:
            encoded = self.transformer_encoder(x)  # [batch_size, seq_length, d_model]
        
        # 特征聚合
        # 平均池化
        avg_pooled = self.global_avg_pool(encoded.transpose(1, 2)).squeeze(-1)  # [batch_size, d_model]
        # 最大池化
        max_pooled = self.global_max_pool(encoded.transpose(1, 2)).squeeze(-1)  # [batch_size, d_model]
        
        # 连接两种池化结果
        pooled = torch.cat([avg_pooled, max_pooled], dim=1)  # [batch_size, d_model*2]
        
        # 分类和回归预测
        classification_output = self.classifier(pooled)  # [batch_size, num_classes]
        regression_output = self.regressor(pooled)  # [batch_size, 1]
        
        if return_attention:
            return classification_output, regression_output, attention_weights
        else:
            return classification_output, regression_output


class MultiTaskLoss(nn.Module):
    """多任务损失函数"""
    
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha  # 分类损失权重
        self.beta = beta    # 回归损失权重
        
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
    
    def forward(self, class_pred, reg_pred, class_target, reg_target):
        class_loss = self.classification_loss(class_pred, class_target)
        reg_loss = self.regression_loss(reg_pred.squeeze(), reg_target)
        
        total_loss = self.alpha * class_loss + self.beta * reg_loss
        
        return total_loss, class_loss, reg_loss


class StockTransformerAdvanced(nn.Module):
    """增强版股票预测Transformer模型"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.input_dim = config.get('input_dim', 100)
        self.d_model = config.get('d_model', 256)
        self.nhead = config.get('nhead', 8)
        self.num_layers = config.get('num_layers', 6)
        self.seq_length = config.get('seq_length', 30)
        self.dropout = config.get('dropout', 0.1)
        self.num_classes = config.get('num_classes', 2)
        
        # 多头输入处理
        self.price_projection = nn.Linear(20, self.d_model // 4)  # 价格相关特征
        self.volume_projection = nn.Linear(10, self.d_model // 4)  # 成交量相关特征
        self.technical_projection = nn.Linear(30, self.d_model // 4)  # 技术指标
        self.fundamental_projection = nn.Linear(40, self.d_model // 4)  # 基本面特征
        
        # 特征融合
        self.feature_fusion = nn.Linear(self.d_model, self.d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=self.seq_length)
        
        # 多尺度Transformer
        self.short_term_transformer = self._build_transformer(self.num_layers // 2)
        self.long_term_transformer = self._build_transformer(self.num_layers // 2)
        
        # 跨期注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.nhead,
            dropout=self.dropout,
            batch_first=True
        )
        
        # 自适应特征选择
        self.feature_selector = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.Sigmoid()
        )
        
        # 最终预测层
        self.final_projection = nn.Linear(self.d_model * 2, self.d_model)
        
        # 多目标预测头
        self.limit_up_head = nn.Linear(self.d_model, 2)  # 涨停预测
        self.big_rise_head = nn.Linear(self.d_model, 2)  # 大涨预测
        self.return_head = nn.Linear(self.d_model, 3)    # 1天、3天、5天涨幅预测
        
    def _build_transformer(self, num_layers):
        """构建Transformer编码器"""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True
        )
        return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: [batch_size, seq_length, input_dim]
        """
        batch_size, seq_length, input_dim = x.shape
        
        # 假设输入特征按类型分组
        price_features = x[:, :, :20]
        volume_features = x[:, :, 20:30]
        technical_features = x[:, :, 30:60]
        fundamental_features = x[:, :, 60:]
        
        # 多头特征投影
        price_embedded = self.price_projection(price_features)
        volume_embedded = self.volume_projection(volume_features)
        technical_embedded = self.technical_projection(technical_features)
        fundamental_embedded = self.fundamental_projection(fundamental_features)
        
        # 特征融合
        combined_features = torch.cat([
            price_embedded, volume_embedded, 
            technical_embedded, fundamental_embedded
        ], dim=-1)
        
        fused_features = self.feature_fusion(combined_features)
        
        # 添加位置编码
        fused_features = fused_features.transpose(0, 1)
        fused_features = self.pos_encoder(fused_features)
        fused_features = fused_features.transpose(0, 1)
        
        # 多尺度处理
        # 短期模式（关注最近的数据）
        short_term_input = fused_features[:, -seq_length//2:, :]
        short_term_output = self.short_term_transformer(short_term_input)
        
        # 长期模式（关注全部历史）
        long_term_output = self.long_term_transformer(fused_features)
        
        # 跨期注意力融合
        cross_attended, _ = self.cross_attention(
            short_term_output, long_term_output, long_term_output
        )
        
        # 特征聚合
        short_pooled = torch.mean(cross_attended, dim=1)  # [batch_size, d_model]
        long_pooled = torch.mean(long_term_output, dim=1)  # [batch_size, d_model]
        
        # 自适应特征选择
        combined_pooled = torch.cat([short_pooled, long_pooled], dim=1)
        attention_weights = self.feature_selector(combined_pooled)
        
        final_features = self.final_projection(combined_pooled) * attention_weights
        
        # 多目标预测
        limit_up_pred = self.limit_up_head(final_features)
        big_rise_pred = self.big_rise_head(final_features)
        return_pred = self.return_head(final_features)
        
        return {
            'limit_up': limit_up_pred,
            'big_rise': big_rise_pred,
            'returns': return_pred
        }


class FocalLoss(nn.Module):
    """Focal Loss - 处理类别不平衡问题"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class StockPredictionEnsemble(nn.Module):
    """模型集成"""
    
    def __init__(self, models: list, weights: Optional[list] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            self.weights = weights
    
    def forward(self, x):
        predictions = []
        
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # 加权平均
        if isinstance(predictions[0], dict):
            # 处理多输出模型
            ensemble_pred = {}
            for key in predictions[0].keys():
                weighted_preds = []
                for i, pred in enumerate(predictions):
                    weighted_preds.append(pred[key] * self.weights[i])
                ensemble_pred[key] = torch.stack(weighted_preds).sum(dim=0)
            return ensemble_pred
        else:
            # 处理单输出模型
            weighted_preds = []
            for i, pred in enumerate(predictions):
                if isinstance(pred, tuple):
                    # 多任务输出
                    weighted_class = pred[0] * self.weights[i]
                    weighted_reg = pred[1] * self.weights[i]
                    weighted_preds.append((weighted_class, weighted_reg))
                else:
                    weighted_preds.append(pred * self.weights[i])
            
            if isinstance(weighted_preds[0], tuple):
                class_preds = torch.stack([p[0] for p in weighted_preds]).sum(dim=0)
                reg_preds = torch.stack([p[1] for p in weighted_preds]).sum(dim=0)
                return class_preds, reg_preds
            else:
                return torch.stack(weighted_preds).sum(dim=0)
