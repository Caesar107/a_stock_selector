#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股智能选股系统主入口
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.data_loader import DataLoader
from src.models.trainer import ModelTrainer
from src.backtest.backtester import Backtester
from src.notification.notifier import Notifier
from config.settings import load_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def train_model(config):
    """训练模型"""
    logger.info("开始训练模型...")
    
    # 加载数据
    data_loader = DataLoader(config)
    train_data, val_data = data_loader.load_training_data()
    
    # 训练模型
    trainer = ModelTrainer(config)
    model = trainer.train(train_data, val_data)
    
    # 保存模型
    trainer.save_model(model, "models_saved/latest_model.pth")
    logger.info("模型训练完成并已保存")


def predict_stocks(config):
    """预测股票并推送"""
    logger.info("开始股票预测...")
    
    # 加载最新数据
    data_loader = DataLoader(config)
    latest_data = data_loader.load_latest_data()
    
    # 加载模型并预测
    trainer = ModelTrainer(config)
    model = trainer.load_model("models_saved/latest_model.pth")
    predictions = trainer.predict(model, latest_data)
    
    # 选择顶部股票
    selected_stocks = trainer.select_top_stocks(predictions, top_n=config.get('top_stocks', 5))
    
    # 推送结果
    notifier = Notifier(config)
    notifier.send_daily_recommendations(selected_stocks)
    
    logger.info(f"已推送 {len(selected_stocks)} 只推荐股票")


def run_backtest(config):
    """运行回测"""
    logger.info("开始历史回测...")
    
    backtester = Backtester(config)
    results = backtester.run_backtest()
    
    # 生成回测报告
    backtester.generate_report(results)
    logger.info("回测完成，报告已生成")


def main():
    parser = argparse.ArgumentParser(description='A股智能选股系统')
    parser.add_argument('--mode', choices=['train', 'predict', 'backtest'], 
                       required=True, help='运行模式')
    parser.add_argument('--config', default='config/config.yaml', 
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    try:
        if args.mode == 'train':
            train_model(config)
        elif args.mode == 'predict':
            predict_stocks(config)
        elif args.mode == 'backtest':
            run_backtest(config)
    except Exception as e:
        logger.error(f"运行错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
