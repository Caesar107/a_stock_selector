# -*- coding: utf-8 -*-
"""
定时任务调度器 - 负责定时运行选股策略
"""

import schedule
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import load_config
from main import predict_stocks, train_model

logger = logging.getLogger(__name__)


class Scheduler:
    """定时任务调度器"""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        self.config = load_config(config_path)
        self.schedule_config = self.config.get('schedule', {})
        
        if not self.schedule_config.get('enabled', False):
            logger.warning("定时任务未启用")
            return
        
        self.setup_schedule()
    
    def setup_schedule(self):
        """设置定时任务"""
        predict_time = self.schedule_config.get('predict_time', '08:30')
        weekend_enabled = self.schedule_config.get('weekend_enabled', False)
        
        if weekend_enabled:
            # 每天运行
            schedule.every().day.at(predict_time).do(self.daily_prediction_job)
        else:
            # 只在工作日运行
            schedule.every().monday.at(predict_time).do(self.daily_prediction_job)
            schedule.every().tuesday.at(predict_time).do(self.daily_prediction_job)
            schedule.every().wednesday.at(predict_time).do(self.daily_prediction_job)
            schedule.every().thursday.at(predict_time).do(self.daily_prediction_job)
            schedule.every().friday.at(predict_time).do(self.daily_prediction_job)
        
        # 每周日重新训练模型
        schedule.every().sunday.at("02:00").do(self.weekly_training_job)
        
        logger.info(f"定时任务已设置: 每日预测时间 {predict_time}")
    
    def daily_prediction_job(self):
        """每日预测任务"""
        try:
            logger.info("开始执行每日预测任务")
            predict_stocks(self.config)
            logger.info("每日预测任务完成")
        except Exception as e:
            logger.error(f"每日预测任务失败: {e}")
    
    def weekly_training_job(self):
        """每周训练任务"""
        try:
            logger.info("开始执行每周训练任务")
            train_model(self.config)
            logger.info("每周训练任务完成")
        except Exception as e:
            logger.error(f"每周训练任务失败: {e}")
    
    def run(self):
        """运行调度器"""
        if not self.schedule_config.get('enabled', False):
            logger.info("定时任务未启用，退出")
            return
        
        logger.info("定时任务调度器启动")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # 每分钟检查一次


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/scheduler.log'),
            logging.StreamHandler()
        ]
    )
    
    scheduler = Scheduler()
    scheduler.run()
