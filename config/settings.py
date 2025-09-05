# -*- coding: utf-8 -*-
"""
配置管理器
"""

import yaml
import os
from typing import Dict, Any
from pathlib import Path

def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 从环境变量覆盖敏感配置
        config = _override_from_env(config)
        
        return config
    except Exception as e:
        raise Exception(f"加载配置失败: {e}")

def _override_from_env(config: Dict) -> Dict:
    """从环境变量覆盖配置"""
    # 数据源配置
    if 'TUSHARE_TOKEN' in os.environ:
        config.setdefault('data', {})['tushare_token'] = os.environ['TUSHARE_TOKEN']
    
    # 邮件配置
    if 'EMAIL_PASSWORD' in os.environ:
        config.setdefault('notification', {}).setdefault('email', {})['sender_password'] = os.environ['EMAIL_PASSWORD']
    
    if 'EMAIL_SENDER' in os.environ:
        config.setdefault('notification', {}).setdefault('email', {})['sender_email'] = os.environ['EMAIL_SENDER']
    
    # 微信推送配置
    if 'WECHAT_SCKEY' in os.environ:
        config.setdefault('notification', {}).setdefault('wechat', {})['sckey'] = os.environ['WECHAT_SCKEY']
    
    # 钉钉配置
    if 'DINGTALK_WEBHOOK' in os.environ:
        config.setdefault('notification', {}).setdefault('dingtalk', {})['webhook_url'] = os.environ['DINGTALK_WEBHOOK']
    
    # Telegram配置
    if 'TELEGRAM_BOT_TOKEN' in os.environ:
        config.setdefault('notification', {}).setdefault('telegram', {})['bot_token'] = os.environ['TELEGRAM_BOT_TOKEN']
    
    if 'TELEGRAM_CHAT_ID' in os.environ:
        config.setdefault('notification', {}).setdefault('telegram', {})['chat_id'] = os.environ['TELEGRAM_CHAT_ID']
    
    return config
