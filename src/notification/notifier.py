# -*- coding: utf-8 -*-
"""
消息推送器 - 负责股票推荐的多渠道推送
"""

import logging
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class Notifier:
    """消息推送器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.notification_config = config.get('notification', {})
        
        # 邮件配置
        self.email_config = self.notification_config.get('email', {})
        
        # 微信推送配置（使用Server酱等服务）
        self.wechat_config = self.notification_config.get('wechat', {})
        
        # 钉钉推送配置
        self.dingtalk_config = self.notification_config.get('dingtalk', {})
        
        # Telegram推送配置
        self.telegram_config = self.notification_config.get('telegram', {})
    
    def send_daily_recommendations(self, selected_stocks: List[Dict]):
        """发送每日股票推荐"""
        logger.info(f"开始推送 {len(selected_stocks)} 只推荐股票")
        
        # 生成推送内容
        message_content = self._generate_recommendation_message(selected_stocks)
        
        # 多渠道推送
        success_count = 0
        
        if self.email_config.get('enabled', False):
            if self._send_email(message_content):
                success_count += 1
        
        if self.wechat_config.get('enabled', False):
            if self._send_wechat(message_content):
                success_count += 1
        
        if self.dingtalk_config.get('enabled', False):
            if self._send_dingtalk(message_content):
                success_count += 1
        
        if self.telegram_config.get('enabled', False):
            if self._send_telegram(message_content):
                success_count += 1
        
        logger.info(f"推送完成，成功 {success_count} 个渠道")
        
        # 保存推送记录
        self._save_notification_log(selected_stocks, success_count)
    
    def _generate_recommendation_message(self, selected_stocks: List[Dict]) -> Dict[str, str]:
        """生成推荐消息内容"""
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 文本版本
        text_content = f"""
🚀 A股智能选股推荐 📈

📅 推荐时间: {current_time}
📊 推荐股票数量: {len(selected_stocks)}

✨ 今日精选股票:
"""
        
        for i, stock in enumerate(selected_stocks, 1):
            confidence_level = "🔥" if stock['probability'] > 0.8 else "⭐" if stock['probability'] > 0.6 else "💡"
            text_content += f"""
{i}. {stock['stock_code']} {confidence_level}
   📈 涨停概率: {stock['probability']:.1%}
   🎯 预测置信度: {stock['confidence']:.1%}
"""
        
        text_content += f"""
⚠️ 风险提示:
• 本推荐基于AI算法分析，仅供参考
• 股市有风险，投资需谨慎
• 请结合自身风险承受能力进行投资决策
• 建议分散投资，控制仓位

📱 祝您投资顺利！
"""
        
        # HTML版本（用于邮件）
        html_content = f"""
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   color: white; padding: 20px; text-align: center; border-radius: 10px; }}
        .stock-card {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; 
                       border-radius: 8px; background: #f9f9f9; }}
        .high-confidence {{ border-left: 5px solid #27ae60; }}
        .medium-confidence {{ border-left: 5px solid #f39c12; }}
        .low-confidence {{ border-left: 5px solid #3498db; }}
        .probability {{ font-size: 18px; font-weight: bold; color: #e74c3c; }}
        .warning {{ background: #fff3cd; border: 1px solid #ffeaa7; 
                   padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .footer {{ text-align: center; color: #666; margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 A股智能选股推荐</h1>
        <p>推荐时间: {current_time}</p>
    </div>
    
    <h2>📊 今日精选股票 ({len(selected_stocks)}只)</h2>
"""
        
        for i, stock in enumerate(selected_stocks, 1):
            confidence_class = ("high-confidence" if stock['probability'] > 0.8 else 
                              "medium-confidence" if stock['probability'] > 0.6 else 
                              "low-confidence")
            
            emoji = "🔥" if stock['probability'] > 0.8 else "⭐" if stock['probability'] > 0.6 else "💡"
            
            html_content += f"""
    <div class="stock-card {confidence_class}">
        <h3>{emoji} {i}. {stock['stock_code']}</h3>
        <p><strong>涨停概率:</strong> <span class="probability">{stock['probability']:.1%}</span></p>
        <p><strong>预测置信度:</strong> {stock['confidence']:.1%}</p>
    </div>
"""
        
        html_content += """
    <div class="warning">
        <h3>⚠️ 重要风险提示</h3>
        <ul>
            <li>本推荐基于AI算法分析，仅供参考</li>
            <li>股市有风险，投资需谨慎</li>
            <li>请结合自身风险承受能力进行投资决策</li>
            <li>建议分散投资，控制仓位</li>
        </ul>
    </div>
    
    <div class="footer">
        <p>📱 祝您投资顺利！</p>
        <p><small>由A股智能选股系统自动生成</small></p>
    </div>
</body>
</html>
"""
        
        return {
            'text': text_content,
            'html': html_content,
            'subject': f'A股选股推荐 - {datetime.now().strftime("%m月%d日")} ({len(selected_stocks)}只精选)'
        }
    
    def _send_email(self, message_content: Dict[str, str]) -> bool:
        """发送邮件推送"""
        try:
            smtp_server = self.email_config.get('smtp_server', 'smtp.gmail.com')
            smtp_port = self.email_config.get('smtp_port', 587)
            sender_email = self.email_config.get('sender_email')
            sender_password = self.email_config.get('sender_password')
            receiver_emails = self.email_config.get('receiver_emails', [])
            
            if not all([sender_email, sender_password, receiver_emails]):
                logger.error("邮件配置不完整")
                return False
            
            # 创建邮件
            msg = MIMEMultipart('alternative')
            msg['Subject'] = message_content['subject']
            msg['From'] = sender_email
            msg['To'] = ', '.join(receiver_emails)
            
            # 添加文本和HTML内容
            text_part = MIMEText(message_content['text'], 'plain', 'utf-8')
            html_part = MIMEText(message_content['html'], 'html', 'utf-8')
            
            msg.attach(text_part)
            msg.attach(html_part)
            
            # 发送邮件
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
            
            logger.info(f"邮件推送成功，发送到 {len(receiver_emails)} 个邮箱")
            return True
            
        except Exception as e:
            logger.error(f"邮件推送失败: {e}")
            return False
    
    def _send_wechat(self, message_content: Dict[str, str]) -> bool:
        """发送微信推送（使用Server酱）"""
        try:
            sckey = self.wechat_config.get('sckey')
            if not sckey:
                logger.error("微信推送配置不完整")
                return False
            
            url = f"https://sctapi.ftqq.com/{sckey}.send"
            
            data = {
                'title': message_content['subject'],
                'desp': message_content['text']
            }
            
            response = requests.post(url, data=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('code') == 0:
                    logger.info("微信推送成功")
                    return True
                else:
                    logger.error(f"微信推送失败: {result.get('message')}")
            else:
                logger.error(f"微信推送HTTP错误: {response.status_code}")
            
            return False
            
        except Exception as e:
            logger.error(f"微信推送失败: {e}")
            return False
    
    def _send_dingtalk(self, message_content: Dict[str, str]) -> bool:
        """发送钉钉推送"""
        try:
            webhook_url = self.dingtalk_config.get('webhook_url')
            if not webhook_url:
                logger.error("钉钉推送配置不完整")
                return False
            
            # 钉钉消息格式
            data = {
                "msgtype": "markdown",
                "markdown": {
                    "title": message_content['subject'],
                    "text": message_content['text'].replace('🚀', '').replace('📈', '').replace('🔥', '').replace('⭐', '').replace('💡', '')
                }
            }
            
            headers = {'Content-Type': 'application/json'}
            response = requests.post(webhook_url, data=json.dumps(data), headers=headers, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('errcode') == 0:
                    logger.info("钉钉推送成功")
                    return True
                else:
                    logger.error(f"钉钉推送失败: {result.get('errmsg')}")
            else:
                logger.error(f"钉钉推送HTTP错误: {response.status_code}")
            
            return False
            
        except Exception as e:
            logger.error(f"钉钉推送失败: {e}")
            return False
    
    def _send_telegram(self, message_content: Dict[str, str]) -> bool:
        """发送Telegram推送"""
        try:
            bot_token = self.telegram_config.get('bot_token')
            chat_id = self.telegram_config.get('chat_id')
            
            if not all([bot_token, chat_id]):
                logger.error("Telegram推送配置不完整")
                return False
            
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            
            # 格式化消息（Telegram支持Markdown）
            formatted_text = message_content['text'].replace('🚀', '🚀').replace('📈', '📈')
            
            data = {
                'chat_id': chat_id,
                'text': formatted_text,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, data=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('ok'):
                    logger.info("Telegram推送成功")
                    return True
                else:
                    logger.error(f"Telegram推送失败: {result.get('description')}")
            else:
                logger.error(f"Telegram推送HTTP错误: {response.status_code}")
            
            return False
            
        except Exception as e:
            logger.error(f"Telegram推送失败: {e}")
            return False
    
    def _save_notification_log(self, selected_stocks: List[Dict], success_count: int):
        """保存推送记录"""
        try:
            log_dir = Path('logs/notifications')
            log_dir.mkdir(parents=True, exist_ok=True)
            
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'stock_count': len(selected_stocks),
                'selected_stocks': selected_stocks,
                'success_channels': success_count,
                'total_channels': sum([
                    self.email_config.get('enabled', False),
                    self.wechat_config.get('enabled', False),
                    self.dingtalk_config.get('enabled', False),
                    self.telegram_config.get('enabled', False)
                ])
            }
            
            # 按日期保存日志
            log_file = log_dir / f"notifications_{datetime.now().strftime('%Y%m%d')}.json"
            
            # 读取现有日志
            if log_file.exists():
                with open(log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            # 添加新记录
            logs.append(log_entry)
            
            # 保存日志
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
            
            logger.info(f"推送记录已保存: {log_file}")
            
        except Exception as e:
            logger.error(f"保存推送记录失败: {e}")
    
    def send_alert(self, alert_type: str, message: str, urgent: bool = False):
        """发送告警消息"""
        try:
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            alert_message = {
                'text': f"""
⚠️ 系统告警 {'🚨 紧急' if urgent else ''}

时间: {current_time}
类型: {alert_type}
消息: {message}

请及时处理！
""",
                'subject': f"{'[紧急]' if urgent else ''}系统告警 - {alert_type}"
            }
            
            # 只通过邮件和钉钉发送告警
            if self.email_config.get('enabled', False):
                self._send_email({**alert_message, 'html': alert_message['text']})
            
            if self.dingtalk_config.get('enabled', False):
                self._send_dingtalk(alert_message)
            
            logger.info(f"告警消息已发送: {alert_type}")
            
        except Exception as e:
            logger.error(f"发送告警失败: {e}")
    
    def send_performance_report(self, performance_data: Dict):
        """发送策略表现报告"""
        try:
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            report_content = f"""
📊 策略表现报告

📅 报告时间: {current_time}

📈 核心指标:
• 总收益率: {performance_data.get('total_return', 0):.2%}
• 年化收益率: {performance_data.get('annualized_return', 0):.2%}
• 夏普比率: {performance_data.get('sharpe_ratio', 0):.2f}
• 最大回撤: {performance_data.get('max_drawdown', 0):.2%}
• 胜率: {performance_data.get('win_rate', 0):.2%}

📊 交易统计:
• 总交易次数: {performance_data.get('total_trades', 0)}
• 平均持仓期: {performance_data.get('avg_holding_period', 0)} 天

🎯 本报告由系统自动生成
"""
            
            message_content = {
                'text': report_content,
                'html': report_content.replace('\n', '<br>'),
                'subject': f"策略表现报告 - {datetime.now().strftime('%Y年%m月%d日')}"
            }
            
            # 发送报告
            if self.email_config.get('enabled', False):
                self._send_email(message_content)
            
            logger.info("策略表现报告已发送")
            
        except Exception as e:
            logger.error(f"发送表现报告失败: {e}")
    
    def test_all_channels(self):
        """测试所有推送渠道"""
        test_message = {
            'text': f"""
🧪 推送测试消息

时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

这是一条测试消息，用于验证推送渠道是否正常工作。

如果您收到此消息，说明推送功能正常！✅
""",
            'html': """
<h2>🧪 推送测试消息</h2>
<p>这是一条测试消息，用于验证推送渠道是否正常工作。</p>
<p>如果您收到此消息，说明推送功能正常！✅</p>
""",
            'subject': "推送系统测试"
        }
        
        results = {}
        
        if self.email_config.get('enabled', False):
            results['email'] = self._send_email(test_message)
        
        if self.wechat_config.get('enabled', False):
            results['wechat'] = self._send_wechat(test_message)
        
        if self.dingtalk_config.get('enabled', False):
            results['dingtalk'] = self._send_dingtalk(test_message)
        
        if self.telegram_config.get('enabled', False):
            results['telegram'] = self._send_telegram(test_message)
        
        logger.info(f"推送测试完成: {results}")
        return results
