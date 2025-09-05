# A股智能选股系统 - 部署指南

## 系统环境要求

### 硬件要求
- **CPU**: 至少4核心，推荐8核心以上
- **内存**: 至少8GB，推荐16GB以上
- **存储**: 至少50GB可用空间
- **GPU**: 可选，NVIDIA GPU用于加速训练（推荐）

### 软件环境
- **操作系统**: Windows 10/11, Ubuntu 18.04+, CentOS 7+
- **Python**: 3.8或更高版本
- **依赖管理**: pip或conda

## 快速部署

### 1. 环境准备

```bash
# 克隆或下载项目
git clone <repository_url>
cd a_stock_selector

# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置系统

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑环境变量配置
# 设置Tushare Token、邮件配置等敏感信息
```

### 3. 配置文件设置

编辑 `config/config.yaml` 文件：

```yaml
# 基础配置示例
data:
  tushare_token: ""  # 将从环境变量读取

notification:
  email:
    enabled: true
    receiver_emails:
      - "your_email@example.com"
  
selection:
  top_stocks: 5
  min_probability: 0.6
```

### 4. 测试运行

```bash
# 测试数据获取
python -c "from src.data import DataLoader; dl = DataLoader({}); print('数据模块正常')"

# 测试推送功能
python -c "from src.notification import Notifier; from config.settings import load_config; n = Notifier(load_config()); n.test_all_channels()"
```

## 使用方法

### 训练模型

```bash
# 训练新模型
python main.py --mode train

# 使用自定义配置
python main.py --mode train --config config/custom_config.yaml
```

### 预测选股

```bash
# 运行选股预测
python main.py --mode predict

# 预测结果将自动推送到配置的渠道
```

### 策略回测

```bash
# 运行历史回测
python main.py --mode backtest

# 回测报告将保存到 logs/backtest_reports/ 目录
```

### 定时运行

```bash
# 启动定时任务调度器
python scheduler.py

# 系统将在每个交易日的指定时间自动运行
```

## 数据源配置

### 1. Tushare配置

1. 注册Tushare账号：https://tushare.pro
2. 获取API Token
3. 在 `.env` 文件中设置：`TUSHARE_TOKEN=your_token`

### 2. AKShare配置

AKShare是免费数据源，无需配置，但可能有访问限制。

## 推送渠道配置

### 1. 邮件推送

```bash
# 在 .env 文件中配置
EMAIL_SENDER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
```

QQ邮箱设置示例：
- SMTP服务器：smtp.qq.com
- 端口：587
- 密码：使用QQ邮箱的授权码

### 2. 微信推送（Server酱）

1. 关注Server酱公众号
2. 获取SCKEY
3. 设置环境变量：`WECHAT_SCKEY=your_sckey`

### 3. 钉钉推送

1. 创建钉钉群
2. 添加自定义机器人
3. 获取Webhook URL
4. 设置：`DINGTALK_WEBHOOK=your_webhook_url`

### 4. Telegram推送

1. 创建Telegram Bot
2. 获取Bot Token和Chat ID
3. 设置环境变量：
   ```
   TELEGRAM_BOT_TOKEN=your_bot_token
   TELEGRAM_CHAT_ID=your_chat_id
   ```

## 生产环境部署

### 1. Docker部署

```dockerfile
# Dockerfile示例
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "scheduler.py"]
```

```bash
# 构建镜像
docker build -t stock-selector .

# 运行容器
docker run -d --name stock-selector \
  --env-file .env \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data:/app/data \
  stock-selector
```

### 2. 系统服务部署

创建systemd服务（Linux）：

```ini
# /etc/systemd/system/stock-selector.service
[Unit]
Description=A股智能选股系统
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/a_stock_selector
Environment=PATH=/path/to/venv/bin
ExecStart=/path/to/venv/bin/python scheduler.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# 启用服务
sudo systemctl enable stock-selector
sudo systemctl start stock-selector
sudo systemctl status stock-selector
```

### 3. 云服务器部署

推荐配置：
- **阿里云ECS**: 2核4GB，按量付费
- **腾讯云CVM**: 2核4GB，包年包月
- **AWS EC2**: t3.medium实例

### 4. 监控和日志

```bash
# 查看运行日志
tail -f logs/app.log
tail -f logs/scheduler.log

# 监控系统状态
ps aux | grep python
df -h  # 检查磁盘空间
free -h  # 检查内存使用
```

## 性能优化

### 1. 数据缓存

```yaml
# config.yaml
data:
  cache_enabled: true
  cache_duration: 3600  # 1小时缓存
```

### 2. 模型优化

```yaml
# 减少模型复杂度
model:
  d_model: 128    # 降低维度
  num_layers: 4   # 减少层数
  seq_length: 20  # 缩短序列长度
```

### 3. 数据库优化

考虑使用数据库存储历史数据：

```python
# 使用SQLite或PostgreSQL
import sqlite3

conn = sqlite3.connect('data/stock_data.db')
# 存储和查询优化
```

## 故障排除

### 1. 常见问题

**问题**: 模块导入错误
```bash
ModuleNotFoundError: No module named 'src'
```
**解决**: 确保在项目根目录运行，或设置PYTHONPATH

**问题**: 数据获取失败
```bash
获取股票列表失败: Connection timeout
```
**解决**: 检查网络连接，或更换数据源

**问题**: 推送失败
```bash
邮件推送失败: Authentication failed
```
**解决**: 检查邮箱密码/授权码，确认SMTP设置

### 2. 日志分析

```bash
# 查看错误日志
grep "ERROR" logs/app.log

# 查看推送记录
ls logs/notifications/

# 查看回测报告
ls logs/backtest_reports/
```

### 3. 性能监控

```bash
# 监控Python进程
top -p $(pgrep -f "python.*main.py")

# 监控磁盘IO
iotop

# 监控网络
netstat -an | grep :80
```

## 安全建议

1. **敏感信息保护**
   - 使用环境变量存储API密钥
   - 不要将 `.env` 文件提交到版本控制

2. **网络安全**
   - 使用HTTPS/SSL连接
   - 定期更新依赖包
   - 设置防火墙规则

3. **数据备份**
   - 定期备份模型文件
   - 备份配置文件
   - 备份历史交易数据

4. **访问控制**
   - 限制服务器访问权限
   - 使用强密码
   - 启用双因素认证

## 技术支持

遇到问题时，请按以下步骤排查：

1. 查看日志文件 `logs/app.log`
2. 检查配置文件 `config/config.yaml`
3. 验证环境变量 `.env`
4. 测试网络连接
5. 检查依赖包版本

如需进一步支持，请提供：
- 详细错误信息
- 系统环境信息
- 配置文件（隐去敏感信息）
- 相关日志片段
