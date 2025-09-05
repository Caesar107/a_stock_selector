# A股智能选股系统 - API文档

本文档描述了系统各个模块的API接口和使用方法。

## 数据模块 (src.data)

### DataLoader

负责A股市场数据的获取和处理。

#### 初始化

```python
from src.data import DataLoader

config = {
    'tushare_token': 'your_token',
    'data_dir': 'data'
}
data_loader = DataLoader(config)
```

#### 主要方法

##### get_stock_list()

获取A股股票列表。

```python
stock_list = data_loader.get_stock_list()
# 返回: pandas.DataFrame
# 列: ['stock_code', 'stock_name']
```

##### get_daily_data(stock_code, start_date, end_date)

获取股票日线数据。

```python
daily_data = data_loader.get_daily_data(
    stock_code='000001', 
    start_date='20230101', 
    end_date='20231231'
)
# 返回: pandas.DataFrame
# 包含价格、成交量、技术指标等
```

##### load_training_data(lookback_days=252)

加载训练数据。

```python
train_data, val_data = data_loader.load_training_data(lookback_days=252)
# 返回: (pandas.DataFrame, pandas.DataFrame)
```

##### load_latest_data()

加载最新数据用于预测。

```python
latest_data = data_loader.load_latest_data()
# 返回: pandas.DataFrame
```

## 特征工程模块 (src.features)

### FeatureEngine

负责特征提取和工程处理。

#### 初始化

```python
from src.features import FeatureEngine

feature_engine = FeatureEngine(config)
```

#### 主要方法

##### engineer_features(df, mode='train')

执行特征工程。

```python
engineered_data = feature_engine.engineer_features(df, mode='train')
# 参数:
#   df: 原始数据DataFrame
#   mode: 'train' 或 'predict'
# 返回: pandas.DataFrame (包含所有特征)
```

##### prepare_model_inputs(df, target_col='limit_up_1d', sequence_length=30)

准备模型输入数据。

```python
X, y = feature_engine.prepare_model_inputs(
    df=engineered_data,
    target_col='limit_up_1d',
    sequence_length=30
)
# 返回: (numpy.ndarray, numpy.ndarray)
# X shape: (samples, sequence_length, features)
# y shape: (samples,)
```

##### normalize_features(X, mode='train')

标准化特征。

```python
X_normalized = feature_engine.normalize_features(X, mode='train')
# 返回: numpy.ndarray
```

## 模型模块 (src.models)

### StockTransformer

基础Transformer模型。

#### 初始化

```python
from src.models import StockTransformer

model_config = {
    'input_dim': 100,
    'd_model': 256,
    'nhead': 8,
    'num_layers': 6,
    'seq_length': 30,
    'dropout': 0.1,
    'num_classes': 2
}

model = StockTransformer(model_config)
```

#### 前向传播

```python
import torch

# 输入: [batch_size, seq_length, input_dim]
x = torch.randn(32, 30, 100)

# 输出: (分类结果, 回归结果)
class_output, reg_output = model(x)
# class_output shape: [batch_size, num_classes]
# reg_output shape: [batch_size, 1]
```

### StockTransformerAdvanced

增强版Transformer模型，支持多任务输出。

```python
from src.models import StockTransformerAdvanced

model = StockTransformerAdvanced(model_config)
outputs = model(x)
# 返回字典:
# {
#     'limit_up': [batch_size, 2],
#     'big_rise': [batch_size, 2], 
#     'returns': [batch_size, 3]
# }
```

### ModelTrainer

模型训练器。

#### 初始化

```python
from src.models import ModelTrainer

trainer = ModelTrainer(config)
```

#### 主要方法

##### train(X_train, y_train, X_val, y_val, model_type='transformer')

训练模型。

```python
model = trainer.train(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    model_type='transformer'  # 或 'advanced_transformer'
)
# 返回: 训练好的模型
```

##### predict(model, X)

预测。

```python
predictions, probabilities = trainer.predict(model, X)
# 返回: (预测类别, 预测概率)
```

##### select_top_stocks(predictions, stock_codes, top_n=10)

选择top股票。

```python
selected_stocks = trainer.select_top_stocks(
    predictions=(preds, probs),
    stock_codes=stock_codes,
    top_n=5
)
# 返回: List[Dict] 包含推荐股票信息
```

##### save_model(model, filepath)

保存模型。

```python
trainer.save_model(model, "models_saved/my_model.pth")
```

##### load_model(filepath)

加载模型。

```python
model = trainer.load_model("models_saved/my_model.pth")
```

## 回测模块 (src.backtest)

### Backtester

策略回测器。

#### 初始化

```python
from src.backtest import Backtester

backtester = Backtester(config)
```

#### 主要方法

##### run_backtest(predictions_data, price_data, start_date, end_date)

运行回测。

```python
results = backtester.run_backtest(
    predictions_data=predictions_df,  # 包含预测结果的DataFrame
    price_data=price_df,              # 包含价格数据的DataFrame
    start_date='2020-01-01',
    end_date='2023-12-31'
)
# 返回: Dict 包含回测结果指标
```

##### generate_report(results)

生成回测报告。

```python
backtester.generate_report(results)
# 生成HTML报告和图表到 logs/backtest_reports/
```

#### 返回结果格式

```python
# results 字典结构
{
    'performance': {
        'total_return': 0.15,        # 总收益率
        'annualized_return': 0.12,   # 年化收益率
        'volatility': 0.20,          # 波动率
        'sharpe_ratio': 0.60,        # 夏普比率
        'max_drawdown': 0.08,        # 最大回撤
        'alpha': 0.05,               # Alpha
        'beta': 1.2                  # Beta
    },
    'trading': {
        'total_trades': 150,         # 总交易次数
        'win_rate': 0.65,           # 胜率
        'avg_holding_period': 1,     # 平均持仓期
        'avg_position_size': 0.1     # 平均仓位大小
    },
    'risk': {
        'value_at_risk_95': -0.03,   # 95% VaR
        'expected_shortfall_95': -0.045,  # 95% ES
        'downside_deviation': 0.15   # 下行偏差
    }
}
```

## 推送模块 (src.notification)

### Notifier

消息推送器。

#### 初始化

```python
from src.notification import Notifier

notifier = Notifier(config)
```

#### 主要方法

##### send_daily_recommendations(selected_stocks)

发送每日推荐。

```python
selected_stocks = [
    {
        'stock_code': '000001',
        'prediction': 1,
        'probability': 0.85,
        'confidence': 0.90
    },
    # ...更多股票
]

notifier.send_daily_recommendations(selected_stocks)
```

##### send_alert(alert_type, message, urgent=False)

发送告警。

```python
notifier.send_alert(
    alert_type="系统错误",
    message="模型训练失败",
    urgent=True
)
```

##### send_performance_report(performance_data)

发送策略表现报告。

```python
notifier.send_performance_report(backtest_results)
```

##### test_all_channels()

测试所有推送渠道。

```python
results = notifier.test_all_channels()
# 返回: Dict 各渠道测试结果
```

## 配置管理

### 加载配置

```python
from config.settings import load_config

# 加载默认配置
config = load_config()

# 加载自定义配置
config = load_config('config/custom_config.yaml')
```

### 配置结构

```python
# config 字典的主要结构
{
    'data': {
        'tushare_token': '',
        'data_dir': 'data',
        'cache_enabled': True
    },
    'model': {
        'input_dim': 100,
        'd_model': 256,
        'nhead': 8,
        'num_layers': 6
    },
    'training': {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 100
    },
    'backtest': {
        'initial_capital': 1000000,
        'commission_rate': 0.0003,
        'max_position_size': 0.1
    },
    'notification': {
        'email': {'enabled': True, ...},
        'wechat': {'enabled': False, ...}
    }
}
```

## 完整使用示例

### 1. 训练模型完整流程

```python
from config.settings import load_config
from src.data import DataLoader
from src.features import FeatureEngine
from src.models import ModelTrainer

# 加载配置
config = load_config()

# 1. 数据获取
data_loader = DataLoader(config)
train_data, val_data = data_loader.load_training_data()

# 2. 特征工程
feature_engine = FeatureEngine(config)
train_features = feature_engine.engineer_features(train_data, mode='train')
val_features = feature_engine.engineer_features(val_data, mode='predict')

# 3. 准备模型输入
X_train, y_train = feature_engine.prepare_model_inputs(train_features)
X_val, y_val = feature_engine.prepare_model_inputs(val_features)

# 4. 标准化
X_train = feature_engine.normalize_features(X_train, mode='train')
X_val = feature_engine.normalize_features(X_val, mode='predict')

# 5. 训练模型
trainer = ModelTrainer(config)
model = trainer.train(X_train, y_train, X_val, y_val)

# 6. 保存模型
trainer.save_model(model, "models_saved/latest_model.pth")
```

### 2. 预测选股完整流程

```python
from config.settings import load_config
from src.data import DataLoader
from src.features import FeatureEngine
from src.models import ModelTrainer
from src.notification import Notifier

# 加载配置
config = load_config()

# 1. 获取最新数据
data_loader = DataLoader(config)
latest_data = data_loader.load_latest_data()

# 2. 特征工程
feature_engine = FeatureEngine(config)
features = feature_engine.engineer_features(latest_data, mode='predict')

# 3. 准备输入
X, _ = feature_engine.prepare_model_inputs(features)
X = feature_engine.normalize_features(X, mode='predict')

# 4. 加载模型并预测
trainer = ModelTrainer(config)
model = trainer.load_model("models_saved/latest_model.pth")
predictions, probabilities = trainer.predict(model, X)

# 5. 选择top股票
stock_codes = latest_data['stock_code'].tolist()
selected_stocks = trainer.select_top_stocks(
    (predictions, probabilities), 
    stock_codes, 
    top_n=5
)

# 6. 推送结果
notifier = Notifier(config)
notifier.send_daily_recommendations(selected_stocks)
```

### 3. 回测完整流程

```python
import pandas as pd
from src.backtest import Backtester

# 准备回测数据
# predictions_data: 包含历史预测结果
# price_data: 包含历史价格数据

backtester = Backtester(config)
results = backtester.run_backtest(
    predictions_data=predictions_df,
    price_data=price_df,
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# 生成报告
backtester.generate_report(results)

# 发送报告
notifier = Notifier(config)
notifier.send_performance_report(results['performance'])
```

## 错误处理

所有模块都包含完善的错误处理机制。常见异常类型：

- `DataLoadError`: 数据加载失败
- `ModelTrainingError`: 模型训练失败  
- `PredictionError`: 预测失败
- `NotificationError`: 推送失败

使用try-catch处理异常：

```python
try:
    model = trainer.train(X_train, y_train, X_val, y_val)
except Exception as e:
    logger.error(f"训练失败: {e}")
    # 错误处理逻辑
```

## 日志记录

所有模块都使用Python标准logging模块记录日志：

```python
import logging

logger = logging.getLogger(__name__)
logger.info("操作成功")
logger.warning("警告信息") 
logger.error("错误信息")
```

日志文件位置：
- `logs/app.log`: 主应用日志
- `logs/scheduler.log`: 定时任务日志
- `logs/notifications/`: 推送记录
- `logs/backtest_reports/`: 回测报告
