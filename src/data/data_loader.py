# -*- coding: utf-8 -*-
"""
数据加载器 - 负责获取A股市场各类数据
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import tushare as ts
import akshare as ak
from pathlib import Path

logger = logging.getLogger(__name__)


class DataLoader:
    """A股数据加载器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_dir = Path(config.get('data_dir', 'data'))
        self.data_dir.mkdir(exist_ok=True)
        
        # 初始化Tushare
        if config.get('tushare_token'):
            ts.set_token(config['tushare_token'])
            self.ts_pro = ts.pro_api()
        else:
            logger.warning("未配置Tushare token，部分功能可能受限")
            self.ts_pro = None
    
    def get_stock_list(self) -> pd.DataFrame:
        """获取A股股票列表"""
        try:
            # 使用akshare获取股票列表
            stock_list = ak.stock_info_a_code_name()
            stock_list.columns = ['stock_code', 'stock_name']
            
            # 过滤掉ST、*ST等风险股票
            stock_list = stock_list[~stock_list['stock_name'].str.contains('ST|退')]
            
            logger.info(f"获取到 {len(stock_list)} 只股票")
            return stock_list
            
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return pd.DataFrame()
    
    def get_daily_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取日线数据"""
        try:
            # 使用akshare获取日线数据
            df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", 
                                   start_date=start_date, end_date=end_date, adjust="qfq")
            
            if df.empty:
                return pd.DataFrame()
            
            # 标准化列名
            df.columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount', 
                         'amplitude', 'change_pct', 'change_amount', 'turnover']
            
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # 计算基础技术指标
            df = self._calculate_basic_indicators(df)
            
            return df
            
        except Exception as e:
            logger.error(f"获取 {stock_code} 日线数据失败: {e}")
            return pd.DataFrame()
    
    def get_minute_data(self, stock_code: str, period: str = "5") -> pd.DataFrame:
        """获取分钟级数据"""
        try:
            # 获取实时分钟数据
            df = ak.stock_zh_a_minute(symbol=stock_code, period=period, adjust="qfq")
            
            if df.empty:
                return pd.DataFrame()
            
            df.columns = ['datetime', 'open', 'close', 'high', 'low', 'volume', 'amount']
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            return df
            
        except Exception as e:
            logger.error(f"获取 {stock_code} 分钟数据失败: {e}")
            return pd.DataFrame()
    
    def get_financial_data(self, stock_code: str) -> pd.DataFrame:
        """获取财务数据"""
        try:
            if not self.ts_pro:
                logger.warning("未配置Tushare，无法获取财务数据")
                return pd.DataFrame()
            
            # 获取基本财务指标
            df = self.ts_pro.fina_indicator(ts_code=stock_code + '.SH' if stock_code.startswith('6') else stock_code + '.SZ')
            
            if df.empty:
                return pd.DataFrame()
            
            # 选择关键财务指标
            key_columns = ['end_date', 'roe', 'roa', 'gross_profit_margin', 
                          'net_profit_margin', 'debt_to_assets', 'current_ratio',
                          'quick_ratio', 'eps', 'bvps', 'pe', 'pb']
            
            df = df[key_columns].copy()
            df['end_date'] = pd.to_datetime(df['end_date'])
            
            return df
            
        except Exception as e:
            logger.error(f"获取 {stock_code} 财务数据失败: {e}")
            return pd.DataFrame()
    
    def get_market_sentiment(self, date: str) -> Dict:
        """获取市场情绪数据"""
        try:
            sentiment_data = {}
            
            # 上证指数数据
            sh_index = ak.stock_zh_index_daily(symbol="sh000001")
            if not sh_index.empty:
                latest = sh_index.iloc[-1]
                sentiment_data['sh_change_pct'] = latest['涨跌幅']
                sentiment_data['sh_volume'] = latest['成交量']
            
            # 深证成指数据
            sz_index = ak.stock_zh_index_daily(symbol="sz399001")
            if not sz_index.empty:
                latest = sz_index.iloc[-1]
                sentiment_data['sz_change_pct'] = latest['涨跌幅']
                sentiment_data['sz_volume'] = latest['成交量']
            
            # 涨跌停统计
            limit_stats = ak.stock_zt_pool_dtgc_em(date=date.replace('-', ''))
            if not limit_stats.empty:
                sentiment_data['limit_up_count'] = len(limit_stats)
            else:
                sentiment_data['limit_up_count'] = 0
            
            # VIX指数等市场情绪指标可以在这里添加
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"获取市场情绪数据失败: {e}")
            return {}
    
    def get_money_flow(self, stock_code: str) -> pd.DataFrame:
        """获取资金流向数据"""
        try:
            # 获取资金流向数据
            df = ak.stock_individual_fund_flow(stock=stock_code, market="SH" if stock_code.startswith('6') else "SZ")
            
            if df.empty:
                return pd.DataFrame()
            
            # 标准化列名
            df.columns = ['date', 'close', 'change_pct', 'main_net_inflow', 
                         'main_net_inflow_pct', 'super_large_net_inflow', 
                         'super_large_net_inflow_pct', 'large_net_inflow',
                         'large_net_inflow_pct', 'medium_net_inflow',
                         'medium_net_inflow_pct', 'small_net_inflow',
                         'small_net_inflow_pct']
            
            df['date'] = pd.to_datetime(df['date'])
            
            return df
            
        except Exception as e:
            logger.error(f"获取 {stock_code} 资金流向数据失败: {e}")
            return pd.DataFrame()
    
    def _calculate_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算基础技术指标"""
        try:
            # MA均线
            df['ma5'] = df['close'].rolling(window=5).mean()
            df['ma10'] = df['close'].rolling(window=10).mean()
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['ma60'] = df['close'].rolling(window=60).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # 布林带
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + 2 * bb_std
            df['bb_lower'] = df['bb_middle'] - 2 * bb_std
            
            # KDJ
            low_min = df['low'].rolling(window=9).min()
            high_max = df['high'].rolling(window=9).max()
            rsv = (df['close'] - low_min) / (high_max - low_min) * 100
            df['k'] = rsv.ewm(com=2).mean()
            df['d'] = df['k'].ewm(com=2).mean()
            df['j'] = 3 * df['k'] - 2 * df['d']
            
            return df
            
        except Exception as e:
            logger.error(f"计算技术指标失败: {e}")
            return df
    
    def load_training_data(self, lookback_days: int = 252) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载训练数据"""
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y%m%d')
        
        all_data = []
        stock_list = self.get_stock_list()
        
        logger.info(f"开始加载 {len(stock_list)} 只股票的训练数据...")
        
        for idx, row in stock_list.iterrows():
            stock_code = row['stock_code']
            
            # 获取日线数据
            daily_data = self.get_daily_data(stock_code, start_date, end_date)
            if daily_data.empty:
                continue
            
            # 获取财务数据
            financial_data = self.get_financial_data(stock_code)
            
            # 获取资金流向数据
            money_flow_data = self.get_money_flow(stock_code)
            
            # 合并数据
            merged_data = self._merge_all_data(daily_data, financial_data, money_flow_data)
            merged_data['stock_code'] = stock_code
            
            all_data.append(merged_data)
            
            if idx % 100 == 0:
                logger.info(f"已处理 {idx}/{len(stock_list)} 只股票")
        
        # 合并所有数据
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # 分割训练集和验证集
        split_date = combined_data['date'].quantile(0.8)
        train_data = combined_data[combined_data['date'] <= split_date]
        val_data = combined_data[combined_data['date'] > split_date]
        
        logger.info(f"训练数据加载完成: 训练集 {len(train_data)} 条，验证集 {len(val_data)} 条")
        
        return train_data, val_data
    
    def load_latest_data(self) -> pd.DataFrame:
        """加载最新数据用于预测"""
        stock_list = self.get_stock_list()
        latest_data = []
        
        today = datetime.now().strftime('%Y%m%d')
        lookback_date = (datetime.now() - timedelta(days=60)).strftime('%Y%m%d')
        
        logger.info("加载最新数据用于预测...")
        
        for idx, row in stock_list.iterrows():
            stock_code = row['stock_code']
            
            # 获取最近60天的数据
            daily_data = self.get_daily_data(stock_code, lookback_date, today)
            if daily_data.empty or len(daily_data) < 30:
                continue
            
            # 获取最新的市场情绪数据
            sentiment_data = self.get_market_sentiment(today)
            
            # 添加市场情绪数据
            for key, value in sentiment_data.items():
                daily_data[key] = value
            
            daily_data['stock_code'] = stock_code
            latest_data.append(daily_data.tail(1))  # 只保留最新一条数据
        
        result = pd.concat(latest_data, ignore_index=True)
        logger.info(f"最新数据加载完成: {len(result)} 只股票")
        
        return result
    
    def _merge_all_data(self, daily_data: pd.DataFrame, 
                       financial_data: pd.DataFrame, 
                       money_flow_data: pd.DataFrame) -> pd.DataFrame:
        """合并所有数据源"""
        result = daily_data.copy()
        
        # 合并财务数据（使用最新的财务数据）
        if not financial_data.empty:
            latest_financial = financial_data.iloc[0]
            for col in financial_data.columns:
                if col != 'end_date':
                    result[f'fin_{col}'] = latest_financial[col]
        
        # 合并资金流向数据
        if not money_flow_data.empty:
            money_flow_data = money_flow_data.rename(columns={'date': 'date'})
            result = pd.merge(result, money_flow_data, on='date', how='left', suffixes=('', '_flow'))
        
        return result
    
    def calculate_target_labels(self, data: pd.DataFrame, 
                              target_days: List[int] = [1, 3, 5]) -> pd.DataFrame:
        """计算目标标签（未来涨幅）"""
        result = data.copy()
        
        for days in target_days:
            # 计算未来N天的涨幅
            future_return = data.groupby('stock_code')['close'].shift(-days) / data['close'] - 1
            result[f'future_return_{days}d'] = future_return
            
            # 创建涨停标签（涨幅>9.5%认为是涨停）
            result[f'limit_up_{days}d'] = (future_return > 0.095).astype(int)
            
            # 创建大涨标签（涨幅>5%）
            result[f'big_rise_{days}d'] = (future_return > 0.05).astype(int)
        
        return result
