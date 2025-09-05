# -*- coding: utf-8 -*-
"""
特征工程器 - 负责提取和构建用于机器学习的特征
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
import talib

logger = logging.getLogger(__name__)


class FeatureEngine:
    """特征工程器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.selected_features = None
        
    def extract_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取技术分析特征"""
        try:
            result = df.copy()
            
            # 价格相关特征
            result['price_momentum_5'] = df['close'] / df['close'].shift(5) - 1
            result['price_momentum_10'] = df['close'] / df['close'].shift(10) - 1
            result['price_momentum_20'] = df['close'] / df['close'].shift(20) - 1
            
            # 成交量特征
            result['volume_ratio_5'] = df['volume'] / df['volume'].rolling(5).mean()
            result['volume_ratio_20'] = df['volume'] / df['volume'].rolling(20).mean()
            result['volume_momentum'] = df['volume'] / df['volume'].shift(1) - 1
            
            # 振幅特征
            result['amplitude'] = (df['high'] - df['low']) / df['close']
            result['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
            result['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
            result['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
            
            # 相对位置特征
            result['close_position_20'] = (df['close'] - df['low'].rolling(20).min()) / \
                                         (df['high'].rolling(20).max() - df['low'].rolling(20).min() + 1e-8)
            
            # 使用TA-Lib计算更多技术指标
            if len(df) > 30:  # 确保有足够的数据
                try:
                    # 动量指标
                    result['rsi_14'] = talib.RSI(df['close'].values, timeperiod=14)
                    result['rsi_6'] = talib.RSI(df['close'].values, timeperiod=6)
                    result['cci'] = talib.CCI(df['high'].values, df['low'].values, df['close'].values)
                    result['williams_r'] = talib.WILLR(df['high'].values, df['low'].values, df['close'].values)
                    
                    # 趋势指标
                    result['adx'] = talib.ADX(df['high'].values, df['low'].values, df['close'].values)
                    result['aroon_up'], result['aroon_down'] = talib.AROON(df['high'].values, df['low'].values)
                    
                    # 波动率指标
                    result['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values)
                    result['natr'] = talib.NATR(df['high'].values, df['low'].values, df['close'].values)
                    
                    # 成交量指标
                    result['ad'] = talib.AD(df['high'].values, df['low'].values, df['close'].values, df['volume'].values)
                    result['obv'] = talib.OBV(df['close'].values, df['volume'].values)
                    
                except Exception as e:
                    logger.warning(f"TA-Lib计算失败: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"提取技术特征失败: {e}")
            return df
    
    def extract_price_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取价格形态特征"""
        try:
            result = df.copy()
            
            # K线形态特征
            result['doji'] = (abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8) < 0.1).astype(int)
            result['hammer'] = ((df['low'] < df['open']) & (df['low'] < df['close']) & 
                               ((df['high'] - np.maximum(df['open'], df['close'])) < 
                                (np.maximum(df['open'], df['close']) - df['low']) * 0.3)).astype(int)
            
            # 价格突破特征
            result['break_high_5'] = (df['close'] > df['high'].rolling(5).max().shift(1)).astype(int)
            result['break_high_10'] = (df['close'] > df['high'].rolling(10).max().shift(1)).astype(int)
            result['break_high_20'] = (df['close'] > df['high'].rolling(20).max().shift(1)).astype(int)
            
            # 支撑阻力特征
            result['near_resistance'] = (df['close'] / df['high'].rolling(20).max() > 0.95).astype(int)
            result['near_support'] = (df['close'] / df['low'].rolling(20).min() < 1.05).astype(int)
            
            # 连续涨跌特征
            price_change = df['close'].pct_change()
            result['consecutive_up'] = (price_change > 0).astype(int).groupby((price_change <= 0).cumsum()).cumsum()
            result['consecutive_down'] = (price_change < 0).astype(int).groupby((price_change >= 0).cumsum()).cumsum()
            
            # 缺口特征
            result['gap_up'] = (df['low'] > df['high'].shift(1)).astype(int)
            result['gap_down'] = (df['high'] < df['low'].shift(1)).astype(int)
            result['gap_size'] = np.where(result['gap_up'], 
                                        (df['low'] - df['high'].shift(1)) / df['close'].shift(1),
                                        np.where(result['gap_down'],
                                               (df['high'] - df['low'].shift(1)) / df['close'].shift(1), 0))
            
            return result
            
        except Exception as e:
            logger.error(f"提取价格形态特征失败: {e}")
            return df
    
    def extract_market_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取市场微观结构特征"""
        try:
            result = df.copy()
            
            # 价量关系特征
            result['price_volume_corr'] = df['close'].rolling(20).corr(df['volume'])
            result['volume_price_trend'] = np.sign(df['close'].diff()) * np.sign(df['volume'].diff())
            
            # 买卖压力特征
            result['buying_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
            result['selling_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-8)
            
            # 流动性特征
            result['turnover_volatility'] = df['turnover'].rolling(10).std() / df['turnover'].rolling(10).mean()
            result['illiquidity'] = abs(df['close'].pct_change()) / (df['amount'] + 1e-8)
            
            # 订单流特征（基于成交额和成交量的近似）
            avg_trade_size = df['amount'] / (df['volume'] + 1e-8)
            result['avg_trade_size'] = avg_trade_size
            result['avg_trade_size_ratio'] = avg_trade_size / avg_trade_size.rolling(20).mean()
            
            return result
            
        except Exception as e:
            logger.error(f"提取市场微观结构特征失败: {e}")
            return df
    
    def extract_cross_sectional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取横截面特征（相对于市场和行业的表现）"""
        try:
            result = df.copy()
            
            # 相对强度特征
            if 'market_return' in df.columns:
                result['relative_strength'] = df['change_pct'] - df['market_return']
                result['beta'] = df['change_pct'].rolling(60).corr(df['market_return']) * \
                               df['change_pct'].rolling(60).std() / df['market_return'].rolling(60).std()
            
            # 市值特征（如果有市值数据）
            if 'market_cap' in df.columns:
                result['market_cap_rank'] = df.groupby('date')['market_cap'].rank(pct=True)
                result['size_factor'] = np.log(df['market_cap'])
            
            # 行业相对表现（如果有行业数据）
            if 'industry' in df.columns:
                industry_return = df.groupby(['date', 'industry'])['change_pct'].transform('mean')
                result['industry_relative_return'] = df['change_pct'] - industry_return
            
            return result
            
        except Exception as e:
            logger.error(f"提取横截面特征失败: {e}")
            return df
    
    def extract_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取市场情绪特征"""
        try:
            result = df.copy()
            
            # 基于现有数据的情绪指标
            if 'limit_up_count' in df.columns:
                result['market_enthusiasm'] = df['limit_up_count'] / 100  # 标准化
            
            if 'sh_change_pct' in df.columns:
                result['market_trend'] = df['sh_change_pct']
                result['market_momentum'] = df['sh_change_pct'].rolling(5).mean()
            
            # VIX类似的恐慌指标（基于波动率）
            returns = df['close'].pct_change()
            result['fear_index'] = returns.rolling(20).std() * np.sqrt(252)  # 年化波动率
            
            # 市场广度指标
            if 'advance_decline_ratio' in df.columns:
                result['market_breadth'] = df['advance_decline_ratio']
            
            return result
            
        except Exception as e:
            logger.error(f"提取情绪特征失败: {e}")
            return df
    
    def create_sequence_features(self, df: pd.DataFrame, sequence_length: int = 30) -> np.ndarray:
        """创建时序特征序列"""
        try:
            # 选择数值型特征
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # 移除目标变量
            target_cols = [col for col in numeric_cols if 'future_return' in col or 'limit_up' in col or 'big_rise' in col]
            feature_cols = [col for col in numeric_cols if col not in target_cols and col != 'date']
            
            df_features = df[feature_cols].copy()
            
            # 处理缺失值
            df_features = df_features.fillna(method='ffill').fillna(0)
            
            sequences = []
            for stock in df['stock_code'].unique():
                stock_data = df_features[df['stock_code'] == stock].values
                
                if len(stock_data) >= sequence_length:
                    for i in range(sequence_length, len(stock_data)):
                        sequences.append(stock_data[i-sequence_length:i])
            
            return np.array(sequences)
            
        except Exception as e:
            logger.error(f"创建序列特征失败: {e}")
            return np.array([])
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, method: str = 'kbest', k: int = 50) -> pd.DataFrame:
        """特征选择"""
        try:
            if method == 'kbest':
                selector = SelectKBest(score_func=f_regression, k=k)
                X_selected = selector.fit_transform(X, y)
                
                selected_feature_names = X.columns[selector.get_support()].tolist()
                self.selected_features = selected_feature_names
                
                logger.info(f"选择了 {len(selected_feature_names)} 个特征")
                return pd.DataFrame(X_selected, columns=selected_feature_names, index=X.index)
            
        except Exception as e:
            logger.error(f"特征选择失败: {e}")
            return X
    
    def engineer_features(self, df: pd.DataFrame, mode: str = 'train') -> pd.DataFrame:
        """主要的特征工程函数"""
        logger.info("开始特征工程...")
        
        # 按股票分组处理
        result_dfs = []
        
        for stock_code in df['stock_code'].unique():
            stock_df = df[df['stock_code'] == stock_code].copy().sort_values('date')
            
            if len(stock_df) < 30:  # 数据太少跳过
                continue
            
            # 提取各类特征
            stock_df = self.extract_technical_features(stock_df)
            stock_df = self.extract_price_pattern_features(stock_df)
            stock_df = self.extract_market_microstructure_features(stock_df)
            stock_df = self.extract_cross_sectional_features(stock_df)
            stock_df = self.extract_sentiment_features(stock_df)
            
            result_dfs.append(stock_df)
        
        if not result_dfs:
            logger.error("没有有效的股票数据进行特征工程")
            return pd.DataFrame()
        
        result = pd.concat(result_dfs, ignore_index=True)
        
        # 处理无穷大和NaN值
        result = result.replace([np.inf, -np.inf], np.nan)
        result = result.fillna(method='ffill').fillna(0)
        
        logger.info(f"特征工程完成，共生成 {len(result.columns)} 个特征")
        
        return result
    
    def prepare_model_inputs(self, df: pd.DataFrame, target_col: str = 'limit_up_1d', 
                           sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """准备模型输入数据"""
        try:
            # 获取特征和标签
            feature_cols = [col for col in df.columns if col not in 
                          ['date', 'stock_code'] + [c for c in df.columns if 'future_return' in c or 'limit_up' in c or 'big_rise' in c]]
            
            X_sequences = []
            y_labels = []
            
            for stock_code in df['stock_code'].unique():
                stock_data = df[df['stock_code'] == stock_code].sort_values('date')
                
                if len(stock_data) < sequence_length + 1:
                    continue
                
                features = stock_data[feature_cols].values
                labels = stock_data[target_col].values
                
                # 创建序列
                for i in range(sequence_length, len(features)):
                    X_sequences.append(features[i-sequence_length:i])
                    y_labels.append(labels[i])
            
            X = np.array(X_sequences)
            y = np.array(y_labels)
            
            logger.info(f"准备模型输入完成: X shape {X.shape}, y shape {y.shape}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"准备模型输入失败: {e}")
            return np.array([]), np.array([])
    
    def normalize_features(self, X: np.ndarray, mode: str = 'train') -> np.ndarray:
        """标准化特征"""
        try:
            original_shape = X.shape
            X_reshaped = X.reshape(-1, X.shape[-1])
            
            if mode == 'train':
                X_scaled = self.scaler.fit_transform(X_reshaped)
            else:
                X_scaled = self.scaler.transform(X_reshaped)
            
            return X_scaled.reshape(original_shape)
            
        except Exception as e:
            logger.error(f"特征标准化失败: {e}")
            return X
