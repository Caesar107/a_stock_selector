# -*- coding: utf-8 -*-
"""
策略回测器
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class Backtester:
    """策略回测器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.backtest_config = config.get('backtest', {})
        
        # 回测参数
        self.initial_capital = self.backtest_config.get('initial_capital', 1000000)  # 初始资金100万
        self.commission_rate = self.backtest_config.get('commission_rate', 0.0003)  # 手续费
        self.max_position_size = self.backtest_config.get('max_position_size', 0.1)  # 单只股票最大仓位
        self.max_stocks = self.backtest_config.get('max_stocks', 10)  # 最大持仓数量
        self.holding_period = self.backtest_config.get('holding_period', 1)  # 持仓天数
        
        # 回测结果
        self.portfolio_history = []
        self.trade_history = []
        self.daily_returns = []
        self.benchmark_returns = []
        
    def run_backtest(self, predictions_data: pd.DataFrame, 
                    price_data: pd.DataFrame, 
                    start_date: str, end_date: str) -> Dict:
        """运行策略回测"""
        logger.info(f"开始回测，时间范围: {start_date} 到 {end_date}")
        
        # 初始化组合
        portfolio = {
            'cash': self.initial_capital,
            'positions': {},  # {stock_code: {'shares': int, 'avg_price': float, 'entry_date': str}}
            'total_value': self.initial_capital
        }
        
        # 按日期遍历
        date_range = pd.date_range(start_date, end_date, freq='D')
        trading_dates = [d for d in date_range if d.weekday() < 5]  # 只考虑工作日
        
        for current_date in trading_dates:
            current_date_str = current_date.strftime('%Y-%m-%d')
            
            # 更新持仓价值
            portfolio = self._update_portfolio_value(portfolio, price_data, current_date_str)
            
            # 处理卖出（到期平仓）
            portfolio = self._process_sells(portfolio, price_data, current_date, current_date_str)
            
            # 获取当日预测信号
            daily_predictions = self._get_daily_predictions(predictions_data, current_date_str)
            
            if not daily_predictions.empty:
                # 处理买入
                portfolio = self._process_buys(portfolio, daily_predictions, price_data, current_date_str)
            
            # 记录当日组合状态
            self._record_portfolio_state(portfolio, current_date_str)
            
            # 计算基准收益
            benchmark_return = self._calculate_benchmark_return(price_data, current_date_str)
            self.benchmark_returns.append(benchmark_return)
        
        # 计算回测结果
        results = self._calculate_backtest_results()
        
        logger.info("回测完成")
        return results
    
    def _get_daily_predictions(self, predictions_data: pd.DataFrame, date: str) -> pd.DataFrame:
        """获取指定日期的预测结果"""
        try:
            daily_data = predictions_data[predictions_data['date'] == date].copy()
            
            # 按预测概率排序
            if 'limit_up_probability' in daily_data.columns:
                daily_data = daily_data.sort_values('limit_up_probability', ascending=False)
            elif 'prediction_score' in daily_data.columns:
                daily_data = daily_data.sort_values('prediction_score', ascending=False)
            
            return daily_data.head(self.max_stocks * 2)  # 获取更多候选以防有些股票无法买入
            
        except Exception as e:
            logger.error(f"获取日期 {date} 的预测数据失败: {e}")
            return pd.DataFrame()
    
    def _update_portfolio_value(self, portfolio: Dict, price_data: pd.DataFrame, date: str) -> Dict:
        """更新组合市值"""
        total_position_value = 0
        
        for stock_code, position in portfolio['positions'].items():
            try:
                # 获取当前价格
                current_price = self._get_stock_price(price_data, stock_code, date)
                if current_price is not None:
                    position_value = position['shares'] * current_price
                    total_position_value += position_value
                    position['current_price'] = current_price
                    position['market_value'] = position_value
                    position['unrealized_pnl'] = position_value - position['shares'] * position['avg_price']
                    
            except Exception as e:
                logger.warning(f"更新股票 {stock_code} 价格失败: {e}")
        
        portfolio['total_value'] = portfolio['cash'] + total_position_value
        return portfolio
    
    def _process_sells(self, portfolio: Dict, price_data: pd.DataFrame, 
                      current_date: datetime, current_date_str: str) -> Dict:
        """处理卖出操作"""
        to_sell = []
        
        for stock_code, position in portfolio['positions'].items():
            # 检查是否到达持仓期限
            entry_date = datetime.strptime(position['entry_date'], '%Y-%m-%d')
            holding_days = (current_date - entry_date).days
            
            if holding_days >= self.holding_period:
                to_sell.append(stock_code)
        
        # 执行卖出
        for stock_code in to_sell:
            portfolio = self._sell_stock(portfolio, stock_code, price_data, current_date_str)
        
        return portfolio
    
    def _process_buys(self, portfolio: Dict, predictions: pd.DataFrame, 
                     price_data: pd.DataFrame, date: str) -> Dict:
        """处理买入操作"""
        available_cash = portfolio['cash']
        current_positions = len(portfolio['positions'])
        
        for _, stock in predictions.iterrows():
            stock_code = stock['stock_code']
            
            # 检查是否已持有
            if stock_code in portfolio['positions']:
                continue
            
            # 检查最大持仓数量
            if current_positions >= self.max_stocks:
                break
            
            # 检查是否有足够现金
            stock_price = self._get_stock_price(price_data, stock_code, date)
            if stock_price is None:
                continue
            
            # 计算买入数量
            position_size = min(
                available_cash * self.max_position_size,
                available_cash / (self.max_stocks - current_positions)
            )
            
            shares = int(position_size / stock_price / 100) * 100  # 买入整手
            if shares < 100:  # 至少买入1手
                continue
            
            # 执行买入
            success = self._buy_stock(portfolio, stock_code, shares, stock_price, date)
            if success:
                available_cash = portfolio['cash']
                current_positions += 1
        
        return portfolio
    
    def _buy_stock(self, portfolio: Dict, stock_code: str, shares: int, 
                   price: float, date: str) -> bool:
        """买入股票"""
        try:
            total_cost = shares * price
            commission = total_cost * self.commission_rate
            total_amount = total_cost + commission
            
            if portfolio['cash'] < total_amount:
                return False
            
            # 更新现金
            portfolio['cash'] -= total_amount
            
            # 添加持仓
            portfolio['positions'][stock_code] = {
                'shares': shares,
                'avg_price': price,
                'entry_date': date,
                'current_price': price,
                'market_value': total_cost,
                'unrealized_pnl': 0
            }
            
            # 记录交易
            self.trade_history.append({
                'date': date,
                'stock_code': stock_code,
                'action': 'buy',
                'shares': shares,
                'price': price,
                'amount': total_amount,
                'commission': commission
            })
            
            logger.debug(f"{date}: 买入 {stock_code} {shares}股，价格 {price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"买入股票 {stock_code} 失败: {e}")
            return False
    
    def _sell_stock(self, portfolio: Dict, stock_code: str, 
                   price_data: pd.DataFrame, date: str) -> Dict:
        """卖出股票"""
        try:
            if stock_code not in portfolio['positions']:
                return portfolio
            
            position = portfolio['positions'][stock_code]
            shares = position['shares']
            sell_price = self._get_stock_price(price_data, stock_code, date)
            
            if sell_price is None:
                logger.warning(f"无法获取 {stock_code} 在 {date} 的价格，跳过卖出")
                return portfolio
            
            total_proceeds = shares * sell_price
            commission = total_proceeds * self.commission_rate
            net_proceeds = total_proceeds - commission
            
            # 更新现金
            portfolio['cash'] += net_proceeds
            
            # 计算盈亏
            cost_basis = shares * position['avg_price']
            realized_pnl = net_proceeds - cost_basis
            
            # 记录交易
            self.trade_history.append({
                'date': date,
                'stock_code': stock_code,
                'action': 'sell',
                'shares': shares,
                'price': sell_price,
                'amount': net_proceeds,
                'commission': commission,
                'realized_pnl': realized_pnl,
                'return_pct': realized_pnl / cost_basis if cost_basis > 0 else 0
            })
            
            # 移除持仓
            del portfolio['positions'][stock_code]
            
            logger.debug(f"{date}: 卖出 {stock_code} {shares}股，价格 {sell_price:.2f}，收益 {realized_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"卖出股票 {stock_code} 失败: {e}")
        
        return portfolio
    
    def _get_stock_price(self, price_data: pd.DataFrame, stock_code: str, date: str) -> Optional[float]:
        """获取股票价格"""
        try:
            stock_data = price_data[
                (price_data['stock_code'] == stock_code) & 
                (price_data['date'] == date)
            ]
            
            if stock_data.empty:
                return None
            
            return float(stock_data.iloc[0]['close'])
            
        except Exception as e:
            logger.error(f"获取股票 {stock_code} 在 {date} 的价格失败: {e}")
            return None
    
    def _calculate_benchmark_return(self, price_data: pd.DataFrame, date: str) -> float:
        """计算基准收益率（沪深300或上证指数）"""
        try:
            # 这里简化为计算所有股票的平均收益
            daily_data = price_data[price_data['date'] == date]
            if daily_data.empty:
                return 0.0
            
            avg_return = daily_data['change_pct'].mean() / 100
            return avg_return
            
        except Exception as e:
            logger.error(f"计算基准收益失败: {e}")
            return 0.0
    
    def _record_portfolio_state(self, portfolio: Dict, date: str):
        """记录组合状态"""
        portfolio_state = {
            'date': date,
            'cash': portfolio['cash'],
            'total_value': portfolio['total_value'],
            'positions_count': len(portfolio['positions']),
            'position_value': portfolio['total_value'] - portfolio['cash']
        }
        
        self.portfolio_history.append(portfolio_state)
        
        # 计算日收益率
        if len(self.portfolio_history) > 1:
            prev_value = self.portfolio_history[-2]['total_value']
            current_value = portfolio['total_value']
            daily_return = (current_value - prev_value) / prev_value
            self.daily_returns.append(daily_return)
        else:
            self.daily_returns.append(0)
    
    def _calculate_backtest_results(self) -> Dict:
        """计算回测结果指标"""
        try:
            # 基础统计
            total_return = (self.portfolio_history[-1]['total_value'] - self.initial_capital) / self.initial_capital
            
            daily_returns_array = np.array(self.daily_returns)
            benchmark_returns_array = np.array(self.benchmark_returns)
            
            # 年化收益率
            trading_days = len(daily_returns_array)
            annualized_return = (1 + total_return) ** (252 / trading_days) - 1
            
            # 波动率
            volatility = np.std(daily_returns_array) * np.sqrt(252)
            
            # 夏普比率（假设无风险利率为3%）
            risk_free_rate = 0.03
            sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # 最大回撤
            portfolio_values = [p['total_value'] for p in self.portfolio_history]
            max_drawdown = self._calculate_max_drawdown(portfolio_values)
            
            # 胜率
            winning_trades = [t for t in self.trade_history if t.get('realized_pnl', 0) > 0]
            total_trades = len([t for t in self.trade_history if t['action'] == 'sell'])
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
            
            # 基准比较
            benchmark_return = np.sum(benchmark_returns_array)
            alpha = total_return - benchmark_return
            
            # 计算Beta
            if len(benchmark_returns_array) > 1:
                beta = np.cov(daily_returns_array, benchmark_returns_array)[0][1] / np.var(benchmark_returns_array)
            else:
                beta = 1.0
            
            results = {
                'performance': {
                    'total_return': total_return,
                    'annualized_return': annualized_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'alpha': alpha,
                    'beta': beta
                },
                'trading': {
                    'total_trades': total_trades,
                    'win_rate': win_rate,
                    'avg_holding_period': self.holding_period,
                    'avg_position_size': self.max_position_size
                },
                'risk': {
                    'value_at_risk_95': np.percentile(daily_returns_array, 5),
                    'expected_shortfall_95': np.mean(daily_returns_array[daily_returns_array <= np.percentile(daily_returns_array, 5)]),
                    'downside_deviation': np.std(daily_returns_array[daily_returns_array < 0]) * np.sqrt(252)
                }
            }
            
            logger.info(f"回测完成，总收益率: {total_return:.2%}，夏普比率: {sharpe_ratio:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"计算回测结果失败: {e}")
            return {}
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """计算最大回撤"""
        peak = portfolio_values[0]
        max_dd = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd
    
    def generate_report(self, results: Dict):
        """生成回测报告"""
        try:
            # 创建报告目录
            report_dir = Path('logs/backtest_reports')
            report_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 保存JSON结果
            with open(report_dir / f'backtest_results_{timestamp}.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # 生成图表
            self._generate_performance_charts(report_dir, timestamp)
            
            # 生成HTML报告
            self._generate_html_report(results, report_dir, timestamp)
            
            logger.info(f"回测报告已生成: {report_dir}")
            
        except Exception as e:
            logger.error(f"生成回测报告失败: {e}")
    
    def _generate_performance_charts(self, report_dir: Path, timestamp: str):
        """生成表现图表"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 组合价值曲线
            dates = [p['date'] for p in self.portfolio_history]
            values = [p['total_value'] for p in self.portfolio_history]
            
            axes[0, 0].plot(pd.to_datetime(dates), values, label='组合价值', linewidth=2)
            axes[0, 0].axhline(y=self.initial_capital, color='r', linestyle='--', label='初始资金')
            axes[0, 0].set_title('组合价值走势')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # 日收益率分布
            axes[0, 1].hist(self.daily_returns, bins=50, alpha=0.7, color='blue')
            axes[0, 1].axvline(x=np.mean(self.daily_returns), color='r', linestyle='--', label='均值')
            axes[0, 1].set_title('日收益率分布')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # 累计收益对比
            cumulative_strategy = np.cumprod(1 + np.array(self.daily_returns))
            cumulative_benchmark = np.cumprod(1 + np.array(self.benchmark_returns))
            
            axes[1, 0].plot(pd.to_datetime(dates[1:]), cumulative_strategy, label='策略', linewidth=2)
            axes[1, 0].plot(pd.to_datetime(dates[1:]), cumulative_benchmark, label='基准', linewidth=2)
            axes[1, 0].set_title('累计收益对比')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # 回撤曲线
            portfolio_values = [p['total_value'] for p in self.portfolio_history]
            peak_values = []
            drawdowns = []
            peak = portfolio_values[0]
            
            for value in portfolio_values:
                if value > peak:
                    peak = value
                peak_values.append(peak)
                drawdowns.append((value - peak) / peak)
            
            axes[1, 1].fill_between(pd.to_datetime(dates), drawdowns, 0, alpha=0.3, color='red')
            axes[1, 1].plot(pd.to_datetime(dates), drawdowns, color='red', linewidth=1)
            axes[1, 1].set_title('回撤曲线')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig(report_dir / f'performance_charts_{timestamp}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 交易分析图
            if self.trade_history:
                self._generate_trading_charts(report_dir, timestamp)
            
        except Exception as e:
            logger.error(f"生成性能图表失败: {e}")
    
    def _generate_trading_charts(self, report_dir: Path, timestamp: str):
        """生成交易分析图表"""
        try:
            trades_df = pd.DataFrame(self.trade_history)
            sell_trades = trades_df[trades_df['action'] == 'sell'].copy()
            
            if sell_trades.empty:
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 单笔收益分布
            if 'realized_pnl' in sell_trades.columns:
                axes[0, 0].hist(sell_trades['realized_pnl'], bins=30, alpha=0.7, color='green')
                axes[0, 0].axvline(x=0, color='r', linestyle='--')
                axes[0, 0].set_title('单笔交易盈亏分布')
                axes[0, 0].set_xlabel('盈亏金额')
                axes[0, 0].grid(True)
            
            # 收益率分布
            if 'return_pct' in sell_trades.columns:
                axes[0, 1].hist(sell_trades['return_pct'], bins=30, alpha=0.7, color='blue')
                axes[0, 1].axvline(x=0, color='r', linestyle='--')
                axes[0, 1].set_title('单笔交易收益率分布')
                axes[0, 1].set_xlabel('收益率')
                axes[0, 1].grid(True)
            
            # 月度交易统计
            sell_trades['date'] = pd.to_datetime(sell_trades['date'])
            monthly_stats = sell_trades.groupby(sell_trades['date'].dt.to_period('M')).agg({
                'realized_pnl': 'sum',
                'stock_code': 'count'
            }).rename(columns={'stock_code': 'trade_count'})
            
            axes[1, 0].bar(range(len(monthly_stats)), monthly_stats['realized_pnl'], alpha=0.7)
            axes[1, 0].set_title('月度盈亏')
            axes[1, 0].set_xlabel('月份')
            axes[1, 0].set_ylabel('盈亏金额')
            axes[1, 0].grid(True)
            
            # 胜率统计
            winning_trades = sell_trades[sell_trades['realized_pnl'] > 0]
            losing_trades = sell_trades[sell_trades['realized_pnl'] <= 0]
            
            labels = ['盈利交易', '亏损交易']
            sizes = [len(winning_trades), len(losing_trades)]
            colors = ['green', 'red']
            
            axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
            axes[1, 1].set_title('交易胜负比')
            
            plt.tight_layout()
            plt.savefig(report_dir / f'trading_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"生成交易图表失败: {e}")
    
    def _generate_html_report(self, results: Dict, report_dir: Path, timestamp: str):
        """生成HTML回测报告"""
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>A股选股策略回测报告</title>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ text-align: center; color: #333; }}
                    .section {{ margin: 30px 0; }}
                    .metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }}
                    .metric-box {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
                    .metric-title {{ font-weight: bold; color: #666; }}
                    .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
                    .positive {{ color: #27ae60; }}
                    .negative {{ color: #e74c3c; }}
                    table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .chart {{ text-align: center; margin: 20px 0; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>A股选股策略回测报告</h1>
                    <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="section">
                    <h2>核心指标</h2>
                    <div class="metrics">
                        <div class="metric-box">
                            <div class="metric-title">总收益率</div>
                            <div class="metric-value {'positive' if results.get('performance', {}).get('total_return', 0) > 0 else 'negative'}">
                                {results.get('performance', {}).get('total_return', 0):.2%}
                            </div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-title">年化收益率</div>
                            <div class="metric-value {'positive' if results.get('performance', {}).get('annualized_return', 0) > 0 else 'negative'}">
                                {results.get('performance', {}).get('annualized_return', 0):.2%}
                            </div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-title">夏普比率</div>
                            <div class="metric-value {'positive' if results.get('performance', {}).get('sharpe_ratio', 0) > 0 else 'negative'}">
                                {results.get('performance', {}).get('sharpe_ratio', 0):.2f}
                            </div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-title">最大回撤</div>
                            <div class="metric-value negative">
                                {results.get('performance', {}).get('max_drawdown', 0):.2%}
                            </div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-title">胜率</div>
                            <div class="metric-value">
                                {results.get('trading', {}).get('win_rate', 0):.2%}
                            </div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-title">交易次数</div>
                            <div class="metric-value">
                                {results.get('trading', {}).get('total_trades', 0)}
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>详细统计</h2>
                    <table>
                        <tr><th>指标</th><th>数值</th></tr>
                        <tr><td>波动率</td><td>{results.get('performance', {}).get('volatility', 0):.2%}</td></tr>
                        <tr><td>Alpha</td><td>{results.get('performance', {}).get('alpha', 0):.2%}</td></tr>
                        <tr><td>Beta</td><td>{results.get('performance', {}).get('beta', 0):.2f}</td></tr>
                        <tr><td>95% VaR</td><td>{results.get('risk', {}).get('value_at_risk_95', 0):.2%}</td></tr>
                        <tr><td>下行偏差</td><td>{results.get('risk', {}).get('downside_deviation', 0):.2%}</td></tr>
                    </table>
                </div>
                
                <div class="section">
                    <h2>图表分析</h2>
                    <div class="chart">
                        <img src="performance_charts_{timestamp}.png" alt="Performance Charts" style="max-width: 100%;">
                    </div>
                    <div class="chart">
                        <img src="trading_analysis_{timestamp}.png" alt="Trading Analysis" style="max-width: 100%;">
                    </div>
                </div>
                
            </body>
            </html>
            """
            
            with open(report_dir / f'backtest_report_{timestamp}.html', 'w', encoding='utf-8') as f:
                f.write(html_content)
            
        except Exception as e:
            logger.error(f"生成HTML报告失败: {e}")
