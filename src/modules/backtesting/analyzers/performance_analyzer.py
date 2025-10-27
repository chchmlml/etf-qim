"""
回测性能分析器
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any
import math

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """回测性能分析器类"""
    def __init__(self):
        """初始化性能分析器"""
        logger.info("初始化性能分析器")
        self.risk_free_rate = 0.02  # 无风险收益率，假设为2%
    
    def analyze_performance(self, backtest_results: Dict[str, Any]) -> Dict[str, float]:
        """分析回测性能
        
        Args:
            backtest_results: 回测结果字典
            
        Returns:
            性能指标字典
        """
        logger.info("开始分析回测性能")
        
        try:
            # 检查回测结果是否有效
            if backtest_results.get('status') != 'success':
                logger.warning("回测结果无效，无法分析性能")
                return {}
            
            # 获取资产曲线数据
            equity_curve = backtest_results.get('equity_curve', [])
            if not equity_curve:
                logger.warning("资产曲线数据为空，无法分析性能")
                return {}
            
            # 转换为DataFrame以便分析
            df = pd.DataFrame(equity_curve)
            df['date'] = pd.to_datetime(df['date'])
            
            # 计算核心性能指标
            performance_metrics = {
                'total_return_pct': self._calculate_total_return(df),
                'annualized_return_pct': self._calculate_annualized_return(df),
                'max_drawdown_pct': self._calculate_max_drawdown(df),
                'sharpe_ratio': self._calculate_sharpe_ratio(df),
                'sortino_ratio': self._calculate_sortino_ratio(df),
                'calmar_ratio': self._calculate_calmar_ratio(df),
                'win_rate_pct': self._calculate_win_rate(backtest_results.get('trades', [])),
                'avg_trade_return_pct': self._calculate_avg_trade_return(backtest_results.get('trades', [])),
                'total_trades': len(backtest_results.get('trades', [])),
                'avg_daily_return_pct': self._calculate_avg_daily_return(df),
                'std_daily_return_pct': self._calculate_std_daily_return(df),
                'best_day_pct': self._calculate_best_day(df),
                'worst_day_pct': self._calculate_worst_day(df),
                'profit_factor': self._calculate_profit_factor(backtest_results.get('trades', []))
            }
            
            logger.info(f"性能分析完成，总收益率={performance_metrics['total_return_pct']:.2f}%, "
                        f"最大回撤={performance_metrics['max_drawdown_pct']:.2f}%, "
                        f"夏普比率={performance_metrics['sharpe_ratio']:.2f}")
            
            return performance_metrics
            
        except Exception as e:
            logger.error(f"性能分析过程中发生错误: {str(e)}")
            return {}
    
    def _calculate_total_return(self, df: pd.DataFrame) -> float:
        """计算总收益率"""
        if len(df) < 2:
            return 0.0
        
        start_equity = df['equity'].iloc[0]
        end_equity = df['equity'].iloc[-1]
        return ((end_equity / start_equity) - 1) * 100
    
    def _calculate_annualized_return(self, df: pd.DataFrame) -> float:
        """计算年化收益率"""
        if len(df) < 2:
            return 0.0
        
        # 计算总收益率
        total_return = self._calculate_total_return(df) / 100
        
        # 计算回测天数
        days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
        
        # 计算年化收益率（假设一年252个交易日）
        if days > 0:
            annualized_return = ((1 + total_return) ** (252 / days)) - 1
            return annualized_return * 100
        
        return 0.0
    
    def _calculate_max_drawdown(self, df: pd.DataFrame) -> float:
        """计算最大回撤"""
        if len(df) < 2:
            return 0.0
        
        # 计算累计最大资产
        df['cumulative_max'] = df['equity'].cummax()
        
        # 计算回撤
        df['drawdown'] = (df['equity'] - df['cumulative_max']) / df['cumulative_max']
        
        # 找到最大回撤
        max_drawdown = df['drawdown'].min()
        
        return max_drawdown * 100
    
    def _calculate_sharpe_ratio(self, df: pd.DataFrame) -> float:
        """计算夏普比率"""
        if len(df) < 2:
            return 0.0
        
        # 计算日收益率
        daily_returns = df['equity'].pct_change().dropna()
        
        if len(daily_returns) == 0:
            return 0.0
        
        # 计算日均收益率和标准差
        avg_daily_return = daily_returns.mean()
        std_daily_return = daily_returns.std()
        
        if std_daily_return == 0:
            return 0.0
        
        # 计算年化夏普比率
        # 无风险收益率转换为日收益率
        daily_risk_free_rate = self.risk_free_rate / 252
        
        # 计算超额收益率
        excess_daily_return = avg_daily_return - daily_risk_free_rate
        
        # 计算夏普比率
        sharpe_ratio = excess_daily_return / std_daily_return * math.sqrt(252)
        
        return sharpe_ratio
    
    def _calculate_sortino_ratio(self, df: pd.DataFrame) -> float:
        """计算索提诺比率"""
        if len(df) < 2:
            return 0.0
        
        # 计算日收益率
        daily_returns = df['equity'].pct_change().dropna()
        
        if len(daily_returns) == 0:
            return 0.0
        
        # 计算日均收益率
        avg_daily_return = daily_returns.mean()
        
        # 计算下行风险（只考虑负收益）
        negative_returns = daily_returns[daily_returns < 0]
        if len(negative_returns) == 0:
            downside_risk = 0.0
        else:
            downside_risk = negative_returns.std()
        
        if downside_risk == 0:
            return 0.0
        
        # 计算年化索提诺比率
        # 无风险收益率转换为日收益率
        daily_risk_free_rate = self.risk_free_rate / 252
        
        # 计算超额收益率
        excess_daily_return = avg_daily_return - daily_risk_free_rate
        
        # 计算索提诺比率
        sortino_ratio = excess_daily_return / downside_risk * math.sqrt(252)
        
        return sortino_ratio
    
    def _calculate_calmar_ratio(self, df: pd.DataFrame) -> float:
        """计算卡玛比率"""
        # 计算年化收益率
        annualized_return = self._calculate_annualized_return(df) / 100
        
        # 计算最大回撤
        max_drawdown = abs(self._calculate_max_drawdown(df) / 100)
        
        if max_drawdown == 0:
            return 0.0
        
        # 计算卡玛比率
        calmar_ratio = annualized_return / max_drawdown
        
        return calmar_ratio
    
    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        """计算胜率"""
        if not trades:
            return 0.0
        
        # 计算盈利交易的数量
        winning_trades = [trade for trade in trades if trade['action'] == 'sell' and trade.get('profit', 0) > 0]
        
        return (len(winning_trades) / len(trades)) * 100 if len(trades) > 0 else 0.0
    
    def _calculate_avg_trade_return(self, trades: List[Dict]) -> float:
        """计算平均每笔交易收益率"""
        if not trades:
            return 0.0
        
        # 计算所有卖出交易的收益率
        sell_trades = [trade for trade in trades if trade['action'] == 'sell']
        
        if not sell_trades:
            return 0.0
        
        # 计算平均收益率
        total_return = sum(trade.get('profit_pct', 0) for trade in sell_trades)
        
        return total_return / len(sell_trades)
    
    def _calculate_avg_daily_return(self, df: pd.DataFrame) -> float:
        """计算日均收益率"""
        if len(df) < 2:
            return 0.0
        
        # 计算日收益率
        daily_returns = df['equity'].pct_change().dropna()
        
        return daily_returns.mean() * 100 if len(daily_returns) > 0 else 0.0
    
    def _calculate_std_daily_return(self, df: pd.DataFrame) -> float:
        """计算日均收益率标准差"""
        if len(df) < 2:
            return 0.0
        
        # 计算日收益率
        daily_returns = df['equity'].pct_change().dropna()
        
        return daily_returns.std() * 100 if len(daily_returns) > 0 else 0.0
    
    def _calculate_best_day(self, df: pd.DataFrame) -> float:
        """计算最佳单日收益率"""
        if len(df) < 2:
            return 0.0
        
        # 计算日收益率
        daily_returns = df['equity'].pct_change().dropna()
        
        if len(daily_returns) == 0:
            return 0.0
        
        best_day_return = daily_returns.max()
        
        return best_day_return * 100
    
    def _calculate_worst_day(self, df: pd.DataFrame) -> float:
        """计算最差单日收益率"""
        if len(df) < 2:
            return 0.0
        
        # 计算日收益率
        daily_returns = df['equity'].pct_change().dropna()
        
        if len(daily_returns) == 0:
            return 0.0
        
        worst_day_return = daily_returns.min()
        
        return worst_day_return * 100
    
    def _calculate_profit_factor(self, trades: List[Dict]) -> float:
        """计算盈利因子"""
        if not trades:
            return 0.0
        
        # 计算所有卖出交易的盈利和亏损
        sell_trades = [trade for trade in trades if trade['action'] == 'sell']
        
        if not sell_trades:
            return 0.0
        
        # 计算总盈利和总亏损
        total_profit = sum(abs(trade.get('profit', 0)) for trade in sell_trades if trade.get('profit', 0) > 0)
        total_loss = sum(abs(trade.get('profit', 0)) for trade in sell_trades if trade.get('profit', 0) < 0)
        
        if total_loss == 0:
            return 0.0
        
        # 计算盈利因子
        profit_factor = total_profit / total_loss
        
        return profit_factor
