"""
回测报告生成器
"""
import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, Any
from datetime import datetime
import seaborn as sns

logger = logging.getLogger(__name__)


class ReportGenerator:
    """回测报告生成器类"""
    def __init__(self):
        """初始化报告生成器"""
        logger.info("初始化报告生成器")
        # 设置中文显示
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
        plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号
    
    def generate_report(self, backtest_results: Dict[str, Any], performance_metrics: Dict[str, float]) -> str:
        """生成回测报告
        
        Args:
            backtest_results: 回测结果字典
            performance_metrics: 性能指标字典
            
        Returns:
            报告文件路径
        """
        logger.info("开始生成回测报告")
        
        try:
            # 创建报告目录
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_dir = os.path.join('reports', f'backtest_report_{timestamp}')
            os.makedirs(report_dir, exist_ok=True)
            
            # 生成报告文件
            report_path = os.path.join(report_dir, 'backtest_report.md')
            
            # 生成图表
            charts_dir = os.path.join(report_dir, 'charts')
            os.makedirs(charts_dir, exist_ok=True)
            
            # 生成各个图表
            charts = {
                'equity_curve': self._generate_equity_curve_chart(backtest_results, charts_dir),
                'drawdown': self._generate_drawdown_chart(backtest_results, charts_dir),
                'daily_returns': self._generate_daily_returns_chart(backtest_results, charts_dir),
                'performance_metrics': self._generate_performance_metrics_chart(performance_metrics, charts_dir),
                'trade_distribution': self._generate_trade_distribution_chart(backtest_results, charts_dir)
            }
            
            # 生成报告内容
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(self._generate_report_content(backtest_results, performance_metrics, charts))
            
            logger.info(f"回测报告生成完成: {report_path}")
            
            return report_path
            
        except Exception as e:
            logger.error(f"生成回测报告失败: {str(e)}")
            return ''
    
    def _generate_report_content(self, backtest_results: Dict[str, Any], 
                                performance_metrics: Dict[str, float], 
                                charts: Dict[str, str]) -> str:
        """生成报告内容"""
        # 基本信息
        start_date = backtest_results.get('start_date', 'N/A')
        end_date = backtest_results.get('end_date', 'N/A')
        initial_capital = backtest_results.get('initial_capital', 0)
        final_equity = backtest_results.get('final_equity', 0)
        
        # 性能指标
        total_return = performance_metrics.get('total_return_pct', 0)
        annualized_return = performance_metrics.get('annualized_return_pct', 0)
        max_drawdown = performance_metrics.get('max_drawdown_pct', 0)
        sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
        sortino_ratio = performance_metrics.get('sortino_ratio', 0)
        calmar_ratio = performance_metrics.get('calmar_ratio', 0)
        total_trades = performance_metrics.get('total_trades', 0)
        win_rate = performance_metrics.get('win_rate_pct', 0)
        
        # 生成报告内容
        content = f"""# ETF量化投资策略回测报告

## 1. 回测概览

| 项目 | 数值 |
|------|------|
| 回测开始日期 | {start_date} |
| 回测结束日期 | {end_date} |
| 初始资金 | {initial_capital:.2f}元 |
| 最终资产 | {final_equity:.2f}元 |
| 总收益率 | {total_return:.2f}% |
| 年化收益率 | {annualized_return:.2f}% |
| 最大回撤 | {max_drawdown:.2f}% |
| 夏普比率 | {sharpe_ratio:.2f} |
| 索提诺比率 | {sortino_ratio:.2f} |
| 卡玛比率 | {calmar_ratio:.2f} |
| 总交易次数 | {total_trades} |
| 胜率 | {win_rate:.2f}% |

## 2. 资产曲线

![资产曲线]({charts.get('equity_curve', '')})

## 3. 回撤分析

![回撤分析]({charts.get('drawdown', '')})

## 4. 日收益率分布

![日收益率分布]({charts.get('daily_returns', '')})

## 5. 交易分布

![交易分布]({charts.get('trade_distribution', '')})

## 6. 性能指标详解

| 指标名称 | 数值 | 说明 |
|---------|------|------|
| 总收益率 | {total_return:.2f}% | 回测期间的总收益率 |
| 年化收益率 | {annualized_return:.2f}% | 将总收益率年化后的收益率 |
| 最大回撤 | {max_drawdown:.2f}% | 回测期间从高点到低点的最大跌幅 |
| 夏普比率 | {sharpe_ratio:.2f} | 每单位风险对应的超额收益，越高越好 |
| 索提诺比率 | {sortino_ratio:.2f} | 每单位下行风险对应的超额收益，越高越好 |
| 卡玛比率 | {calmar_ratio:.2f} | 年化收益率与最大回撤的比值，越高越好 |
| 胜率 | {win_rate:.2f}% | 盈利交易占总交易的比例 |
| 总交易次数 | {total_trades} | 回测期间的总交易次数 |

## 7. 总结与建议

### 7.1 总结

- 回测期间，策略取得了{total_return:.2f}%的总收益率，年化收益率为{annualized_return:.2f}%。
- 最大回撤为{max_drawdown:.2f}%，表明策略在极端市场情况下的风险控制能力。
- 夏普比率为{sharpe_ratio:.2f}，表明策略在承担单位风险的情况下能够获得较好的超额收益。
- 总共进行了{total_trades}笔交易，胜率为{win_rate:.2f}%。

### 7.2 建议

- 如果夏普比率较低，可以考虑优化策略以提高风险调整后收益。
- 如果最大回撤较大，可以考虑加入风险控制机制以降低极端损失。
- 建议在实盘前进行更多参数优化和敏感性分析。
- 可以考虑与其他策略进行组合，以分散风险。

---

报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return content
    
    def _generate_equity_curve_chart(self, backtest_results: Dict[str, Any], charts_dir: str) -> str:
        """生成资产曲线图"""
        try:
            equity_curve = backtest_results.get('equity_curve', [])
            if not equity_curve:
                return ''
            
            df = pd.DataFrame(equity_curve)
            df['date'] = pd.to_datetime(df['date'])
            
            plt.figure(figsize=(12, 6))
            plt.plot(df['date'], df['equity'])
            plt.title('资产曲线')
            plt.xlabel('日期')
            plt.ylabel('资产价值（元）')
            plt.grid(True)
            
            # 格式化x轴日期
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
            plt.gcf().autofmt_xdate()
            
            # 保存图表
            chart_path = os.path.join(charts_dir, 'equity_curve.png')
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return os.path.basename(chart_path)
            
        except Exception as e:
            logger.error(f"生成资产曲线图失败: {str(e)}")
            return ''
    
    def _generate_drawdown_chart(self, backtest_results: Dict[str, Any], charts_dir: str) -> str:
        """生成回撤图"""
        try:
            equity_curve = backtest_results.get('equity_curve', [])
            if not equity_curve:
                return ''
            
            df = pd.DataFrame(equity_curve)
            df['date'] = pd.to_datetime(df['date'])
            
            # 计算累计最大资产
            df['cumulative_max'] = df['equity'].cummax()
            
            # 计算回撤
            df['drawdown'] = (df['equity'] - df['cumulative_max']) / df['cumulative_max'] * 100
            
            plt.figure(figsize=(12, 6))
            plt.fill_between(df['date'], df['drawdown'], 0, color='red', alpha=0.3)
            plt.plot(df['date'], df['drawdown'], color='red')
            plt.title('回撤分析')
            plt.xlabel('日期')
            plt.ylabel('回撤百分比（%）')
            plt.grid(True)
            
            # 格式化x轴日期
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
            plt.gcf().autofmt_xdate()
            
            # 保存图表
            chart_path = os.path.join(charts_dir, 'drawdown.png')
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return os.path.basename(chart_path)
            
        except Exception as e:
            logger.error(f"生成回撤图失败: {str(e)}")
            return ''
    
    def _generate_daily_returns_chart(self, backtest_results: Dict[str, Any], charts_dir: str) -> str:
        """生成日收益率分布图"""
        try:
            equity_curve = backtest_results.get('equity_curve', [])
            if not equity_curve:
                return ''
            
            df = pd.DataFrame(equity_curve)
            
            # 计算日收益率
            df['daily_return_pct'] = df['equity'].pct_change() * 100
            
            plt.figure(figsize=(12, 6))
            sns.histplot(df['daily_return_pct'].dropna(), bins=50, kde=True)
            plt.title('日收益率分布')
            plt.xlabel('日收益率（%）')
            plt.ylabel('频率')
            plt.grid(True)
            
            # 保存图表
            chart_path = os.path.join(charts_dir, 'daily_returns.png')
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return os.path.basename(chart_path)
            
        except Exception as e:
            logger.error(f"生成日收益率分布图失败: {str(e)}")
            return ''
    
    def _generate_performance_metrics_chart(self, performance_metrics: Dict[str, float], charts_dir: str) -> str:
        """生成性能指标雷达图"""
        try:
            # 选择要显示的指标
            metrics = ['total_return_pct', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'win_rate_pct']
            labels = ['总收益率(%)', '夏普比率', '索提诺比率', '卡玛比率', '胜率(%)']
            
            # 获取指标值
            values = []
            for metric in metrics:
                # 对部分指标进行缩放，以便在雷达图上更好地显示
                if metric == 'total_return_pct':
                    # 将总收益率缩放到0-100
                    values.append(min(100, performance_metrics.get(metric, 0)))  
                elif metric == 'win_rate_pct':
                    # 胜率已经是0-100的范围
                    values.append(performance_metrics.get(metric, 0))
                else:
                    # 风险调整指标缩放到0-5
                    values.append(min(5, performance_metrics.get(metric, 0)))  
            
            # 创建雷达图
            plt.figure(figsize=(10, 8))
            
            # 计算角度
            angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
            
            # 闭合雷达图
            values = values + values[:1]
            angles = angles + angles[:1]
            labels = labels + labels[:1]
            
            # 绘制雷达图
            ax = plt.subplot(111, polar=True)
            ax.plot(angles, values, 'o-', linewidth=2)
            ax.fill(angles, values, alpha=0.25)
            
            # 设置标签
            ax.set_thetagrids(np.degrees(angles), labels)
            ax.set_title('性能指标雷达图')
            ax.grid(True)
            
            # 保存图表
            chart_path = os.path.join(charts_dir, 'performance_metrics.png')
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return os.path.basename(chart_path)
            
        except Exception as e:
            logger.error(f"生成性能指标雷达图失败: {str(e)}")
            return ''
    
    def _generate_trade_distribution_chart(self, backtest_results: Dict[str, Any], charts_dir: str) -> str:
        """生成交易分布图"""
        try:
            trades = backtest_results.get('trades', [])
            if not trades:
                return ''
            
            # 按月份统计交易次数
            df = pd.DataFrame(trades)
            df['date'] = pd.to_datetime(df['date'])
            
            # 按月分组统计买入和卖出交易次数
            df['month'] = df['date'].dt.to_period('M')
            
            # 计算买入和卖出交易次数
            buy_counts = df[df['action'] == 'buy'].groupby('month').size()
            sell_counts = df[df['action'] == 'sell'].groupby('month').size()
            
            # 转换为DataFrame
            monthly_counts = pd.DataFrame({
                'Buy': buy_counts,
                'Sell': sell_counts
            }).fillna(0)
            
            # 绘制柱状图
            plt.figure(figsize=(12, 6))
            monthly_counts.plot(kind='bar', ax=plt.gca())
            plt.title('交易分布（按月）')
            plt.xlabel('月份')
            plt.ylabel('交易次数')
            plt.grid(True, axis='y')
            plt.xticks(rotation=45)
            
            # 保存图表
            chart_path = os.path.join(charts_dir, 'trade_distribution.png')
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return os.path.basename(chart_path)
            
        except Exception as e:
            logger.error(f"生成交易分布图失败: {str(e)}")
            return ''
