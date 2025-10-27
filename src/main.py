"""
ETF量化投资分析系统主入口
"""
import logging
import os
from datetime import datetime

# 导入系统工作流
from src.workflow.etf_analysis_workflow import run_etf_analysis_workflow
from src.utils.logger_utils import setup_logger

# 设置日志
logger = setup_logger()

def main():
    """系统主入口函数"""
    logger.info("=" * 50)
    logger.info(f"开始运行ETF量化投资分析系统 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 50)

    try:
        # 运行系统工作流
        result = run_etf_analysis_workflow()

        # 打印运行结果摘要
        logger.info("=" * 50)
        logger.info("ETF量化投资分析系统运行摘要:")
        logger.info(f"状态: {result.get('status')}")

        # 打印各个模块的运行结果
        if result.get('data_acquisition'):
            data_result = result['data_acquisition']
            logger.info(f"数据抓取模块: 状态={data_result.get('status')}, 获取了{data_result.get('total_etfs', 0)}个ETF的数据")

        if result.get('trading_strategy'):
            strategy_result = result['trading_strategy']
            logger.info(f"交易策略模块: 状态={strategy_result.get('status')}, 生成了{strategy_result.get('buy_signals_count', 0)}个买入信号, {strategy_result.get('sell_signals_count', 0)}个卖出信号")

        if result.get('backtesting'):
            backtest_result = result['backtesting']
            logger.info(f"回测模块: 状态={backtest_result.get('status')}, 进行了{backtest_result.get('total_trades', 0)}笔交易")

            # 打印性能指标
            if backtest_result.get('performance_metrics'):
                metrics = backtest_result['performance_metrics']
                logger.info(f"回测性能: 总收益率={metrics.get('total_return_pct', 0):.2f}%, 最大回撤={metrics.get('max_drawdown_pct', 0):.2f}%, 夏普比率={metrics.get('sharpe_ratio', 0):.2f}")

        # 打印错误信息
        if result.get('errors'):
            logger.error(f"系统运行过程中出现{len(result['errors'])}个错误:")
            for i, error in enumerate(result['errors'], 1):
                logger.error(f"  {i}. {error}")

        # 打印报告路径
        if result.get('backtesting') and result['backtesting'].get('report_path'):
            logger.info(f"回测报告已生成: {result['backtesting']['report_path']}")

    except Exception as e:
        logger.error(f"系统运行过程中发生未捕获的异常: {str(e)}")

    finally:
        logger.info("=" * 50)
        logger.info(f"ETF量化投资分析系统运行结束 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 50)
