"""
回测模块的LangGraph工作流
根据数据抓取生成模块使用交易策略模块进行回测数据，用买入和卖出策略进行组合回测，数据生成报表
"""
import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, List, Any, Optional
from langgraph.graph import StateGraph, START, END
from datetime import datetime, timedelta

# 导入相关模块
from src.utils.data_utils import load_dataframe, load_json, save_dataframe
from src.modules.backtesting.engine.backtest_engine import BacktestEngine
from src.modules.backtesting.analyzers.performance_analyzer import PerformanceAnalyzer
from src.modules.backtesting.reporters.report_generator import ReportGenerator

logger = logging.getLogger(__name__)

# 定义工作流状态
class BacktestingState:
    """回测模块的状态类"""
    def __init__(self):
        self.etf_data_dict: Dict[str, pd.DataFrame] = {}
        self.buy_signals: Dict[str, List[Dict]] = {}
        self.sell_signals: Dict[str, List[Dict]] = {}
        self.backtest_results: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, float] = {}
        self.report_path: Optional[str] = None
        self.errors: List[str] = []

# 定义工作流节点函数
def load_backtest_data_node(state: BacktestingState) -u003e Dict[str, Any]:
    """加载回测数据节点"""
    logger.info("开始加载回测数据")
    try:
        # 查找最新的PE数据
        data_dirs = [d for d in os.listdir(os.path.join('data', 'raw')) 
                     if d.startswith('etf_pe_data_')]
        if not data_dirs:
            error_msg = "没有找到PE数据"
            logger.error(error_msg)
            errors = state.errors.copy()
            errors.append(error_msg)
            return {"errors": errors}
        
        latest_data_dir = max(data_dirs)
        data_path = os.path.join('data', 'raw', latest_data_dir)
        
        # 加载所有ETF的PE数据
        etf_data_dict = {}
        pe_files = [f for f in os.listdir(data_path) if f.endswith('_pe.csv')]
        
        for file in pe_files:
            code = file.split('_')[0]
            file_path = os.path.join(data_path, file)
            pe_data = load_dataframe(file_path)
            
            # 为回测准备数据
            if 'date' in pe_data.columns and 'pe' in pe_data.columns:
                # 确保日期是datetime类型
                if not pd.api.types.is_datetime64_any_dtype(pe_data['date']):
                    pe_data['date'] = pd.to_datetime(pe_data['date'])
                
                # 排序并保留所需列
                pe_data = pe_data.sort_values('date')
                etf_data_dict[code] = pe_data
        
        logger.info(f"成功加载{len(etf_data_dict)}个ETF的回测数据")
        
        return {
            "etf_data_dict": etf_data_dict,
            "errors": state.errors.copy()
        }
    except Exception as e:
        error_msg = f"加载回测数据失败: {str(e)}"
        logger.error(error_msg)
        errors = state.errors.copy()
        errors.append(error_msg)
        return {"errors": errors}

def load_trading_signals_node(state: BacktestingState) -u003e Dict[str, Any]:
    """加载交易信号节点"""
    logger.info("开始加载交易信号")
    if not state.etf_data_dict:
        error_msg = "没有ETF数据，无法加载交易信号"
        logger.error(error_msg)
        errors = state.errors.copy()
        errors.append(error_msg)
        return {"errors": errors}
    
    try:
        # 模拟交易信号数据（实际应用中应该从交易策略模块的输出中加载）
        buy_signals = {}
        sell_signals = {}
        
        # 为演示目的，我们根据PE分位点生成一些模拟的交易信号
        # 实际应用中，这里应该加载交易策略模块生成的信号
        for code, data in state.etf_data_dict.items():
            # 计算PE分位点
            valid_pe = data['pe'].dropna()
            if len(valid_pe) u003e 0:
                # 为每个ETF生成一些买入和卖出信号
                buy_signals_list = []
                sell_signals_list = []
                
                # 遍历数据生成信号
                for i in range(1, len(data)):
                    current_date = data.iloc[i]['date']
                    current_pe = data.iloc[i]['pe']
                    
                    if pd.isna(current_pe):
                        continue
                    
                    # 计算当前PE的分位点（简单计算，实际应用中可能需要更复杂的方法）
                    percentile = np.sum(valid_pe u003c= current_pe) / len(valid_pe) if len(valid_pe) u003e 0 else 0.5
                    
                    # 低估值时生成买入信号
                    if percentile u003c= 0.2 and i % 10 == 0:  # 每10个交易日检查一次
                        buy_signals_list.append({
                            'date': current_date.strftime('%Y-%m-%d'),
                            'price': current_pe,  # 这里用PE作为价格的模拟
                            'volume': 1000,  # 模拟买入数量
                            'signal_strength': 1.0 - percentile
                        })
                    
                    # 高估值时生成卖出信号
                    elif percentile u003e= 0.8 and i % 10 == 0:  # 每10个交易日检查一次
                        sell_signals_list.append({
                            'date': current_date.strftime('%Y-%m-%d'),
                            'price': current_pe,  # 这里用PE作为价格的模拟
                            'volume': 500,  # 模拟卖出数量
                            'signal_strength': percentile - 0.5
                        })
                
                buy_signals[code] = buy_signals_list
                sell_signals[code] = sell_signals_list
        
        logger.info(f"成功加载交易信号: 买入信号={sum(len(s) for s in buy_signals.values())}, "
                    f"卖出信号={sum(len(s) for s in sell_signals.values())}")
        
        return {
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "errors": state.errors.copy()
        }
    except Exception as e:
        error_msg = f"加载交易信号失败: {str(e)}"
        logger.error(error_msg)
        errors = state.errors.copy()
        errors.append(error_msg)
        return {"errors": errors}

def run_backtest_node(state: BacktestingState) -u003e Dict[str, Any]:
    """运行回测节点"""
    logger.info("开始运行回测")
    if not state.etf_data_dict or not state.buy_signals or not state.sell_signals:
        error_msg = "缺少回测数据或交易信号，无法运行回测"
        logger.error(error_msg)
        errors = state.errors.copy()
        errors.append(error_msg)
        return {"errors": errors}
    
    try:
        # 创建回测引擎实例
        backtest_engine = BacktestEngine()
        
        # 运行回测
        backtest_results = backtest_engine.run_backtest(
            etf_data_dict=state.etf_data_dict,
            buy_signals=state.buy_signals,
            sell_signals=state.sell_signals,
            initial_capital=100000  # 初始资金10万元
        )
        
        logger.info(f"回测运行完成，总交易日数: {backtest_results.get('total_days', 0)}")
        
        # 保存回测结果
        backtest_dir = os.path.join('data', 'backtest_results', datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(backtest_dir, exist_ok=True)
        
        # 保存交易记录
        if 'trades' in backtest_results:
            trades_df = pd.DataFrame(backtest_results['trades'])
            trades_df.to_csv(os.path.join(backtest_dir, 'trades.csv'), index=False)
        
        # 保存资产曲线
        if 'equity_curve' in backtest_results:
            equity_df = pd.DataFrame(backtest_results['equity_curve'])
            equity_df.to_csv(os.path.join(backtest_dir, 'equity_curve.csv'), index=False)
        
        return {
            "backtest_results": backtest_results,
            "errors": state.errors.copy()
        }
    except Exception as e:
        error_msg = f"运行回测失败: {str(e)}"
        logger.error(error_msg)
        errors = state.errors.copy()
        errors.append(error_msg)
        return {"errors": errors}

def analyze_performance_node(state: BacktestingState) -u003e Dict[str, Any]:
    """分析回测性能节点"""
    logger.info("开始分析回测性能")
    if not state.backtest_results:
        error_msg = "没有回测结果，无法分析性能"
        logger.error(error_msg)
        errors = state.errors.copy()
        errors.append(error_msg)
        return {"errors": errors}
    
    try:
        # 创建性能分析器实例
        performance_analyzer = PerformanceAnalyzer()
        
        # 分析性能
        performance_metrics = performance_analyzer.analyze_performance(
            backtest_results=state.backtest_results
        )
        
        # 打印关键性能指标
        logger.info(f"回测性能分析完成: 总收益率={performance_metrics.get('total_return_pct', 0):.2f}%, "
                    f"最大回撤={performance_metrics.get('max_drawdown_pct', 0):.2f}%, "
                    f"夏普比率={performance_metrics.get('sharpe_ratio', 0):.2f}")
        
        return {
            "performance_metrics": performance_metrics,
            "errors": state.errors.copy()
        }
    except Exception as e:
        error_msg = f"分析回测性能失败: {str(e)}"
        logger.error(error_msg)
        errors = state.errors.copy()
        errors.append(error_msg)
        return {"errors": errors}

def generate_report_node(state: BacktestingState) -u003e Dict[str, Any]:
    """生成回测报告节点"""
    logger.info("开始生成回测报告")
    if not state.backtest_results or not state.performance_metrics:
        error_msg = "缺少回测结果或性能指标，无法生成报告"
        logger.error(error_msg)
        errors = state.errors.copy()
        errors.append(error_msg)
        return {"errors": errors}
    
    try:
        # 创建报告生成器实例
        report_generator = ReportGenerator()
        
        # 生成报告
        report_path = report_generator.generate_report(
            backtest_results=state.backtest_results,
            performance_metrics=state.performance_metrics
        )
        
        logger.info(f"回测报告生成完成: {report_path}")
        
        return {
            "report_path": report_path,
            "errors": state.errors.copy()
        }
    except Exception as e:
        error_msg = f"生成回测报告失败: {str(e)}"
        logger.error(error_msg)
        errors = state.errors.copy()
        errors.append(error_msg)
        return {"errors": errors}

def create_backtesting_workflow() -u003e StateGraph:
    """创建回测模块工作流"""
    workflow = StateGraph(BacktestingState)
    
    # 添加节点
    workflow.add_node("load_backtest_data", load_backtest_data_node)
    workflow.add_node("load_trading_signals", load_trading_signals_node)
    workflow.add_node("run_backtest", run_backtest_node)
    workflow.add_node("analyze_performance", analyze_performance_node)
    workflow.add_node("generate_report", generate_report_node)
    
    # 定义边
    workflow.add_edge(START, "load_backtest_data")
    workflow.add_edge("load_backtest_data", "load_trading_signals")
    workflow.add_edge("load_trading_signals", "run_backtest")
    workflow.add_edge("run_backtest", "analyze_performance")
    workflow.add_edge("analyze_performance", "generate_report")
    workflow.add_edge("generate_report", END)
    
    # 编译工作流
    return workflow.compile()

def run_backtesting_workflow() -u003e Dict[str, Any]:
    """运行回测模块工作流"""
    logger.info("启动回测模块工作流")
    
    try:
        # 创建并运行工作流
        app = create_backtesting_workflow()
        result = app.invoke({})
        
        # 构建返回结果
        final_result = {
            'status': 'success' if not result['errors'] else 'partial_success',
            'performance_metrics': result.get('performance_metrics'),
            'report_path': result.get('report_path'),
            'total_trades': len(result.get('backtest_results', {}).get('trades', [])),
            'errors': result.get('errors', [])
        }
        
        logger.info(f"回测模块工作流运行完成，状态: {final_result['status']}")
        return final_result
        
    except Exception as e:
        logger.error(f"回测模块工作流运行失败: {str(e)}")
        return {
            'status': 'error',
            'message': str(e),
            'errors': [str(e)]
        }
