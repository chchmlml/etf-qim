"""
ETF量化投资分析系统主工作流
连接数据抓取、交易策略和回测模块
"""
import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, List, Any, Optional
from langgraph.graph import StateGraph, START, END
from datetime import datetime, timedelta

# 导入相关模块
from src.modules.data_acquisition.workflow import run_data_acquisition_workflow
from src.modules.trading_strategy.workflow import run_trading_strategy_workflow
from src.modules.backtesting.workflow import run_backtesting_workflow
from src.utils.data_utils import load_dataframe, load_json, save_dataframe, save_json
from src.utils.logger_utils import setup_logger

# 设置日志
logger = setup_logger()

# 定义工作流状态
class ETFAnalysisState:
    """ETF量化投资分析系统的状态类"""
    def __init__(self):
        self.data_acquisition_result: Optional[Dict[str, Any]] = None
        self.trading_strategy_result: Optional[Dict[str, Any]] = None
        self.backtesting_result: Optional[Dict[str, Any]] = None
        self.errors: List[str] = []

# 定义工作流节点函数
def data_acquisition_node(state: ETFAnalysisState) -> Dict[str, Any]:
    """数据抓取节点"""
    logger.info("启动数据抓取模块")
    try:
        # 运行数据抓取工作流
        result = run_data_acquisition_workflow()
        
        logger.info(f"数据抓取模块运行完成，状态: {result.get('status')}")
        
        # 检查是否有错误
        if result.get('status') != 'success':
            error_msg = f"数据抓取模块运行失败: {result.get('message')}"
            logger.error(error_msg)
            errors = state.errors.copy()
            errors.append(error_msg)
            return {"data_acquisition_result": result, "errors": errors}
        
        return {
            "data_acquisition_result": result,
            "errors": state.errors.copy()
        }
    except Exception as e:
        error_msg = f"数据抓取模块运行异常: {str(e)}"
        logger.error(error_msg)
        errors = state.errors.copy()
        errors.append(error_msg)
        return {"errors": errors}

def trading_strategy_node(state: ETFAnalysisState) -> Dict[str, Any]:
    """交易策略节点"""
    logger.info("启动交易策略模块")
    if not state.data_acquisition_result:
        error_msg = "数据抓取模块未运行成功，无法运行交易策略模块"
        logger.error(error_msg)
        errors = state.errors.copy()
        errors.append(error_msg)
        return {"errors": errors}
    
    try:
        # 运行交易策略工作流
        result = run_trading_strategy_workflow()
        
        logger.info(f"交易策略模块运行完成，状态: {result.get('status')}")
        
        # 检查是否有错误
        if result.get('status') != 'success':
            error_msg = f"交易策略模块运行失败: {result.get('message')}"
            logger.error(error_msg)
            errors = state.errors.copy()
            errors.append(error_msg)
            return {"trading_strategy_result": result, "errors": errors}
        
        return {
            "trading_strategy_result": result,
            "errors": state.errors.copy()
        }
    except Exception as e:
        error_msg = f"交易策略模块运行异常: {str(e)}"
        logger.error(error_msg)
        errors = state.errors.copy()
        errors.append(error_msg)
        return {"errors": errors}

def backtesting_node(state: ETFAnalysisState) -> Dict[str, Any]:
    """回测节点"""
    logger.info("启动回测模块")
    if not state.trading_strategy_result:
        error_msg = "交易策略模块未运行成功，无法运行回测模块"
        logger.error(error_msg)
        errors = state.errors.copy()
        errors.append(error_msg)
        return {"errors": errors}
    
    try:
        # 运行回测工作流
        result = run_backtesting_workflow()
        
        logger.info(f"回测模块运行完成，状态: {result.get('status')}")
        
        # 检查是否有错误
        if result.get('status') != 'success':
            error_msg = f"回测模块运行失败: {result.get('message')}"
            logger.error(error_msg)
            errors = state.errors.copy()
            errors.append(error_msg)
            return {"backtesting_result": result, "errors": errors}
        
        return {
            "backtesting_result": result,
            "errors": state.errors.copy()
        }
    except Exception as e:
        error_msg = f"回测模块运行异常: {str(e)}"
        logger.error(error_msg)
        errors = state.errors.copy()
        errors.append(error_msg)
        return {"errors": errors}

def create_etf_analysis_workflow() -> StateGraph:
    """创建ETF量化投资分析系统主工作流"""
    workflow = StateGraph(ETFAnalysisState)
    
    # 添加节点
    workflow.add_node("data_acquisition", data_acquisition_node)
    workflow.add_node("trading_strategy", trading_strategy_node)
    workflow.add_node("backtesting", backtesting_node)
    
    # 定义边
    workflow.add_edge(START, "data_acquisition")
    workflow.add_edge("data_acquisition", "trading_strategy")
    workflow.add_edge("trading_strategy", "backtesting")
    workflow.add_edge("backtesting", END)
    
    # 编译工作流
    return workflow.compile()

def run_etf_analysis_workflow() -> Dict[str, Any]:
    """运行ETF量化投资分析系统主工作流"""
    logger.info("启动ETF量化投资分析系统")
    
    try:
        # 创建并运行工作流
        app = create_etf_analysis_workflow()
        result = app.invoke({})
        
        # 构建返回结果
        final_result = {
            'status': 'success' if not result['errors'] else 'partial_success',
            'data_acquisition': result.get('data_acquisition_result'),
            'trading_strategy': result.get('trading_strategy_result'),
            'backtesting': result.get('backtesting_result'),
            'errors': result.get('errors', [])
        }
        
        # 记录最终结果
        if final_result['status'] == 'success':
            logger.info("ETF量化投资分析系统运行成功")
        else:
            logger.warning(f"ETF量化投资分析系统运行完成，但有{len(final_result['errors'])}个错误")
            
        # 保存运行结果
        output_dir = os.path.join('data', 'system', datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(output_dir, exist_ok=True)
        save_json(final_result, os.path.join(output_dir, 'system_result.json'))
        
        return final_result
        
    except Exception as e:
        logger.error(f"ETF量化投资分析系统运行失败: {str(e)}")
        return {
            'status': 'error',
            'message': str(e),
            'errors': [str(e)]
        }

# 如果直接运行此脚本，则执行工作流
if __name__ == "__main__":
    result = run_etf_analysis_workflow()
    logger.info(f"系统运行结果: {result}")
