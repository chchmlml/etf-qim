"""
交易策略模块的LangGraph工作流
根据PE数据计算估值分位点，生成ETF买入和卖出信号
"""
import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, List, Any, Optional
from langgraph.graph import StateGraph, START, END
from datetime import datetime

# 导入相关模块
from src.utils.storage_util import load_dataframe, load_json, save_dataframe, save_json
from src.modules.trading_strategy.models.strategy_config import StrategyConfig
from src.modules.trading_strategy.buy_strategies.quantile_buy_strategy import QuantileBuyStrategy
from src.modules.trading_strategy.sell_strategies.quantile_sell_strategy import QuantileSellStrategy

logger = logging.getLogger(__name__)

# 定义工作流状态
class TradingStrategyState:
    """交易策略模块的状态类"""
    def __init__(self):
        self.etf_pe_data: Dict[str, pd.DataFrame] = {}
        self.quantile_data: Dict[str, Dict] = {}
        self.buy_signals: List[Dict] = []
        self.sell_signals: List[Dict] = []
        self.config: Optional[StrategyConfig] = None
        self.errors: List[str] = []

# 定义工作流节点函数
def load_etf_pe_data_node(state: TradingStrategyState) -u003e Dict[str, Any]:
    """加载ETF的PE数据节点"""
    logger.info("开始加载ETF的PE数据")
    try:
        # 查找最新的PE数据
        data_dirs = [d for d in os.listdir(os.path.join('data', 'processed'))
                     if d.startswith('etf_pe_data_')]
        if not data_dirs:
            error_msg = "没有找到PE数据"
            logger.error(error_msg)
            errors = state.errors.copy()
            errors.append(error_msg)
            return {"errors": errors}

        latest_data_dir = max(data_dirs)
        data_path = os.path.join('data', 'processed', latest_data_dir)

        # 加载所有ETF的PE数据
        etf_pe_data = {}
        pe_files = [f for f in os.listdir(data_path) if f.endswith('_pe.csv')]

        for file in pe_files:
            code = file.split('_')[0]
            file_path = os.path.join(data_path, file)
            pe_data = load_dataframe(file_path)

            # 确保日期是datetime类型
            if 'date' in pe_data.columns and not pd.api.types.is_datetime64_any_dtype(pe_data['date']):
                pe_data['date'] = pd.to_datetime(pe_data['date'])

            # 排序数据
            if 'date' in pe_data.columns:
                pe_data = pe_data.sort_values('date')

            # 只保留最近5年的数据用于计算分位点
            if 'date' in pe_data.columns and not pe_data.empty:
                five_years_ago = datetime.now() - pd.DateOffset(years=5)
                pe_data = pe_data[pe_data['date'] u003e= five_years_ago]

            etf_pe_data[code] = pe_data

        logger.info(f"成功加载{len(etf_pe_data)}个ETF的PE数据")

        return {
            "etf_pe_data": etf_pe_data,
            "errors": state.errors.copy()
        }
    except Exception as e:
        error_msg = f"加载ETF的PE数据失败: {str(e)}"
        logger.error(error_msg)
        errors = state.errors.copy()
        errors.append(error_msg)
        return {"errors": errors}

def initialize_strategy_config_node(state: TradingStrategyState) -u003e Dict[str, Any]:
    """初始化策略配置节点"""
    logger.info("初始化策略配置")
    try:
        # 这里可以从配置文件加载，或者使用默认配置
        config = StrategyConfig(
            total_investment=100000.0,  # 总投入资金10万元
            max_etfs=10,                  # 最多买入10支ETF
            max_position_ratio=0.2,       # 每支ETF最高占比20%
            low_quantile_threshold=0.2,   # 20%分位点以下为低估值
            high_quantile_threshold=0.8,  # 80%分位点以上为高估值
            retracement_threshold=0.05    # 5%回撤阈值
        )

        logger.info(f"策略配置初始化完成: 总资金={config.total_investment}, 最大ETF数量={config.max_etfs}")

        return {
            "config": config,
            "errors": state.errors.copy()
        }
    except Exception as e:
        error_msg = f"初始化策略配置失败: {str(e)}"
        logger.error(error_msg)
        errors = state.errors.copy()
        errors.append(error_msg)
        return {"errors": errors}

def calculate_quantiles_node(state: TradingStrategyState) -u003e Dict[str, Any]:
    """计算ETF的估值分位点节点"""
    logger.info("开始计算ETF的估值分位点")
    if not state.etf_pe_data:
        error_msg = "没有ETF的PE数据，无法计算分位点"
        logger.error(error_msg)
        errors = state.errors.copy()
        errors.append(error_msg)
        return {"errors": errors}

    try:
        quantile_data = {}

        for code, pe_data in state.etf_pe_data.items():
            try:
                # 获取有效的PE数据
                valid_pe = pe_data['pe'].dropna()

                if len(valid_pe) u003e 0:
                    # 获取最新的PE值
                    latest_pe = valid_pe.iloc[-1] if len(valid_pe) u003e 0 else np.nan

                    # 计算分位点
                    percentile = np.sum(valid_pe u003c= latest_pe) / len(valid_pe) if len(valid_pe) u003e 0 else 0.5

                    # 计算历史分位点值
                    q20, q50, q80 = np.nanpercentile(valid_pe, [20, 50, 80])

                    # 构建分位点数据
                    quantile_data[code] = {
                        'code': code,
                        'current': latest_pe,
                        'current_percentile': percentile,
                        'q20': q20,
                        'q50': q50,
                        'q80': q80,
                        'data_points': len(valid_pe),
                        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }

                    # 记录日志
                    logger.debug(f"{code} 估值分位点: 当前PE={latest_pe:.2f}, 分位点={percentile:.2f}")
                else:
                    logger.warning(f"{code} 没有有效的PE数据")

            except Exception as e:
                logger.error(f"计算{code}的分位点失败: {str(e)}")

        # 保存分位点数据
        output_dir = os.path.join('data', 'strategy', datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(output_dir, exist_ok=True)
        save_json(quantile_data, os.path.join(output_dir, 'etf_quantiles.json'))

        logger.info(f"成功计算{len(quantile_data)}个ETF的估值分位点")

        return {
            "quantile_data": quantile_data,
            "errors": state.errors.copy()
        }
    except Exception as e:
        error_msg = f"计算ETF的估值分位点失败: {str(e)}"
        logger.error(error_msg)
        errors = state.errors.copy()
        errors.append(error_msg)
        return {"errors": errors}

def generate_buy_signals_node(state: TradingStrategyState) -u003e Dict[str, Any]:
    """生成买入信号节点"""
    logger.info("开始生成买入信号")
    if not state.etf_pe_data or not state.quantile_data or not state.config:
        error_msg = "缺少数据或配置，无法生成买入信号"
        logger.error(error_msg)
        errors = state.errors.copy()
        errors.append(error_msg)
        return {"errors": errors}

    try:
        # 创建买入策略实例
        buy_strategy = QuantileBuyStrategy(config=state.config)

        # 生成买入信号
        buy_signals = []
        for code, pe_data in state.etf_pe_data.items():
            if code in state.quantile_data:
                signals = buy_strategy.generate_signals(
                    code=code,
                    pe_data=pe_data,
                    quantile_data=state.quantile_data[code]
                )
                buy_signals.extend(signals)

        # 按信号强度排序并限制数量
        buy_signals.sort(key=lambda x: x.get('signal_strength', 0), reverse=True)
        buy_signals = buy_signals[:state.config.max_etfs]  # 限制买入ETF的数量

        # 保存买入信号
        output_dir = os.path.join('data', 'strategy', datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(output_dir, exist_ok=True)
        if buy_signals:
            save_json(buy_signals, os.path.join(output_dir, 'buy_signals.json'))
            logger.info(f"成功生成{len(buy_signals)}个买入信号")
        else:
            logger.info("没有生成买入信号")

        return {
            "buy_signals": buy_signals,
            "errors": state.errors.copy()
        }
    except Exception as e:
        error_msg = f"生成买入信号失败: {str(e)}"
        logger.error(error_msg)
        errors = state.errors.copy()
        errors.append(error_msg)
        return {"errors": errors}

def generate_sell_signals_node(state: TradingStrategyState) -u003e Dict[str, Any]:
    """生成卖出信号节点"""
    logger.info("开始生成卖出信号")
    if not state.etf_pe_data or not state.quantile_data or not state.config:
        error_msg = "缺少数据或配置，无法生成卖出信号"
        logger.error(error_msg)
        errors = state.errors.copy()
        errors.append(error_msg)
        return {"errors": errors}

    try:
        # 创建卖出策略实例
        sell_strategy = QuantileSellStrategy(config=state.config)

        # 生成卖出信号
        sell_signals = []
        for code, pe_data in state.etf_pe_data.items():
            if code in state.quantile_data:
                signals = sell_strategy.generate_signals(
                    code=code,
                    pe_data=pe_data,
                    quantile_data=state.quantile_data[code]
                )
                sell_signals.extend(signals)

        # 按信号强度排序
        sell_signals.sort(key=lambda x: x.get('signal_strength', 0), reverse=True)

        # 保存卖出信号
        output_dir = os.path.join('data', 'strategy', datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(output_dir, exist_ok=True)
        if sell_signals:
            save_json(sell_signals, os.path.join(output_dir, 'sell_signals.json'))
            logger.info(f"成功生成{len(sell_signals)}个卖出信号")
        else:
            logger.info("没有生成卖出信号")

        return {
            "sell_signals": sell_signals,
            "errors": state.errors.copy()
        }
    except Exception as e:
        error_msg = f"生成卖出信号失败: {str(e)}"
        logger.error(error_msg)
        errors = state.errors.copy()
        errors.append(error_msg)
        return {"errors": errors}

def create_trading_strategy_workflow() -u003e StateGraph:
    """创建交易策略模块工作流"""
    workflow = StateGraph(TradingStrategyState)

    # 添加节点
    workflow.add_node("load_etf_pe_data", load_etf_pe_data_node)
    workflow.add_node("initialize_strategy_config", initialize_strategy_config_node)
    workflow.add_node("calculate_quantiles", calculate_quantiles_node)
    workflow.add_node("generate_buy_signals", generate_buy_signals_node)
    workflow.add_node("generate_sell_signals", generate_sell_signals_node)

    # 定义边
    workflow.add_edge(START, "load_etf_pe_data")
    workflow.add_edge("load_etf_pe_data", "initialize_strategy_config")
    workflow.add_edge("initialize_strategy_config", "calculate_quantiles")
    workflow.add_edge("calculate_quantiles", "generate_buy_signals")
    workflow.add_edge("generate_buy_signals", "generate_sell_signals")
    workflow.add_edge("generate_sell_signals", END)

    # 编译工作流
    return workflow.compile()

def run_trading_strategy_workflow() -u003e Dict[str, Any]:
    """运行交易策略模块工作流"""
    logger.info("启动交易策略模块工作流")

    try:
        # 创建并运行工作流
        app = create_trading_strategy_workflow()
        result = app.invoke({})

        # 构建返回结果
        final_result = {
            'status': 'success' if not result['errors'] else 'partial_success',
            'buy_signals_count': len(result.get('buy_signals', [])),
            'sell_signals_count': len(result.get('sell_signals', [])),
            'quantile_data_count': len(result.get('quantile_data', {})),
            'errors': result.get('errors', [])
        }

        logger.info(f"交易策略模块工作流运行完成，状态: {final_result['status']}")
        return final_result

    except Exception as e:
        logger.error(f"交易策略模块工作流运行失败: {str(e)}")
        return {
            'status': 'error',
            'message': str(e),
            'errors': [str(e)]
        }
