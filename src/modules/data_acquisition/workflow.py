"""
数据抓取生成模块的LangGraph工作流
负责抓取所有股票型ETF、ETD的指标（PE必须）、绘制最近一年的PE指标图，计算分位线
"""
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Any, Optional
from langgraph.graph import StateGraph, START, END
from datetime import datetime, timedelta

# 导入数据源和工具函数
from src.utils.baostock_data_source import BaostockDataSource
from src.utils.logger_util import setup_logger
from src.utils.storage_util import ensure_directory, save_dataframe, generate_pe_chart

logger = setup_logger("data_acquisition")

# 定义工作流状态
class DataAcquisitionState:
    """数据抓取生成模块的状态类"""

    def __init__(self):
        self.etf_list: Optional[pd.DataFrame] = None
        self.pe_data_dict: Dict[str, pd.DataFrame] = {}
        self.quantile_data: Dict[str, Dict[str, float]] = {}
        self.chart_paths: Dict[str, str] = {}
        self.errors: List[str] = []
        self.timestamp: str = datetime.now().strftime('%Y%m%d_%H%M%S')


# 定义工作流节点函数
def fetch_etf_list_node(state: DataAcquisitionState) -> Dict[str, Any]:
    """获取ETF/ETD列表节点"""
    logger.info("开始获取股票型ETF/ETD列表")
    try:
        # 使用Baostock数据源获取ETF列表
        data_source = BaostockDataSource()
        etf_list = data_source.get_stock_etf_list()

        # 过滤出股票型ETF
        stock_etfs = etf_list[etf_list['type'] == 'stock']
        logger.info(f"成功获取{len(stock_etfs)}个股票型ETF")

        return {
            "etf_list": stock_etfs,
            "errors": state.errors.copy()
        }
    except Exception as e:
        error_msg = f"获取ETF列表失败: {str(e)}"
        logger.error(error_msg)
        errors = state.errors.copy()
        errors.append(error_msg)
        return {"errors": errors}


def fetch_pe_data_node(state: DataAcquisitionState) -> Dict[str, Any]:
    """获取PE数据节点"""
    logger.info("开始获取PE数据")
    if state.etf_list is None or state.etf_list.empty:
        error_msg = "没有ETF列表数据，无法获取PE数据"
        logger.error(error_msg)
        errors = state.errors.copy()
        errors.append(error_msg)
        return {"errors": errors}

    try:
        data_source = BaostockDataSource()
        pe_data_dict = {}

        # 定义时间范围（最近一年）
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        # 批量获取PE数据
        for _, row in state.etf_list.iterrows():
            code = row['code']
            name = row['code_name']
            try:
                pe_data = data_source.get_index_pe(
                    code=code,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )
                if not pe_data.empty:
                    pe_data_dict[code] = pe_data
                    logger.debug(f"成功获取{name}({code})的PE数据")
            except Exception as e:
                logger.warning(f"获取{name}({code})的PE数据失败: {str(e)}")
                continue

        logger.info(f"成功获取{len(pe_data_dict)}个ETF的PE数据")

        # 保存原始数据到本地
        raw_data_dir = os.path.join('data', 'raw', f'etf_pe_data_{state.timestamp}')
        ensure_directory(raw_data_dir)

        for code, data in pe_data_dict.items():
            file_path = os.path.join(raw_data_dir, f'{code}_pe.csv')
            save_dataframe(data, file_path)

        return {
            "pe_data_dict": pe_data_dict,
            "errors": state.errors.copy()
        }
    except Exception as e:
        error_msg = f"获取PE数据失败: {str(e)}"
        logger.error(error_msg)
        errors = state.errors.copy()
        errors.append(error_msg)
        return {"errors": errors}


def calculate_quantiles_node(state: DataAcquisitionState) -> Dict[str, Any]:
    """计算分位线节点"""
    logger.info("开始计算PE分位线")
    if not state.pe_data_dict:
        error_msg = "没有PE数据，无法计算分位线"
        logger.error(error_msg)
        errors = state.errors.copy()
        errors.append(error_msg)
        return {"errors": errors}

    try:
        quantile_data = {}

        for code, pe_data in state.pe_data_dict.items():
            # 去除NaN值
            valid_pe = pe_data['pe'].dropna()
            if len(valid_pe) > 0:
                # 计算分位线
                quantiles = {
                    '5%': float(np.percentile(valid_pe, 5)),
                    '20%': float(np.percentile(valid_pe, 20)),
                    '50%': float(np.percentile(valid_pe, 50)),
                    '80%': float(np.percentile(valid_pe, 80)),
                    '95%': float(np.percentile(valid_pe, 95)),
                    'mean': float(valid_pe.mean()),
                    'current': float(valid_pe.iloc[-1]),
                    'current_percentile': float(np.sum(valid_pe <= valid_pe.iloc[-1]) / len(valid_pe))
                }
                quantile_data[code] = quantiles

        # 保存分位线数据到本地
        processed_data_dir = os.path.join('data', 'processed', f'etf_quantile_data_{state.timestamp}')
        ensure_directory(processed_data_dir)

        for code, data in quantile_data.items():
            file_path = os.path.join(processed_data_dir, f'{code}_quantiles.json')
            with open(file_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(data, f, ensure_ascii=False, indent=2)

        return {
            "quantile_data": quantile_data,
            "errors": state.errors.copy()
        }
    except Exception as e:
        error_msg = f"计算分位线失败: {str(e)}"
        logger.error(error_msg)
        errors = state.errors.copy()
        errors.append(error_msg)
        return {"errors": errors}


def generate_charts_node(state: DataAcquisitionState) -> Dict[str, Any]:
    """生成PE图表节点"""
    logger.info("开始生成PE指标图")
    if not state.pe_data_dict or not state.quantile_data:
        error_msg = "没有PE数据或分位线数据，无法生成图表"
        logger.error(error_msg)
        errors = state.errors.copy()
        errors.append(error_msg)
        return {"errors": errors}

    try:
        chart_paths = {}

        # 确保图表保存目录存在
        chart_dir = os.path.join('data', 'visuals', f'etf_pe_charts_{state.timestamp}')
        ensure_directory(chart_dir)

        # 批量生成图表
        for code, pe_data in state.pe_data_dict.items():
            if code in state.quantile_data:
                try:
                    # 获取ETF名称
                    etf_name = None
                    if state.etf_list is not None:
                        etf_row = state.etf_list[state.etf_list['code'] == code]
                        if not etf_row.empty:
                            etf_name = etf_row.iloc[0]['code_name']

                    # 生成图表
                    chart_path = os.path.join(chart_dir, f'{code}_pe_chart.png')
                    generate_pe_chart(
                        pe_data=pe_data,
                        quantile_data=state.quantile_data[code],
                        etf_name=etf_name or code,
                        save_path=chart_path
                    )
                    chart_paths[code] = chart_path
                    logger.debug(f"成功生成{code}的PE图表")
                except Exception as e:
                    logger.warning(f"生成{code}的PE图表失败: {str(e)}")
                    continue

        logger.info(f"成功生成{len(chart_paths)}个ETF的PE图表")

        return {
            "chart_paths": chart_paths,
            "errors": state.errors.copy()
        }
    except Exception as e:
        error_msg = f"生成PE图表失败: {str(e)}"
        logger.error(error_msg)
        errors = state.errors.copy()
        errors.append(error_msg)
        return {"errors": errors}


def create_data_acquisition_workflow() -> StateGraph:
    """创建数据抓取生成模块工作流"""
    workflow = StateGraph(DataAcquisitionState)

    # 添加节点
    workflow.add_node("fetch_etf_list", fetch_etf_list_node)
    workflow.add_node("fetch_pe_data", fetch_pe_data_node)
    workflow.add_node("calculate_quantiles", calculate_quantiles_node)
    workflow.add_node("generate_charts", generate_charts_node)

    # 定义边
    workflow.add_edge(START, "fetch_etf_list")
    workflow.add_edge("fetch_etf_list", "fetch_pe_data")
    workflow.add_edge("fetch_pe_data", "calculate_quantiles")
    workflow.add_edge("calculate_quantiles", "generate_charts")
    workflow.add_edge("generate_charts", END)

    # 编译工作流
    return workflow.compile()


def run_data_acquisition_workflow() -> Dict[str, Any]:
    """运行数据抓取生成模块工作流"""
    logger.info("启动数据抓取生成模块工作流")

    try:
        # 创建并运行工作流
        app = create_data_acquisition_workflow()
        result = app.invoke({})

        # 构建返回结果
        final_result = {
            'status': 'success' if not result['errors'] else 'partial_success',
            'etf_count': len(result.get('pe_data_dict', {})),
            'chart_count': len(result.get('chart_paths', {})),
            'timestamp': result.get('timestamp'),
            'errors': result.get('errors', [])
        }

        logger.info(f"数据抓取生成模块工作流运行完成，状态: {final_result['status']}")
        return final_result

    except Exception as e:
        logger.error(f"数据抓取生成模块工作流运行失败: {str(e)}")
        return {
            'status': 'error',
            'message': str(e),
            'errors': [str(e)]
        }


if __name__ == "__main__":

    try:
        # 运行数据抓取工作流
        logger.info("===== 数据抓取模块测试开始 =====")
        result = run_data_acquisition_workflow()

        # 打印工作流执行结果
        logger.info("\n===== 数据抓取模块测试结果 =====")
        logger.info(f"状态: {result.get('status')}")
        logger.info(f"成功获取PE数据的ETF数量: {result.get('etf_count', 0)}")
        logger.info(f"成功生成图表数量: {result.get('chart_count', 0)}")
        logger.info(f"时间戳: {result.get('timestamp')}")

        # 打印错误信息（如果有）
        errors = result.get('errors', [])
        if errors:
            logger.info(f"\n错误列表 ({len(errors)}):")
            for i, error in enumerate(errors, 1):
                logger.info(f"  {i}. {error}")

        logger.info("\n===== 数据抓取模块测试结束 =====")

    except Exception as e:
        logger.info(f"\n测试执行失败: {str(e)}")
        import traceback

        traceback.print_exc()
