import json
import os
from typing import Dict

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from src.utils.logger_util import setup_logger

logger = setup_logger("storage_util")

def ensure_directory(directory_path: str) -> None:
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        logger.info(f"创建目录: {directory_path}")


def save_dataframe(df: pd.DataFrame, file_path: str) -> None:
    """将DataFrame保存到CSV文件"""
    try:
        # 确保文件所在目录存在
        directory = os.path.dirname(file_path)
        if directory:
            ensure_directory(directory)

        # 保存DataFrame到CSV文件
        df.to_csv(file_path, index=False, encoding='utf-8')
        logger.debug(f"成功保存数据到: {file_path}")
    except Exception as e:
        logger.error(f"保存数据到{file_path}失败: {str(e)}")
        # 可以选择是否抛出异常，这里选择记录错误并继续执行


def generate_pe_chart(pe_data: pd.DataFrame, quantile_data: Dict[str, float], etf_name: str, save_path: str) -> None:
    """生成ETF的PE走势图，并标记分位线"""
    try:
        # 确保保存目录存在
        directory = os.path.dirname(save_path)
        if directory:
            ensure_directory(directory)

        # 设置中文显示
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
        plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

        # 创建图形
        plt.figure(figsize=(12, 6))

        # 转换日期格式
        pe_data['date'] = pd.to_datetime(pe_data['date'])

        # 绘制PE曲线
        plt.plot(pe_data['date'], pe_data['pe'], 'b-', label='PE值')

        # 绘制分位线
        plt.axhline(y=quantile_data['5%'], color='g', linestyle='--', alpha=0.7, label='5%分位线')
        plt.axhline(y=quantile_data['20%'], color='c', linestyle='--', alpha=0.7, label='20%分位线')
        plt.axhline(y=quantile_data['50%'], color='y', linestyle='--', alpha=0.7, label='50%分位线(中位数)')
        plt.axhline(y=quantile_data['80%'], color='m', linestyle='--', alpha=0.7, label='80%分位线')
        plt.axhline(y=quantile_data['95%'], color='r', linestyle='--', alpha=0.7, label='95%分位线')
        plt.axhline(y=quantile_data['mean'], color='k', linestyle='-', alpha=0.5, label='平均值')

        # 标记当前值
        current_date = pe_data['date'].iloc[-1]
        current_pe = quantile_data['current']
        plt.scatter(current_date, current_pe, color='r', s=100, zorder=5, label=f'当前PE: {current_pe:.2f}')

        # 添加网格线
        plt.grid(True, linestyle='--', alpha=0.7)

        # 设置日期格式
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)

        # 添加标题和标签
        plt.title(f'{etf_name} 近一年PE走势图\n当前分位点: {quantile_data["current_percentile"]:.2%}')
        plt.xlabel('日期')
        plt.ylabel('PE值')

        # 添加图例
        plt.legend(loc='best')

        # 调整布局
        plt.tight_layout()

        # 保存图片
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.debug(f"成功生成PE图表并保存到: {save_path}")
    except Exception as e:
        logger.error(f"生成PE图表失败: {str(e)}")
        # 确保关闭图形，避免内存泄漏
        plt.close()
        # 抛出异常，让调用方知道发生了错误
        raise
