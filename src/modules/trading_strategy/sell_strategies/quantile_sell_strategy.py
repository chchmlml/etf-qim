"""
基于分位点的卖出策略
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict
from datetime import datetime

from src.modules.trading_strategy.models.strategy_config import StrategyConfig

logger = logging.getLogger(__name__)


class QuantileSellStrategy:
    """基于分位点的卖出策略"""
    def __init__(self, config: StrategyConfig):
        """初始化卖出策略"""
        self.config = config
        
    def generate_signals(self, code: str, pe_data: pd.DataFrame, quantile_data: Dict) -u003e List[Dict]:
        """根据PE数据和分位点数据生成卖出信号"""
        signals = []
        
        try:
            # 获取当前PE和分位点
            current_pe = quantile_data.get('current', 0)
            current_percentile = quantile_data.get('current_percentile', 0.0)
            
            # 如果当前分位点高于阈值，生成卖出信号
            if current_percentile u003e= self.config.high_quantile_threshold:
                # 计算卖出比例（分位点越高，卖出比例越大）
                # 分位点在0.8-1.0之间，卖出比例在0.3-1.0之间
                sell_ratio = min(1.0, 0.3 + (current_percentile - 0.8) / 0.2 * 0.7)
                
                # 获取最新日期
                latest_date = pe_data['date'].iloc[-1] if not pe_data.empty else datetime.now()
                
                # 生成卖出信号
                signal = {
                    'date': latest_date.strftime('%Y-%m-%d') if isinstance(latest_date, pd.Timestamp) else latest_date,
                    'code': code,
                    'action': 'sell',
                    'pe': current_pe,
                    'percentile': current_percentile,
                    'sell_ratio': sell_ratio,
                    'signal_type': 'quantile',
                    'signal_strength': sell_ratio
                }
                signals.append(signal)
                logger.debug(f"为{code}生成基于分位点的卖出信号: 分位点={current_percentile:.4f}, 卖出比例={sell_ratio:.2f}")
            
        except Exception as e:
            logger.error(f"为{code}生成卖出信号失败: {str(e)}")
        
        return signals
