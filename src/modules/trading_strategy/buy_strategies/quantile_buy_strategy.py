"""
基于分位点的买入策略
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict
from datetime import datetime

from src.modules.trading_strategy.models.strategy_config import StrategyConfig

logger = logging.getLogger(__name__)


class QuantileBuyStrategy:
    """基于分位点的买入策略"""
    def __init__(self, config: StrategyConfig):
        """初始化买入策略"""
        self.config = config
        
    def generate_signals(self, code: str, pe_data: pd.DataFrame, quantile_data: Dict) -u003e List[Dict]:
        """根据PE数据和分位点数据生成买入信号"""
        signals = []
        
        try:
            # 计算每支ETF的最大可买入金额
            max_investment_per_etf = self.config.total_investment * self.config.max_position_ratio
            
            # 获取当前PE和分位点
            current_pe = quantile_data.get('current', 0)
            current_percentile = quantile_data.get('current_percentile', 1.0)
            
            # 如果当前分位点低于阈值，生成买入信号
            if current_percentile u003c= self.config.low_quantile_threshold:
                # 基于分位点计算买入权重（分位点越低，买入权重越大）
                buy_weight = 1.0 - current_percentile / self.config.low_quantile_threshold
                
                # 计算买入金额（基于权重）
                buy_amount = max_investment_per_etf * buy_weight
                
                # 获取最新日期
                latest_date = pe_data['date'].iloc[-1] if not pe_data.empty else datetime.now()
                
                # 生成买入信号
                signal = {
                    'date': latest_date.strftime('%Y-%m-%d') if isinstance(latest_date, pd.Timestamp) else latest_date,
                    'code': code,
                    'action': 'buy',
                    'pe': current_pe,
                    'percentile': current_percentile,
                    'amount': buy_amount,
                    'weight': buy_weight,
                    'signal_strength': buy_weight
                }
                signals.append(signal)
                logger.debug(f"为{code}生成买入信号: 分位点={current_percentile:.4f}, 买入金额={buy_amount:.2f}元")
            
        except Exception as e:
            logger.error(f"为{code}生成买入信号失败: {str(e)}")
        
        return signals
