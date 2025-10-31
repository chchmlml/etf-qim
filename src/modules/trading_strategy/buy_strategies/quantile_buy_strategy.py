import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
import os

from src.modules.trading_strategy.models.strategy_config import StrategyConfig
from src.utils.logger_util import setup_logger

logger = setup_logger("quantile_buy_strategy")

class QuantileBuyStrategy:
    """基于PE分位点的买入策略"""
    
    def __init__(self, config: StrategyConfig):
        """初始化买入策略"""
        self.config = config
        logger.info(f"初始化分位点买入策略，配置: {config.get_buy_params()}")
    
    def load_pe_data(self, pe_data_file: str) -> pd.DataFrame:
        """加载PE数据"""
        if not os.path.exists(pe_data_file):
            raise FileNotFoundError(f"PE数据文件不存在: {pe_data_file}")
        
        try:
            # 加载PE数据，跳过表头行
            pe_data = pd.read_csv(pe_data_file, names=['date', 'symbol', 'pe'], skiprows=1)
            pe_data['date'] = pd.to_datetime(pe_data['date'])
            
            # 数据清洗：移除无效的PE值
            pe_data = pe_data[pe_data['pe'].replace([np.inf, -np.inf], np.nan).notna()]
            pe_data = pe_data[pe_data['pe'] > 0]  # 移除负PE值
            
            if len(pe_data) == 0:
                raise ValueError("有效PE数据为空")
            
            # 按日期排序
            pe_data = pe_data.sort_values('date')
            
            return pe_data
            
        except Exception as e:
            logger.error(f"加载PE数据失败: {str(e)}")
            raise
    
    def calculate_quantiles(self, pe_series: pd.Series) -> Dict[str, float]:
        """计算PE分位点"""
        # 计算关键分位点
        quantiles = {
            0.05: pe_series.quantile(0.05),
            0.2: pe_series.quantile(0.2),
            0.5: pe_series.quantile(0.5),
            0.8: pe_series.quantile(0.8),
            0.95: pe_series.quantile(0.95)
        }
        
        # 获取最新PE值
        current_pe = pe_series.iloc[-1] if not pe_series.empty else np.nan
        
        # 计算当前PE的分位点位置
        try:
            # 使用scipy的percentileofscore计算分位点
            from scipy import stats
            current_percentile = stats.percentileofscore(pe_series, current_pe) / 100.0
        except ImportError:
            # 如果没有scipy，使用numpy实现近似计算
            current_percentile = np.mean(pe_series <= current_pe)
        except Exception as e:
            logger.warning(f"计算分位点失败: {str(e)}")
            current_percentile = 0.5  # 默认值
        
        # 返回分位点信息
        return {
            "current_pe": current_pe,
            "current_percentile": current_percentile,
            "quantiles": quantiles,
            "pe_mean": pe_series.mean(),
            "pe_std": pe_series.std(),
            "pe_min": pe_series.min(),
            "pe_max": pe_series.max(),
            "data_points": len(pe_series)
        }
    
    def determine_buy_amount(self, 
                            current_price: float, 
                            current_investment: float, 
                            current_position: float, 
                            current_percentile: float) -> Dict[str, Any]:
        """确定买入金额和数量"""
        # 计算可用于投资的最大金额（考虑最大持仓比例）
        max_investment = self.config.initial_investment * self.config.max_position_percent
        available_investment = max_investment - current_investment
        
        # 如果没有可用资金，不买入
        if available_investment < self.config.min_transaction_amount:
            return {
                "should_buy": False,
                "reason": "可用资金不足最小交易金额"
            }
        
        # 根据分位点计算买入金额
        # 分位点越低，买入金额越多
        percentile_factor = 1.0 - current_percentile  # 分位点因子
        base_buy_amount = available_investment * self.config.max_buy_percent_per_trade
        adjusted_buy_amount = base_buy_amount * percentile_factor * self.config.buy_amount_multiplier
        
        # 确保买入金额不低于最小交易金额
        buy_amount = max(adjusted_buy_amount, self.config.min_transaction_amount)
        
        # 计算买入数量（向下取整）
        buy_quantity = int(buy_amount / current_price)
        
        # 计算实际买入金额
        actual_buy_amount = buy_quantity * current_price
        
        # 计算交易成本
        transaction_cost = actual_buy_amount * self.config.transaction_cost_percent
        
        # 检查买入后是否超过最大持仓限制
        if (current_investment + actual_buy_amount + transaction_cost) > max_investment:
            # 调整买入数量
            remaining_amount = max_investment - current_investment
            adjusted_quantity = int(remaining_amount / (current_price * (1 + self.config.transaction_cost_percent)))
            if adjusted_quantity > 0:
                buy_quantity = adjusted_quantity
                actual_buy_amount = buy_quantity * current_price
                transaction_cost = actual_buy_amount * self.config.transaction_cost_percent
            else:
                return {
                    "should_buy": False,
                    "reason": "买入后会超过最大持仓限制"
                }
        
        return {
            "should_buy": True,
            "quantity": buy_quantity,
            "price": current_price,
            "amount": actual_buy_amount,
            "total_amount": actual_buy_amount + transaction_cost,
            "transaction_cost": transaction_cost,
            "percentile_factor": percentile_factor,
            "available_investment": available_investment
        }
    
    def should_buy(self, current_percentile: float) -> bool:
        """根据分位点判断是否应该买入"""
        # 如果当前PE分位点低于买入阈值，考虑买入
        if current_percentile < self.config.buy_quantile_threshold:
            logger.info(f"买入条件满足: 当前分位点 {current_percentile:.2f} 低于阈值 {self.config.buy_quantile_threshold}")
            return True
        
        logger.info(f"买入条件不满足: 当前分位点 {current_percentile:.2f} 不低于阈值 {self.config.buy_quantile_threshold}")
        return False
    
    def execute(self, 
               pe_data_file: str, 
               current_price: float, 
               current_investment: float = 0.0, 
               current_position: float = 0.0, 
               as_of_date: Optional[datetime] = None) -> Dict[str, Any]:
        """执行买入策略"""
        try:
            logger.info(f"执行买入策略: ETF代码={self.config.etf_code}, 当前价格={current_price}")
            
            # 加载PE数据
            pe_data = self.load_pe_data(pe_data_file)
            
            # 如果指定了日期，过滤数据
            if as_of_date:
                pe_data = pe_data[pe_data['date'] <= as_of_date]
            
            # 计算分位点
            quantile_info = self.calculate_quantiles(pe_data['pe'])
            current_percentile = quantile_info['current_percentile']
            
            # 判断是否应该买入
            if self.should_buy(current_percentile):
                # 确定买入金额和数量
                buy_decision = self.determine_buy_amount(
                    current_price,
                    current_investment,
                    current_position,
                    current_percentile
                )
                
                if buy_decision['should_buy']:
                    # 构建买入结果
                    result = {
                        "should_buy": True,
                        "quantity": buy_decision['quantity'],
                        "price": buy_decision['price'],
                        "total_amount": buy_decision['total_amount'],
                        "transaction_cost": buy_decision['transaction_cost'],
                        "reason": f"PE分位点 {current_percentile:.2f} 低于买入阈值 {self.config.buy_quantile_threshold}",
                        "quantiles": quantile_info,
                        "timestamp": datetime.now().isoformat()
                    }
                    logger.info(f"买入决策: 买入 {buy_decision['quantity']} 份，价格 {buy_decision['price']}，总金额 {buy_decision['total_amount']:.2f}")
                    return result
                else:
                    # 不满足资金条件
                    result = {
                        "should_buy": False,
                        "reason": buy_decision.get('reason', '未满足买入条件'),
                        "quantiles": quantile_info,
                        "timestamp": datetime.now().isoformat()
                    }
                    logger.info(f"不买入: {result['reason']}")
                    return result
            else:
                # 不满足分位点条件
                result = {
                    "should_buy": False,
                    "reason": f"PE分位点 {current_percentile:.2f} 不低于买入阈值 {self.config.buy_quantile_threshold}",
                    "quantiles": quantile_info,
                    "timestamp": datetime.now().isoformat()
                }
                logger.info(f"不买入: {result['reason']}")
                return result
                
        except Exception as e:
            logger.error(f"执行买入策略失败: {str(e)}")
            # 返回错误信息，避免整个流程中断
            return {
                "should_buy": False,
                "reason": f"策略执行失败: {str(e)}",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def backtest(self, 
                pe_data_file: str, 
                price_data_file: str, 
                start_date: str, 
                end_date: str) -> Dict[str, Any]:
        """回测买入策略"""
        # 这是一个简化的回测实现，可以根据需要扩展
        try:
            # 加载PE数据
            pe_data = self.load_pe_data(pe_data_file)
            
            # 加载价格数据
            price_data = pd.read_csv(price_data_file)
            price_data['date'] = pd.to_datetime(price_data['date'])
            
            # 合并数据
            merged_data = pd.merge(pe_data, price_data, on=['date', 'symbol'], how='inner')
            
            # 过滤日期范围
            merged_data = merged_data[
                (merged_data['date'] >= pd.to_datetime(start_date)) & 
                (merged_data['date'] <= pd.to_datetime(end_date))
            ]
            
            # 回测结果
            backtest_results = {
                "start_date": start_date,
                "end_date": end_date,
                "total_trades": 0,
                "total_buy_amount": 0.0,
                "trade_dates": [],
                "details": []
            }
            
            # 模拟交易
            current_investment = 0.0
            current_position = 0.0
            
            for i, row in merged_data.iterrows():
                # 计算截至当前日期的历史PE数据的分位点
                historical_pe = merged_data.iloc[:i+1]['pe']
                quantile_info = self.calculate_quantiles(historical_pe)
                current_percentile = quantile_info['current_percentile']
                
                # 判断是否应该买入
                if self.should_buy(current_percentile):
                    # 确定买入金额和数量
                    buy_decision = self.determine_buy_amount(
                        row['price'],
                        current_investment,
                        current_position,
                        current_percentile
                    )
                    
                    if buy_decision['should_buy']:
                        # 更新回测结果
                        backtest_results['total_trades'] += 1
                        backtest_results['total_buy_amount'] += buy_decision['total_amount']
                        backtest_results['trade_dates'].append(row['date'].strftime('%Y-%m-%d'))
                        
                        # 记录交易详情
                        backtest_results['details'].append({
                            "date": row['date'].strftime('%Y-%m-%d'),
                            "price": row['price'],
                            "quantity": buy_decision['quantity'],
                            "total_amount": buy_decision['total_amount'],
                            "pe": row['pe'],
                            "percentile": current_percentile
                        })
                        
                        # 更新当前持仓和投资金额
                        current_position += buy_decision['quantity']
                        current_investment += buy_decision['total_amount']
            
            logger.info(f"回测完成: 总交易次数={backtest_results['total_trades']}, 总买入金额={backtest_results['total_buy_amount']:.2f}")
            return backtest_results
            
        except Exception as e:
            logger.error(f"回测失败: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def execute_with_analysis(self,
                             pe_analysis_result: Dict[str, Any],
                             current_price: float,
                             current_investment: float = 0.0,
                             current_position: float = 0.0) -> Dict[str, Any]:
        """使用预计算的PE分析结果执行买入策略"""
        try:
            logger.info(f"使用预计算PE分析结果执行买入策略: ETF代码={self.config.etf_code}, 当前价格={current_price}")
            
            # 获取预计算的分位点信息
            quantile_info = pe_analysis_result
            current_percentile = quantile_info['current_percentile']
            
            # 判断是否应该买入
            if self.should_buy(current_percentile):
                # 确定买入金额和数量
                buy_decision = self.determine_buy_amount(
                    current_price,
                    current_investment,
                    current_position,
                    current_percentile
                )
                
                if buy_decision['should_buy']:
                    # 构建买入结果
                    result = {
                        "should_buy": True,
                        "quantity": buy_decision['quantity'],
                        "price": buy_decision['price'],
                        "total_amount": buy_decision['total_amount'],
                        "transaction_cost": buy_decision['transaction_cost'],
                        "reason": f"PE分位点 {current_percentile:.2f} 低于买入阈值 {self.config.buy_quantile_threshold}",
                        "quantiles": quantile_info,
                        "timestamp": datetime.now().isoformat()
                    }
                    logger.info(f"买入决策: 买入 {buy_decision['quantity']} 份，价格 {buy_decision['price']}，总金额 {buy_decision['total_amount']:.2f}")
                    return result
                else:
                    return buy_decision
            else:
                return {
                    "should_buy": False,
                    "reason": f"当前分位点 {current_percentile:.2f} 不低于买入阈值 {self.config.buy_quantile_threshold}"
                }
        except Exception as e:
            logger.error(f"执行买入策略失败: {str(e)}")
            return {
                "should_buy": False,
                "reason": f"策略执行错误: {str(e)}"
            }
