import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
import os

from src.modules.trading_strategy.models.strategy_config import StrategyConfig
from src.utils.logger_util import setup_logger

logger = setup_logger("quantile_sell_strategy")

class QuantileSellStrategy:
    """基于PE分位点的卖出策略"""

    def __init__(self, config: StrategyConfig):
        """初始化卖出策略"""
        self.config = config
        # 使用已有的配置属性而不是调用get_sell_params方法
        logger.info(f"初始化分位点卖出策略，配置: {self.config.to_dict()}")

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

    def determine_sell_amount(self,
                             current_price: float,
                             current_position: float,
                             current_percentile: float,
                             purchase_percentile: float = 0.0) -> Dict[str, Any]:
        """确定卖出金额和数量"""
        # 如果没有持仓，不卖出
        if current_position <= 0:
            return {
                "should_sell": False,
                "reason": "当前没有持仓"
            }

        # 根据分位点计算卖出数量比例
        # 分位点越高，卖出比例越大
        percentile_factor = min(1.0, (current_percentile - self.config.sell_quantile_threshold) / (1.0 - self.config.sell_quantile_threshold))
        sell_ratio = percentile_factor * self.config.sell_amount_multiplier

        # 确保卖出比例在合理范围内
        sell_ratio = max(0.05, min(1.0, sell_ratio))  # 最小卖出5%，最大卖出100%

        # 计算卖出数量
        sell_quantity = int(current_position * sell_ratio)

        # 确保至少卖出1份
        if sell_quantity < 1:
            sell_quantity = 1

        # 计算卖出金额
        sell_amount = sell_quantity * current_price

        # 计算交易成本
        transaction_cost = sell_amount * self.config.transaction_cost_percent

        return {
            "should_sell": True,
            "quantity": sell_quantity,
            "price": current_price,
            "amount": sell_amount,
            "total_amount": sell_amount - transaction_cost,
            "transaction_cost": transaction_cost,
            "percentile_factor": percentile_factor,
            "sell_ratio": sell_ratio
        }

    def should_sell(self,
                   current_percentile: float,
                   purchase_percentile: float = 0.0,
                   current_price: Optional[float] = None,
                   purchase_price: Optional[float] = None) -> Dict[str, bool]:
        """根据多种条件判断是否应该卖出"""
        # 初始化各条件判断结果
        conditions = {
            "quantile_condition": False,
            "profit_condition": False,
            "stop_loss_condition": False
        }

        # 1. 分位点条件：如果当前PE分位点高于卖出阈值，考虑卖出
        if current_percentile > self.config.sell_quantile_threshold:
            conditions["quantile_condition"] = True
            logger.info(f"分位点卖出条件满足: 当前分位点 {current_percentile:.2f} 高于阈值 {self.config.sell_quantile_threshold}")
        else:
            logger.info(f"分位点卖出条件不满足: 当前分位点 {current_percentile:.2f} 不高于阈值 {self.config.sell_quantile_threshold}")

        # 2. 止盈条件：如果有利润且达到止盈阈值，考虑卖出
        if current_price is not None and purchase_price is not None:
            profit_ratio = (current_price - purchase_price) / purchase_price
            if profit_ratio > self.config.profit_taking_threshold:
                conditions["profit_condition"] = True
                logger.info(f"止盈条件满足: 当前利润率 {profit_ratio:.2%} 高于止盈阈值 {self.config.profit_taking_threshold:.2%}")
            else:
                logger.info(f"止盈条件不满足: 当前利润率 {profit_ratio:.2%} 不高于止盈阈值 {self.config.profit_taking_threshold:.2%}")

        # 3. 止损条件：如果亏损且达到止损阈值，考虑卖出
        if current_price is not None and purchase_price is not None:
            loss_ratio = (current_price - purchase_price) / purchase_price
            if loss_ratio < self.config.stop_loss_threshold:
                conditions["stop_loss_condition"] = True
                logger.info(f"止损条件满足: 当前亏损率 {loss_ratio:.2%} 低于止损阈值 {self.config.stop_loss_threshold:.2%}")
            else:
                logger.info(f"止损条件不满足: 当前亏损率 {loss_ratio:.2%} 不低于止损阈值 {self.config.stop_loss_threshold:.2%}")

        # 综合判断：满足任一条件即考虑卖出
        overall_decision = any(conditions.values())

        return {
            "should_sell": overall_decision,
            "conditions": conditions
        }

    def execute(self,
               pe_data_file: str,
               current_price: float,
               current_position: float = 0.0,
               purchase_percentile: float = 0.0,
               purchase_price: Optional[float] = None,
               as_of_date: Optional[datetime] = None) -> Dict[str, Any]:
        """执行卖出策略"""
        try:
            logger.info(f"执行卖出策略: ETF代码={self.config.etf_code}, 当前价格={current_price}, 当前持仓={current_position}")

            # 加载PE数据
            pe_data = self.load_pe_data(pe_data_file)

            # 如果指定了日期，过滤数据
            if as_of_date:
                pe_data = pe_data[pe_data['date'] <= as_of_date]

            # 计算分位点
            quantile_info = self.calculate_quantiles(pe_data['pe'])
            current_percentile = quantile_info['current_percentile']

            # 判断是否应该卖出
            sell_decision = self.should_sell(
                current_percentile,
                purchase_percentile,
                current_price,
                purchase_price
            )

            if sell_decision['should_sell']:
                # 确定卖出金额和数量
                sell_amount_decision = self.determine_sell_amount(
                    current_price,
                    current_position,
                    current_percentile,
                    purchase_percentile
                )

                if sell_amount_decision['should_sell']:
                    # 构建卖出结果
                    # 确定触发卖出的主要原因
                    trigger_reason = ""
                    if sell_decision['conditions']['stop_loss_condition']:
                        trigger_reason = "达到止损阈值"
                    elif sell_decision['conditions']['profit_condition']:
                        trigger_reason = "达到止盈阈值"
                    elif sell_decision['conditions']['quantile_condition']:
                        trigger_reason = f"PE分位点 {current_percentile:.2f} 高于卖出阈值 {self.config.sell_quantile_threshold}"

                    result = {
                        "should_sell": True,
                        "quantity": sell_amount_decision['quantity'],
                        "price": sell_amount_decision['price'],
                        "total_amount": sell_amount_decision['total_amount'],
                        "transaction_cost": sell_amount_decision['transaction_cost'],
                        "reason": trigger_reason,
                        "conditions": sell_decision['conditions'],
                        "quantiles": quantile_info,
                        "timestamp": datetime.now().isoformat()
                    }
                    logger.info(f"卖出决策: 卖出 {sell_amount_decision['quantity']} 份，价格 {sell_amount_decision['price']}，总金额 {sell_amount_decision['total_amount']:.2f}")
                    return result
                else:
                    # 不满足卖出数量条件
                    result = {
                        "should_sell": False,
                        "reason": sell_amount_decision.get('reason', '未满足卖出条件'),
                        "quantiles": quantile_info,
                        "timestamp": datetime.now().isoformat()
                    }
                    logger.info(f"不卖出: {result['reason']}")
                    return result
            else:
                # 不满足卖出条件
                result = {
                    "should_sell": False,
                    "reason": "未满足任何卖出条件",
                    "conditions": sell_decision['conditions'],
                    "quantiles": quantile_info,
                    "timestamp": datetime.now().isoformat()
                }
                logger.info(f"不卖出: {result['reason']}")
                return result

        except Exception as e:
            logger.error(f"执行卖出策略失败: {str(e)}")
            # 返回错误信息，避免整个流程中断
            return {
                "should_sell": False,
                "reason": f"策略执行失败: {str(e)}",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def backtest(self,
                pe_data_file: str,
                price_data_file: str,
                start_date: str,
                end_date: str,
                initial_position: float = 0.0) -> Dict[str, Any]:
        """回测卖出策略"""
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
                "total_sell_amount": 0.0,
                "trade_dates": [],
                "details": []
            }

            # 模拟交易
            current_position = initial_position
            purchase_price = None  # 简化处理，假设只有一次买入
            purchase_percentile = 0.0

            for i, row in merged_data.iterrows():
                # 为了简化回测，假设在第一天买入
                if i == 0 and current_position == 0:
                    current_position = 1000  # 假设买入1000份
                    purchase_price = row['price']
                    # 计算买入时的分位点
                    historical_pe = merged_data.iloc[:i+1]['pe']
                    buy_quantile_info = self.calculate_quantiles(historical_pe)
                    purchase_percentile = buy_quantile_info['current_percentile']

                    logger.info(f"回测初始化: 买入 {current_position} 份，价格 {purchase_price}")

                # 计算截至当前日期的历史PE数据的分位点
                historical_pe = merged_data.iloc[:i+1]['pe']
                quantile_info = self.calculate_quantiles(historical_pe)
                current_percentile = quantile_info['current_percentile']

                # 判断是否应该卖出
                sell_decision = self.should_sell(
                    current_percentile,
                    purchase_percentile,
                    row['price'],
                    purchase_price
                )

                if sell_decision['should_sell'] and current_position > 0:
                    # 确定卖出金额和数量
                    sell_amount_decision = self.determine_sell_amount(
                        row['price'],
                        current_position,
                        current_percentile,
                        purchase_percentile
                    )

                    if sell_amount_decision['should_sell']:
                        # 更新回测结果
                        backtest_results['total_trades'] += 1
                        backtest_results['total_sell_amount'] += sell_amount_decision['total_amount']
                        backtest_results['trade_dates'].append(row['date'].strftime('%Y-%m-%d'))

                        # 记录交易详情
                        backtest_results['details'].append({
                            "date": row['date'].strftime('%Y-%m-%d'),
                            "price": row['price'],
                            "quantity": sell_amount_decision['quantity'],
                            "total_amount": sell_amount_decision['total_amount'],
                            "pe": row['pe'],
                            "percentile": current_percentile,
                            "conditions": sell_decision['conditions']
                        })

                        # 更新当前持仓
                        current_position -= sell_amount_decision['quantity']

                        # 如果清仓，重置买入价格和分位点
                        if current_position == 0:
                            purchase_price = None
                            purchase_percentile = 0.0

            logger.info(f"回测完成: 总交易次数={backtest_results['total_trades']}, 总卖出金额={backtest_results['total_sell_amount']:.2f}, 最终持仓={current_position}")
            return backtest_results

        except Exception as e:
            logger.error(f"回测失败: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    # 在QuantileSellStrategy类中添加以下方法
    def execute_with_analysis(self,
                          pe_analysis_result: Dict[str, Any],
                          current_price: float,
                          current_position: float = 0.0,
                          purchase_percentile: float = 0.0,
                          purchase_price: Optional[float] = None) -> Dict[str, Any]:
        """使用预计算的PE分析结果执行卖出策略"""
        try:
            logger.info(f"使用预计算PE分析结果执行卖出策略: ETF代码={self.config.etf_code}, 当前价格={current_price}, 当前持仓={current_position}")

            # 获取预计算的分位点信息
            quantile_info = pe_analysis_result
            current_percentile = quantile_info['current_percentile']

            # 判断是否应该卖出
            sell_decision = self.should_sell(
                current_percentile,
                purchase_percentile,
                current_price,
                purchase_price
            )

            if sell_decision['should_sell']:
                # 确定卖出金额和数量
                sell_amount_decision = self.determine_sell_amount(
                    current_price,
                    current_position,
                    current_percentile,
                    purchase_percentile
                )

                if sell_amount_decision['should_sell']:
                    # 构建卖出结果
                    result = {
                        "should_sell": True,
                        "quantity": sell_amount_decision['quantity'],
                        "price": sell_amount_decision['price'],
                        "total_amount": sell_amount_decision['total_amount'],
                        "transaction_cost": sell_amount_decision['transaction_cost'],
                        "reason": f"满足卖出条件: {sell_decision['conditions']}",
                        "quantiles": quantile_info,
                        "timestamp": datetime.now().isoformat()
                    }
                    logger.info(f"卖出决策: 卖出 {sell_amount_decision['quantity']} 份，价格 {sell_amount_decision['price']}，总金额 {sell_amount_decision['total_amount']:.2f}")
                    return result
                else:
                    return sell_amount_decision
            else:
                return {
                    "should_sell": False,
                    "reason": f"不满足任何卖出条件",
                    "conditions": sell_decision['conditions']
                }
        except Exception as e:
            logger.error(f"执行卖出策略失败: {str(e)}")
            return {
                "should_sell": False,
                "reason": f"策略执行错误: {str(e)}"
            }
