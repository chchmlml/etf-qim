"""
回测引擎类，用于执行ETF策略回测
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class BacktestEngine:
    """回测引擎类"""
    def __init__(self):
        """初始化回测引擎"""
        logger.info("初始化回测引擎")
        
    def run_backtest(self, 
                     etf_data_dict: Dict[str, pd.DataFrame],
                     buy_signals: Dict[str, List[Dict]],
                     sell_signals: Dict[str, List[Dict]],
                     initial_capital: float = 100000.0,
                     start_date: str = None,
                     end_date: str = None
                    ) -u003e Dict[str, Any]:
        """运行回测
        
        Args:
            etf_data_dict: ETF数据字典，键为ETF代码，值为包含PE等数据的DataFrame
            buy_signals: 买入信号字典，键为ETF代码，值为买入信号列表
            sell_signals: 卖出信号字典，键为ETF代码，值为卖出信号列表
            initial_capital: 初始资金
            start_date: 回测开始日期
            end_date: 回测结束日期
            
        Returns:
            回测结果字典
        """
        logger.info(f"开始回测: 初始资金={initial_capital:.2f}元")
        
        try:
            # 准备回测数据
            all_dates = []
            for code, df in etf_data_dict.items():
                if 'date' in df.columns:
                    all_dates.extend(df['date'].tolist())
            
            if not all_dates:
                logger.error("没有有效的日期数据")
                return {'status': 'error', 'message': '没有有效的日期数据'}
            
            # 确定回测的日期范围
            all_dates = sorted(list(set(all_dates)))
            
            # 如果没有指定日期范围，则使用数据中的日期范围
            if start_date is None:
                start_date = all_dates[0]
            else:
                start_date = pd.to_datetime(start_date)
            
            if end_date is None:
                end_date = all_dates[-1]
            else:
                end_date = pd.to_datetime(end_date)
            
            # 筛选回测期间的日期
            backtest_dates = [date for date in all_dates if start_date u003c= date u003c= end_date]
            
            if not backtest_dates:
                logger.error(f"在指定的日期范围内没有数据: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
                return {'status': 'error', 'message': '在指定的日期范围内没有数据'}
            
            logger.info(f"回测日期范围: {backtest_dates[0].strftime('%Y-%m-%d')} 至 {backtest_dates[-1].strftime('%Y-%m-%d')}, 共{len(backtest_dates)}个交易日")
            
            # 初始化回测状态
            cash = initial_capital
            positions = {}
            trades = []
            equity_curve = []
            
            # 按日期进行回测
            for date_idx, current_date in enumerate(backtest_dates):
                logger.debug(f"回测日期: {current_date.strftime('%Y-%m-%d')} ({date_idx+1}/{len(backtest_dates)})")
                
                # 处理当天的交易信号
                self._process_signals(
                    current_date=current_date,
                    etf_data_dict=etf_data_dict,
                    buy_signals=buy_signals,
                    sell_signals=sell_signals,
                    cash=cash,
                    positions=positions,
                    trades=trades
                )
                
                # 计算当日的资产总值
                total_equity = cash + self._calculate_positions_value(positions, current_date, etf_data_dict)
                
                # 记录资产曲线
                equity_curve.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'equity': total_equity,
                    'cash': cash,
                    'positions_value': total_equity - cash,
                    'daily_return': 0.0 if date_idx == 0 else 
                                  (total_equity / equity_curve[date_idx-1]['equity'] - 1)
                })
            
            # 构建回测结果
            backtest_results = {
                'status': 'success',
                'initial_capital': initial_capital,
                'final_equity': equity_curve[-1]['equity'] if equity_curve else initial_capital,
                'total_return': (equity_curve[-1]['equity'] / initial_capital - 1) if equity_curve else 0.0,
                'total_days': len(backtest_dates),
                'trades': trades,
                'equity_curve': equity_curve,
                'positions': positions,
                'cash': cash,
                'start_date': backtest_dates[0].strftime('%Y-%m-%d'),
                'end_date': backtest_dates[-1].strftime('%Y-%m-%d')
            }
            
            logger.info(f"回测完成: 初始资金={initial_capital:.2f}元, 最终资产={backtest_results['final_equity']:.2f}元, "
                        f"总收益率={(backtest_results['total_return'] * 100):.2f}%")
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"回测过程中发生错误: {str(e)}")
            return {'status': 'error', 'message': str(e)}
        
    def _process_signals(self, current_date: datetime, 
                        etf_data_dict: Dict[str, pd.DataFrame],
                        buy_signals: Dict[str, List[Dict]],
                        sell_signals: Dict[str, List[Dict]],
                        cash: float, positions: Dict, trades: List):
        """处理当日的交易信号"""
        # 将datetime转换为字符串以便比较
        date_str = current_date.strftime('%Y-%m-%d')
        
        # 处理卖出信号
        for code, signals in sell_signals.items():
            for signal in signals:
                if signal.get('date') == date_str and code in positions:
                    # 获取当前价格（使用PE作为价格的模拟）
                    price = self._get_price_at_date(code, current_date, etf_data_dict)
                    if price u003c= 0:
                        continue
                    
                    # 计算卖出数量
                    sell_ratio = signal.get('sell_ratio', 0.5)
                    sell_quantity = int(positions[code]['quantity'] * sell_ratio)
                    
                    if sell_quantity u003e 0:
                        # 执行卖出交易
                        sell_amount = sell_quantity * price
                        cash += sell_amount
                        positions[code]['quantity'] -= sell_quantity
                        
                        # 记录交易
                        trades.append({
                            'date': date_str,
                            'code': code,
                            'action': 'sell',
                            'quantity': sell_quantity,
                            'price': price,
                            'amount': sell_amount,
                            'reason': 'quantile_based_sell',
                            'signal_strength': signal.get('signal_strength', 0.0)
                        })
                        
                        logger.debug(f"卖出 {code}: {sell_quantity} 份额, 价格: {price:.2f}, 金额: {sell_amount:.2f}元")
                        
                        # 如果持仓数量为0，则移除该ETF的持仓
                        if positions[code]['quantity'] == 0:
                            del positions[code]
        
        # 处理买入信号
        for code, signals in buy_signals.items():
            for signal in signals:
                if signal.get('date') == date_str:
                    # 获取当前价格（使用PE作为价格的模拟）
                    price = self._get_price_at_date(code, current_date, etf_data_dict)
                    if price u003c= 0 or cash u003c= 0:
                        continue
                    
                    # 计算买入数量
                    buy_amount = signal.get('amount', cash * 0.1)  # 默认买入可用资金的10%
                    buy_quantity = int(min(buy_amount, cash) / price)
                    
                    if buy_quantity u003e 0:
                        # 执行买入交易
                        buy_amount_actual = buy_quantity * price
                        cash -= buy_amount_actual
                        
                        # 更新持仓
                        if code not in positions:
                            positions[code] = {'quantity': 0, 'avg_cost': 0.0}
                        
                        # 更新平均成本
                        total_cost = positions[code]['quantity'] * positions[code]['avg_cost'] + buy_amount_actual
                        positions[code]['quantity'] += buy_quantity
                        positions[code]['avg_cost'] = total_cost / positions[code]['quantity']
                        
                        # 记录交易
                        trades.append({
                            'date': date_str,
                            'code': code,
                            'action': 'buy',
                            'quantity': buy_quantity,
                            'price': price,
                            'amount': buy_amount_actual,
                            'reason': 'quantile_based_buy',
                            'signal_strength': signal.get('signal_strength', 0.0)
                        })
                        
                        logger.debug(f"买入 {code}: {buy_quantity} 份额, 价格: {price:.2f}, 金额: {buy_amount_actual:.2f}元")
        
    def _get_price_at_date(self, code: str, date: datetime, etf_data_dict: Dict[str, pd.DataFrame]) -u003e float:
        """获取指定日期的ETF价格（使用PE作为价格的模拟）"""
        if code not in etf_data_dict:
            return 0.0
        
        df = etf_data_dict[code]
        
        # 查找指定日期的数据
        if 'date' in df.columns:
            # 将日期列转换为datetime类型（如果尚未转换）
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            
            # 查找指定日期的数据
            date_mask = df['date'] == date
            if date_mask.any():
                # 使用PE作为价格的模拟
                return df.loc[date_mask, 'pe'].iloc[0] if 'pe' in df.columns else 0.0
        
        return 0.0
        
    def _calculate_positions_value(self, positions: Dict, date: datetime, etf_data_dict: Dict[str, pd.DataFrame]) -u003e float:
        """计算当前持仓的总价值"""
        total_value = 0.0
        
        for code, pos in positions.items():
            if code in etf_data_dict:
                price = self._get_price_at_date(code, date, etf_data_dict)
                total_value += pos['quantity'] * price
        
        return total_value
