# trading_strategy
# 完成项目的交易策略模块
#
# 1、整体配置文件
# strategy_config.py
# 定义了总投入、最大买入基金数量、每支占比、基金分位线和回撤线
#
# 2、买入策略：读取基金的pe数据文件 #File:
# 050001_pe_data.csv
# 计算基金的低分位线、高分位线，当天处于低分位线进行买入一个头寸操作；注意每支最大占比限制；
#
# 3、卖出操作同买入操作计算基金的低分位线、高分位线，买入后基金已经突破高分位线，当天回撤达到1回撤线标准，进行清仓操作；
#
# 4、低分位线、高分位线处于之间不操作，持有策略
#
# 5、每个买卖策略单独一个文件，代码设计上方便后期扩展更多策略
#
# 6、每次交易写入文件文件格式trade_transactions.csv
#
# trade_id,date,symbol,trade_type,price,quantity,commission,notes
#
# 1,2023-01-10,050020,BUY,1.300,1000,0.0013,Initial purchase based on low PE
#
# 2,2023-03-15,050020,SELL,1.450,500,0.0007,Partial sell due to PE rebound
#
# 3,2023-05-20,050020,BUY,1.400,200,0.0003,Add to position after minor correction
#
#
# 7、交易逻辑在workflow.py实现，要基于langgraph实现流​


from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

from src.modules.trading_strategy.models.strategy_config import StrategyConfig
from src.modules.trading_strategy.buy_strategies.quantile_buy_strategy import QuantileBuyStrategy
from src.modules.trading_strategy.sell_strategies.quantile_sell_strategy import QuantileSellStrategy
from src.utils.logger_util import setup_logger
from src.utils.storage_util import save_dataframe, generate_pe_chart

logger = setup_logger("trading_strategy_workflow")


class TradingState(TypedDict):
    """交易策略状态定义"""
    config: StrategyConfig
    etf_code: str
    current_date: datetime
    current_price: float
    current_position: float
    current_investment: float
    purchase_percentile: float
    pe_data_file: str
    buy_result: Dict[str, Any]
    sell_result: Dict[str, Any]
    action_taken: str
    # 新增字段用于存储PE数据分析结果
    pe_analysis_result: Dict[str, Any]


def initialize_state(config: StrategyConfig, etf_code: str, current_date: datetime,
                     current_price: float, current_position: float = 0.0,
                     current_investment: float = 0.0, purchase_percentile: float = 0.0) -> TradingState:
    """初始化交易状态"""
    # 构建PE数据文件路径
    pe_data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                                "data", f"{etf_code}_pe_data.csv")

    return TradingState(
        config=config,
        etf_code=etf_code,
        current_date=current_date,
        current_price=current_price,
        current_position=current_position,
        current_investment=current_investment,
        purchase_percentile=purchase_percentile,
        pe_data_file=pe_data_file,
        buy_result={},
        sell_result={},
        action_taken="",
        pe_analysis_result={}  # 初始化PE分析结果字段
    )


def analyze_pe_data(state: TradingState) -> TradingState:
    """分析PE数据状态，收集最近1-2年的数据并计算分位点"""
    logger.info(f"分析 {state['etf_code']} 的PE数据")

    # 检查PE数据文件是否存在
    if not os.path.exists(state['pe_data_file']):
        raise FileNotFoundError(f"PE数据文件不存在: {state['pe_data_file']}")

    # 加载PE数据
    try:
        pe_data = pd.read_csv(state['pe_data_file'], names=['date', 'symbol', 'pe'], skiprows=1)
        pe_data['date'] = pd.to_datetime(pe_data['date'])

        # 检查数据是否有效
        if len(pe_data) == 0:
            raise ValueError("PE数据为空")

        logger.info(f"成功加载 {state['etf_code']} 的PE数据，包含 {len(pe_data)} 条记录")

        # 过滤最近1-2年的数据
        # 计算日期范围：从当前日期往前推2年
        two_years_ago = state['current_date'] - timedelta(days=365 * 2)
        # 如果数据量不足2年，则使用最早的可用数据
        if pe_data['date'].min() > two_years_ago:
            logger.warning(f"数据不足2年，使用全部可用数据")
            filtered_data = pe_data
        else:
            filtered_data = pe_data[pe_data['date'] >= two_years_ago]
            logger.info(
                f"过滤后的数据包含 {len(filtered_data)} 条记录，时间范围: {filtered_data['date'].min().strftime('%Y-%m-%d')} 至 {filtered_data['date'].max().strftime('%Y-%m-%d')}")

        # 数据清洗：移除无效的PE值
        filtered_data = filtered_data[filtered_data['pe'].replace([np.inf, -np.inf], np.nan).notna()]
        filtered_data = filtered_data[filtered_data['pe'] > 0]  # 移除负PE值

        if len(filtered_data) == 0:
            raise ValueError("过滤后有效PE数据为空")

        # 按日期排序
        filtered_data = filtered_data.sort_values('date')

        # 计算分位点
        pe_series = filtered_data['pe']

        # 计算关键分位点
        quantiles = {
            0.05: pe_series.quantile(0.05),
            0.2: pe_series.quantile(0.2),  # 低分位线
            0.5: pe_series.quantile(0.5),
            0.8: pe_series.quantile(0.8),  # 高分位线
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

        # 构建分析结果
        pe_analysis_result = {
            "current_pe": current_pe,
            "current_percentile": current_percentile,
            "quantiles": quantiles,
            "pe_mean": pe_series.mean(),
            "pe_std": pe_series.std(),
            "pe_min": pe_series.min(),
            "pe_max": pe_series.max(),
            "data_points": len(pe_series),
            "time_range": {
                "start": filtered_data['date'].min().strftime('%Y-%m-%d'),
                "end": filtered_data['date'].max().strftime('%Y-%m-%d')
            }
        }

        logger.info(
            f"PE数据分析完成: 当前PE={current_pe:.2f}, 当前分位点={current_percentile:.2f}, 低分位线(20%)={quantiles[0.2]:.2f}, 高分位线(80%)={quantiles[0.8]:.2f}")

        # 将分析结果保存到状态中
        state['pe_analysis_result'] = pe_analysis_result

        # 保存原始数据到状态中，供后续使用
        state['pe_data'] = filtered_data

    except Exception as e:
        logger.error(f"加载PE数据失败: {str(e)}")
        raise

    return state


def make_buy_decision(state: TradingState) -> TradingState:
    """买入决策状态"""
    logger.info(f"对 {state['etf_code']} 进行买入决策分析")

    # 创建买入策略实例
    buy_strategy = QuantileBuyStrategy(state['config'])

    # 执行买入策略 - 使用预计算的PE分析结果
    buy_result = buy_strategy.execute_with_analysis(
        state['pe_analysis_result'],
        state['current_price'],
        state['current_investment'],
        state['current_position']
    )

    # 更新状态
    state['buy_result'] = buy_result

    if buy_result.get('should_buy', False):
        logger.info(f"买入决策: 执行买入操作，数量={buy_result['quantity']}")
    else:
        logger.info(f"买入决策: 不执行买入操作")

    return state


def make_sell_decision(state: TradingState) -> TradingState:
    """卖出决策状态"""
    logger.info(f"对 {state['etf_code']} 进行卖出决策分析")

    # 创建卖出策略实例
    sell_strategy = QuantileSellStrategy(state['config'])

    # 执行卖出策略 - 使用预计算的PE分析结果
    sell_result = sell_strategy.execute_with_analysis(
        state['pe_analysis_result'],
        state['current_price'],
        state['current_position'],
        state['purchase_percentile']
    )

    # 更新状态
    state['sell_result'] = sell_result

    if sell_result.get('should_sell', False):
        logger.info(f"卖出决策: 执行卖出操作，数量={sell_result['quantity']}")
    else:
        logger.info(f"卖出决策: 不执行卖出操作")

    return state


def determine_next_action(state: TradingState) -> str:
    """确定下一个动作"""
    # 先检查是否需要卖出
    if state['sell_result'].get('should_sell', False):
        state['action_taken'] = "sell"
        return "execute_sell"

    # 再检查是否需要买入
    elif state['buy_result'].get('should_buy', False):
        state['action_taken'] = "buy"
        return "execute_buy"

    # 如果既不需要买入也不需要卖出，则结束
    else:
        state['action_taken'] = "hold"
        return "no_action"


def execute_buy(state: TradingState) -> TradingState:
    """执行买入操作"""
    buy_result = state['buy_result']

    # 更新持仓和投资金额
    state['current_position'] += buy_result['quantity']
    state['current_investment'] += buy_result['total_amount']

    # 更新买入分位点
    state['purchase_percentile'] = buy_result['quantiles']['current_percentile']

    logger.info(
        f"执行买入: ETF={state['etf_code']}, 数量={buy_result['quantity']}, 价格={buy_result['price']}, 总金额={buy_result['total_amount']:.2f}")

    # 生成PE图表
    try:
        save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                                 "data", "plots", f"{state['etf_code']}_pe_buy_analysis.png")
        generate_pe_chart(
            pd.read_csv(state['pe_data_file'], names=['date', 'symbol', 'pe'], skiprows=1),
            buy_result['quantiles'],
            state['etf_code'],
            save_path
        )
    except Exception as e:
        logger.warning(f"生成PE图表失败: {str(e)}")

    return state


def execute_sell(state: TradingState) -> TradingState:
    """执行卖出操作"""
    sell_result = state['sell_result']

    # 更新持仓和投资金额
    state['current_position'] -= sell_result['quantity']
    state['current_investment'] = max(0, state['current_investment'] - sell_result['total_amount'])

    # 如果清仓，重置买入分位点
    if state['current_position'] == 0:
        state['purchase_percentile'] = 0.0

    logger.info(
        f"执行卖出: ETF={state['etf_code']}, 数量={sell_result['quantity']}, 价格={sell_result['price']}, 总金额={sell_result['total_amount']:.2f}")

    # 生成PE图表
    try:
        save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                                 "data", "plots", f"{state['etf_code']}_pe_sell_analysis.png")
        generate_pe_chart(
            pd.read_csv(state['pe_data_file'], names=['date', 'symbol', 'pe'], skiprows=1),
            sell_result['quantiles'],
            state['etf_code'],
            save_path
        )
    except Exception as e:
        logger.warning(f"生成PE图表失败: {str(e)}")

    return state


def no_action(state: TradingState) -> TradingState:
    """无操作状态"""
    logger.info(
        f"无操作: ETF={state['etf_code']}, 当前价格={state['current_price']}, 当前持仓={state['current_position']}")
    return state


def create_trading_workflow() -> StateGraph:
    """创建交易策略工作流"""
    # 创建状态图
    workflow = StateGraph(TradingState)

    # 添加节点
    workflow.add_node("analyze_pe_data", analyze_pe_data)
    workflow.add_node("make_buy_decision", make_buy_decision)
    workflow.add_node("make_sell_decision", make_sell_decision)
    workflow.add_node("execute_buy", execute_buy)
    workflow.add_node("execute_sell", execute_sell)
    workflow.add_node("no_action", no_action)

    # 设置边
    workflow.set_entry_point("analyze_pe_data")
    workflow.add_edge("analyze_pe_data", "make_buy_decision")
    workflow.add_edge("make_buy_decision", "make_sell_decision")
    workflow.add_conditional_edges(
        "make_sell_decision",
        determine_next_action,
        {
            "execute_buy": "execute_buy",
            "execute_sell": "execute_sell",
            "no_action": "no_action"
        }
    )
    workflow.add_edge("execute_buy", END)
    workflow.add_edge("execute_sell", END)
    workflow.add_edge("no_action", END)

    return workflow


def run_trading_strategy(config: StrategyConfig, etf_code: str, current_date: datetime,
                         current_price: float, current_position: float = 0.0,
                         current_investment: float = 0.0, purchase_percentile: float = 0.0) -> Dict[str, Any]:
    """运行交易策略"""
    try:
        # 初始化状态
        initial_state = initialize_state(
            config,
            etf_code,
            current_date,
            current_price,
            current_position,
            current_investment,
            purchase_percentile
        )

        # 创建并编译工作流
        workflow = create_trading_workflow()
        app = workflow.compile()

        # 运行工作流
        result = app.invoke(initial_state)

        # 格式化结果
        formatted_result = {
            "etf_code": result["etf_code"],
            "current_date": result["current_date"].strftime("%Y-%m-%d"),
            "action_taken": result["action_taken"],
            "current_position": result["current_position"],
            "current_investment": result["current_investment"],
            "purchase_percentile": result["purchase_percentile"],
        }

        # 添加买入或卖出结果
        if result["action_taken"] == "buy" and "buy_result" in result:
            formatted_result["buy_details"] = result["buy_result"]
        elif result["action_taken"] == "sell" and "sell_result" in result:
            formatted_result["sell_details"] = result["sell_result"]

        logger.info(f"交易策略执行完成: {formatted_result}")

        return formatted_result

    except Exception as e:
        logger.error(f"交易策略执行失败: {str(e)}")
        raise


if __name__ == "__main__":
    """交易策略模块的启动入口"""
    try:
        # 导入必要的依赖
        from src.modules.trading_strategy.models.strategy_config import StrategyConfig

        # 示例配置 - 这里可以根据需要修改配置参数
        config = StrategyConfig(
            initial_investment=100000.0,  # 初始投资金额 10万元
            max_position_percent=0.2,  # 单支ETF最大持仓比例20%
            max_buy_percent_per_trade=0.1,  # 每次最大买入比例10%
            min_transaction_amount=1000.0,  # 最小交易金额1000元
            buy_quantile_threshold=0.2,  # 买入分位点阈值0.2
            sell_quantile_threshold=0.8,  # 卖出分位点阈值0.8
            transaction_cost_percent=0.001,  # 交易成本比例0.1%
            buy_amount_multiplier=1.5  # 买入金额乘数1.5
        )

        # 设置测试参数
        etf_code = "050001"  # ETF代码
        current_date = datetime.now()  # 当前日期
        current_price = 1.25  # 当前价格
        current_position = 0.0  # 当前持仓份额
        current_investment = 0.0  # 当前投资金额
        purchase_percentile = 0.0  # 买入时的分位点

        logger.info(f"开始运行交易策略: ETF={etf_code}, 初始投资={config.initial_investment:.2f}")

        # 运行交易策略
        result = run_trading_strategy(
            config,
            etf_code,
            current_date,
            current_price,
            current_position,
            current_investment,
            purchase_percentile
        )

        # 打印结果
        print("\n交易策略执行结果:")
        print(f"ETF代码: {result['etf_code']}")
        print(f"当前日期: {result['current_date']}")
        print(f"采取动作: {result['action_taken']}")
        print(f"当前持仓: {result['current_position']}")
        print(f"当前投资: {result['current_investment']:.2f}")

        if result['action_taken'] == 'buy' and 'buy_details' in result:
            print("\n买入详情:")
            print(f"买入数量: {result['buy_details'].get('quantity', 0)}")
            print(f"买入价格: {result['buy_details'].get('price', 0.0):.2f}")
            print(f"总金额: {result['buy_details'].get('total_amount', 0.0):.2f}")
            print(f"原因: {result['buy_details'].get('reason', '')}")
        elif result['action_taken'] == 'sell' and 'sell_details' in result:
            print("\n卖出详情:")
            print(f"卖出数量: {result['sell_details'].get('quantity', 0)}")
            print(f"卖出价格: {result['sell_details'].get('price', 0.0):.2f}")
            print(f"总金额: {result['sell_details'].get('total_amount', 0.0):.2f}")
            print(f"原因: {result['sell_details'].get('reason', '')}")
        else:
            print("\n无交易操作")

    except FileNotFoundError as e:
        logger.error(f"文件不存在: {str(e)}")
        print(f"错误: 未找到PE数据文件，请确保 {etf_code}_pe_data.csv 文件存在于data目录下")
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        print(f"程序执行出错: {str(e)}")
