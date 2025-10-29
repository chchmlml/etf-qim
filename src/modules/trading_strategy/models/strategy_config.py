from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import json
import os

@dataclass
class StrategyConfig:
    """交易策略配置类"""
    # 基本配置
    strategy_name: str = "quantile_based_strategy"
    etf_code: str = ""
    initial_investment: float = 10000.0
    max_position_percent: float = 1.0  # 最大持仓比例，1.0表示100%
    transaction_cost_percent: float = 0.001  # 交易成本比例，0.001表示0.1%
    min_transaction_amount: float = 100.0  # 最小交易金额
    
    # 买入策略配置
    buy_quantile_threshold: float = 0.2  # 买入分位点阈值，低于该分位点时考虑买入
    buy_amount_multiplier: float = 1.0  # 买入金额乘数
    max_buy_percent_per_trade: float = 0.2  # 单次交易最大买入比例
    
    # 卖出策略配置
    sell_quantile_threshold: float = 0.8  # 卖出分位点阈值，高于该分位点时考虑卖出
    sell_amount_multiplier: float = 1.0  # 卖出金额乘数
    profit_taking_threshold: float = 0.1  # 止盈阈值，0.1表示10%
    stop_loss_threshold: float = -0.05  # 止损阈值，-0.05表示-5%
    
    # 数据相关配置
    lookback_period: int = 365  # 回看期天数
    data_frequency: str = "daily"  # 数据频率
    
    # 风险控制配置
    max_drawdown_limit: float = -0.2  # 最大回撤限制，-0.2表示-20%
    portfolio_diversification_count: int = 5  # 投资组合分散数量
    
    # 自定义参数
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后的数据验证"""
        # 验证百分比参数
        if not 0 <= self.max_position_percent <= 1.0:
            raise ValueError("max_position_percent must be between 0 and 1")
        
        if not 0 <= self.transaction_cost_percent <= 0.1:
            raise ValueError("transaction_cost_percent must be between 0 and 0.1")
        
        if not 0 <= self.buy_quantile_threshold <= 1.0:
            raise ValueError("buy_quantile_threshold must be between 0 and 1")
        
        if not 0 <= self.sell_quantile_threshold <= 1.0:
            raise ValueError("sell_quantile_threshold must be between 0 and 1")
        
        if not 0 <= self.max_buy_percent_per_trade <= 1.0:
            raise ValueError("max_buy_percent_per_trade must be between 0 and 1")
        
        # 验证数值参数
        if self.initial_investment <= 0:
            raise ValueError("initial_investment must be positive")
        
        if self.min_transaction_amount <= 0:
            raise ValueError("min_transaction_amount must be positive")
        
        if self.lookback_period <= 0:
            raise ValueError("lookback_period must be positive")
        
        # 验证买入卖出阈值关系
        if self.buy_quantile_threshold >= self.sell_quantile_threshold:
            raise ValueError("buy_quantile_threshold must be less than sell_quantile_threshold")
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return {
            "strategy_name": self.strategy_name,
            "etf_code": self.etf_code,
            "initial_investment": self.initial_investment,
            "max_position_percent": self.max_position_percent,
            "transaction_cost_percent": self.transaction_cost_percent,
            "min_transaction_amount": self.min_transaction_amount,
            "buy_quantile_threshold": self.buy_quantile_threshold,
            "buy_amount_multiplier": self.buy_amount_multiplier,
            "max_buy_percent_per_trade": self.max_buy_percent_per_trade,
            "sell_quantile_threshold": self.sell_quantile_threshold,
            "sell_amount_multiplier": self.sell_amount_multiplier,
            "profit_taking_threshold": self.profit_taking_threshold,
            "stop_loss_threshold": self.stop_loss_threshold,
            "lookback_period": self.lookback_period,
            "data_frequency": self.data_frequency,
            "max_drawdown_limit": self.max_drawdown_limit,
            "portfolio_diversification_count": self.portfolio_diversification_count,
            "custom_params": self.custom_params
        }
    
    def save_to_file(self, file_path: str) -> None:
        """将配置保存到文件"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'StrategyConfig':
        """从文件加载配置"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"配置文件不存在: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
    
    def update(self, updates: Dict[str, Any]) -> 'StrategyConfig':
        """更新配置参数"""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif key == "custom_params":
                self.custom_params.update(value)
            else:
                raise AttributeError(f"配置中不存在属性: {key}")
        
        # 重新验证
        self.__post_init__()
        return self
    
    def validate_against_market(self, market_cap: float, avg_daily_volume: float) -> bool:
        """根据市场数据验证配置是否合理"""
        # 检查最小交易金额是否合理
        if self.min_transaction_amount > avg_daily_volume * 0.01:
            # 最小交易金额超过日均交易量的1%，可能难以执行
            return False
        
        # 检查初始投资是否合理
        if self.initial_investment > market_cap * 0.001:
            # 初始投资超过市值的0.1%，可能对市场产生影响
            return False
        
        return True
    
    def get_buy_params(self) -> Dict[str, Any]:
        """获取买入策略相关参数"""
        return {
            "buy_quantile_threshold": self.buy_quantile_threshold,
            "buy_amount_multiplier": self.buy_amount_multiplier,
            "max_buy_percent_per_trade": self.max_buy_percent_per_trade,
            "transaction_cost_percent": self.transaction_cost_percent,
            "min_transaction_amount": self.min_transaction_amount,
            "lookback_period": self.lookback_period
        }
    
    def get_sell_params(self) -> Dict[str, Any]:
        """获取卖出策略相关参数"""
        return {
            "sell_quantile_threshold": self.sell_quantile_threshold,
            "sell_amount_multiplier": self.sell_amount_multiplier,
            "profit_taking_threshold": self.profit_taking_threshold,
            "stop_loss_threshold": self.stop_loss_threshold,
            "transaction_cost_percent": self.transaction_cost_percent,
            "lookback_period": self.lookback_period
        }

@dataclass
class BacktestConfig:
    """回测配置类"""
    start_date: str
    end_date: str
    initial_cash: float
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    benchmark_symbol: str = ""
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    risk_free_rate: float = 0.02
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_cash": self.initial_cash,
            "commission_rate": self.commission_rate,
            "slippage_rate": self.slippage_rate,
            "benchmark_symbol": self.benchmark_symbol,
            "rebalance_frequency": self.rebalance_frequency,
            "risk_free_rate": self.risk_free_rate
        }

@dataclass
class ETFConfig:
    """ETF配置类"""
    code: str
    name: str
    industry: str
    market: str  # sh, sz, etc.
    inception_date: str
    underlying_index: str
    management_fee: float
    tracking_error: float
    is_leveraged: bool = False
    leverage_ratio: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return {
            "code": self.code,
            "name": self.name,
            "industry": self.industry,
            "market": self.market,
            "inception_date": self.inception_date,
            "underlying_index": self.underlying_index,
            "management_fee": self.management_fee,
            "tracking_error": self.tracking_error,
            "is_leveraged": self.is_leveraged,
            "leverage_ratio": self.leverage_ratio
        }

@dataclass
class PortfolioConfig:
    """投资组合配置类"""
    name: str
    description: str
    etf_configs: List[ETFConfig] = field(default_factory=list)
    asset_allocation: Dict[str, float] = field(default_factory=dict)  # {etf_code: weight}
    rebalance_strategy: str = "periodic"  # periodic, threshold, etc.
    rebalance_threshold: float = 0.05  # 再平衡阈值，0.05表示5%
    
    def __post_init__(self):
        """初始化后的数据验证"""
        # 验证资产配置权重和为1
        if self.asset_allocation:
            total_weight = sum(self.asset_allocation.values())
            if not abs(total_weight - 1.0) < 1e-6:
                raise ValueError(f"资产配置权重和必须为1，当前为: {total_weight}")
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return {
            "name": self.name,
            "description": self.description,
            "etf_configs": [etf.to_dict() for etf in self.etf_configs],
            "asset_allocation": self.asset_allocation,
            "rebalance_strategy": self.rebalance_strategy,
            "rebalance_threshold": self.rebalance_threshold
        }
