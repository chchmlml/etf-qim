"""
交易策略配置类
"""


class StrategyConfig:
    """交易策略配置类"""
    def __init__(self, 
                 total_investment: float = 100000.0,  # 总投入资金
                 max_etfs: int = 10,                  # 总买入ETF数量
                 max_position_ratio: float = 0.2,     # 每支ETF最高买入占比
                 low_quantile_threshold: float = 0.2, # 低估值分位点阈值
                 high_quantile_threshold: float = 0.8,# 高估值分位点阈值
                 retracement_threshold: float = 0.05  # 回撤阈值
                ):
        """初始化策略配置"""
        self.total_investment = total_investment
        self.max_etfs = max_etfs
        self.max_position_ratio = max_position_ratio
        self.low_quantile_threshold = low_quantile_threshold
        self.high_quantile_threshold = high_quantile_threshold
        self.retracement_threshold = retracement_threshold
