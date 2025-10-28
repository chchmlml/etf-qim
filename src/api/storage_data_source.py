# Implementation of the FinancialDataSource interface using local file

import pandas as pd
from typing import List, Optional
import logging
from src.api.data_source_interface import FinancialDataSource, DataSourceError, NoDataFoundError, LoginError

logger = logging.getLogger(__name__)


class StorageDataSource(FinancialDataSource):
    """
    从本地文件中统计需要的数据
    """

    def get_etf_list(self, date: Optional[str] = None) -> pd.DataFrame:
        """获取指定日期的ETF列表"""
        pass

    def get_etf_pe_data(self, date: Optional[str] = None) -> pd.DataFrame:
        """获取指定日期所有ETF的PE数据"""
        pass

    def get_historical_k_data(self, code: str, start_date: str, end_date: str, frequency: str = "d",
                              adjust_flag: str = "3", fields: Optional[List[str]] = None) -> pd.DataFrame:
        pass

    def get_stock_basic_info(self, code: str) -> pd.DataFrame:
        pass

    def get_trade_dates(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        pass

    def get_all_stock(self, date: Optional[str] = None) -> pd.DataFrame:
        pass

    def get_deposit_rate_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        pass

    def get_loan_rate_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        pass

    def get_required_reserve_ratio_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None,
                                        year_type: str = '0') -> pd.DataFrame:
        pass

    def get_money_supply_data_month(self, start_date: Optional[str] = None,
                                    end_date: Optional[str] = None) -> pd.DataFrame:
        pass

    def get_money_supply_data_year(self, start_date: Optional[str] = None,
                                   end_date: Optional[str] = None) -> pd.DataFrame:
        pass

    def get_shibor_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        pass
