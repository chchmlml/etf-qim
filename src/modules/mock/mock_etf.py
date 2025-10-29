import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from src.utils.logger_util import setup_logger
import matplotlib.pyplot as plt

# 配置日志
logger = setup_logger(__name__)


class MockETFDataGenerator:
    """模拟ETF数据生成器"""

    def __init__(self):
        """初始化模拟数据生成器"""
        # 优化1：将保存目录改为项目目录下的data目录
        current_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        self.base_path = f'{current_path}/data'
        os.makedirs(self.base_path, exist_ok=True)

    def generate_etf_list(self, count: int = 200) -> list:
        """生成ETF代码列表

        Args:
            count: 需要生成的ETF数量

        Returns:
            ETF代码列表
        """
        # 生成200个ETF代码，格式为050001到050200
        return [f'05{i:04d}' for i in range(1, count + 1)]

    def generate_trading_dates(self, years: int = 2) -> pd.DatetimeIndex:
        """生成最近指定年数的交易日历

        Args:
            years: 需要生成的年数

        Returns:
            交易日索引
        """
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=365 * years)

        # 生成日期范围
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B'表示工作日

        # 简单模拟中国市场节假日（这里简化处理，实际应用中可能需要更复杂的假日处理）
        chinese_holidays = []
        # 添加一些固定假日
        for year in range(start_date.year, end_date.year + 1):
            chinese_holidays.extend([
                f'{year}-01-01',  # 元旦
                f'{year}-02-12', f'{year}-02-13', f'{year}-02-14', f'{year}-02-15', f'{year}-02-16',  # 春节
                f'{year}-04-05',  # 清明节
                f'{year}-05-01',  # 劳动节
                f'{year}-06-14',  # 端午节
                f'{year}-10-01', f'{year}-10-02', f'{year}-10-03', f'{year}-10-04', f'{year}-10-05'  # 国庆节
            ])

        # 移除假日
        trading_dates = date_range[~date_range.isin(pd.to_datetime(chinese_holidays))]

        logger.info(f"生成了{len(trading_dates)}个交易日数据（{years}年）")
        return trading_dates

    def generate_etf_pe_data(self, etf_code: str, trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """为单个ETF生成符合正态分布的PE历史数据

        Args:
            etf_code: ETF代码
            trading_dates: 交易日历

        Returns:
            包含日期和PE数据的DataFrame
        """
        # 设置随机种子，确保可重复
        np.random.seed(int(etf_code) * 100)  # 使用ETF代码作为种子

        # 为每个ETF设置不同的均值和标准差，使数据更加真实多样
        base_mean = 15 + (int(etf_code[-2:]) % 10) * 0.5  # 基础PE均值在15-20之间
        base_std = 3 + (int(etf_code[-2:]) % 10) * 0.2  # 基础标准差在3-5之间

        # 生成符合正态分布的PE值
        pe_values = np.random.normal(loc=base_mean, scale=base_std, size=len(trading_dates))

        # 添加一些趋势性，使数据更真实
        time_trend = np.linspace(0, 5, len(trading_dates)) * (1 if np.random.random() > 0.5 else -1)  # 随机上升或下降趋势
        pe_values = pe_values + time_trend

        # 确保PE值为正数
        pe_values = np.maximum(5, pe_values)  # PE值不低于5

        # 构建DataFrame
        data = pd.DataFrame({
            'date': trading_dates.strftime('%Y-%m-%d'),
            'symbol': etf_code,
            'pe': pe_values
        })

        # 添加一些随机缺失值，使数据更真实
        missing_rate = 0.02  # 2%的数据缺失
        mask = np.random.random(len(data)) < missing_rate
        data.loc[mask, 'pe'] = np.nan

        return data

    def generate_etf_close_price_data(self, etf_code: str, trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """为单个ETF生成收盘价历史数据

        Args:
            etf_code: ETF代码
            trading_dates: 交易日历

        Returns:
            包含日期和收盘价数据的DataFrame
        """
        # 设置随机种子，确保可重复
        np.random.seed(int(etf_code) * 100 + 1)  # 使用ETF代码+1作为种子，与PE数据区分

        # 为每个ETF设置不同的均值和标准差
        base_mean = 1.2 + (int(etf_code[-2:]) % 10) * 0.05  # 基础收盘价均值在1.2-1.7之间
        base_std = 0.1 + (int(etf_code[-2:]) % 10) * 0.01  # 基础标准差在0.1-0.2之间

        # 生成符合正态分布的收盘价
        close_prices = np.random.normal(loc=base_mean, scale=base_std, size=len(trading_dates))

        # 添加一些趋势性，使数据更真实
        time_trend = np.linspace(0, 0.5, len(trading_dates)) * (1 if np.random.random() > 0.5 else -1)  # 随机上升或下降趋势
        close_prices = close_prices + time_trend

        # 确保收盘价为正数
        close_prices = np.maximum(0.5, close_prices)  # 收盘价不低于0.5

        # 构建DataFrame
        data = pd.DataFrame({
            'date': trading_dates.strftime('%Y-%m-%d'),
            'symbol': etf_code,
            'close_price': close_prices
        })

        # 添加一些随机缺失值，使数据更真实
        missing_rate = 0.02  # 2%的数据缺失
        mask = np.random.random(len(data)) < missing_rate
        data.loc[mask, 'close_price'] = np.nan

        return data

    def generate_all_etf_data(self, etf_count: int = 200, years: int = 2) -> dict:
        """生成所有ETF的数据

        Args:
            etf_count: ETF数量
            years: 历史数据年数

        Returns:
            ETF数据字典，键为ETF代码，值为包含PE和收盘价数据的字典
        """
        logger.info(f"开始生成{etf_count}个ETF的{years}年历史数据")

        # 生成ETF代码列表
        etf_codes = self.generate_etf_list(etf_count)

        # 生成交易日历
        trading_dates = self.generate_trading_dates(years)

        # 为每个ETF生成数据
        etf_data_dict = {}
        for i, etf_code in enumerate(etf_codes, 1):
            if i % 20 == 0 or i == etf_count:  # 每20个ETF打印一次进度
                logger.info(f"已生成{i}/{etf_count}个ETF数据")

            # 生成单个ETF的PE和收盘价数据
            etf_data_dict[etf_code] = {
                'pe': self.generate_etf_pe_data(etf_code, trading_dates),
                'close_price': self.generate_etf_close_price_data(etf_code, trading_dates)
            }

        logger.info(f"成功生成{etf_count}个ETF的历史数据")
        return etf_data_dict

    def save_etf_data_to_csv(self, etf_data_dict: dict) -> None:
        """将ETF数据保存到CSV文件

        Args:
            etf_data_dict: ETF数据字典
        """
        # 为每个ETF生成单独的文件
        for etf_code, data_dict in etf_data_dict.items():
            # 保存PE数据
            pe_file_path = os.path.join(self.base_path, f'{etf_code}_pe_data.csv')
            data_dict['pe'].to_csv(pe_file_path, index=False)
            logger.info(f"成功将ETF {etf_code} 的PE数据保存到{pe_file_path}")

            # 保存收盘价数据
            close_price_file_path = os.path.join(self.base_path, f'{etf_code}_close_price_data.csv')
            data_dict['close_price'].to_csv(close_price_file_path, index=False)
            logger.info(f"成功将ETF {etf_code} 的收盘价数据保存到{close_price_file_path}")


# 主函数，方便直接运行生成数据
def main():
    """主函数"""
    # 创建数据生成器实例
    generator = MockETFDataGenerator()

    # 生成ETF数据 - 根据需求可以调整etf_count参数
    etf_data_dict = generator.generate_all_etf_data(etf_count=1, years=2)

    # 保存数据到CSV文件
    generator.save_etf_data_to_csv(etf_data_dict)


if __name__ == "__main__":
    main()
