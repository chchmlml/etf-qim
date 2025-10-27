# Utility functions, including the Baostock login context manager and logging setup

import logging
import os
from typing import Optional
import sys

# 定义日志级别对应的颜色
class ColoredFormatter(logging.Formatter):
    # ANSI 颜色代码
    COLORS = {
        'DEBUG': '\033[94m',      # 蓝色
        'INFO': '\033[37m',       # 白色
        'WARNING': '\033[93m',    # 黄色
        'ERROR': '\033[91m',      # 红色
        'CRITICAL': '\033[95m',   # 紫色
        'RESET': '\033[0m'        # 重置颜色
    }

    def format(self, record):
        # 保存原始的格式化字符串
        original_fmt = self._fmt

        # 根据日志级别添加颜色
        if record.levelname in self.COLORS:
            self._fmt = f"{self.COLORS[record.levelname]}{original_fmt}{self.COLORS['RESET']}"

        # 调用父类的format方法
        result = super().format(record)

        # 恢复原始的格式化字符串
        self._fmt = original_fmt

        return result


def setup_logger(name: str, log_dir: Optional[str] = None) -> logging.Logger:
    """设置统一的日志配置，增加按天归档功能

    Args:
        name: logger的名称
        log_dir: 日志文件目录，如果为None则使用默认的logs目录

    Returns:
        配置好的logger实例
    """
    logging.getLogger().setLevel(logging.DEBUG)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    if log_dir is None:
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_script_dir))
        log_dir = os.path.join(project_root, 'logs')

    os.makedirs(log_dir, exist_ok=True)

    # 创建带有时间戳的格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 创建控制台处理器并设置颜色
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # 在支持颜色的终端上使用彩色格式化器
    if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
    else:
        console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger

# --- 示例用法 (与之前相同) ---
if __name__ == "__main__":

    # 初始化第一个logger
    my_logger_1 = setup_logger("agent_state")
    my_logger_1.info("This is an info message from app_main.")
    my_logger_1.debug("This is a debug message from app_main.")
    my_logger_1.warning("This is a warning message from app_main.")
    my_logger_1.error("This is an error message from app_main.")


# 预定义的图标
SUCCESS_ICON = "✅"
ERROR_ICON = "❌"
WAIT_ICON = "⏳"
