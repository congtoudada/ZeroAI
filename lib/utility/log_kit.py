import os
import time
from loguru import logger
from enum import Enum

from zero.info.base_info import BaseInfo


class LogKit:
    """
    日志工具
    """
    class Level(Enum):
        TRACE = 0
        DEBUG = 1
        INFO = 2
        SUCCESS = 3
        WARNING = 4
        ERROR = 5
        CRITICAL = 6

    LevelArray = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]

    @staticmethod
    def get_level(index: int) -> str:
        if index < 0 or index > 7:
            return LogKit.LevelArray[0]
        return LogKit.LevelArray[index]

    @staticmethod
    def load_info(baseInfo: BaseInfo) -> bool:
        if baseInfo is None:
            logger.warning("加载日志模块失败，baseInfo为None")
            return False
        return LogKit.load_params(baseInfo.log_enable, baseInfo.log_output_path, baseInfo.log_level, baseInfo.log_clean)

    @staticmethod
    def load_params(enable: bool, output: str = None, level: int = 0, is_clean=False) -> bool:
        """
        加载日志配置
        :param enable: 是否开启日志
        :param output: 日志输出路径（文件名自带日期前缀）
        :param level: 日志等级，小于该等级的日志不打印
        :param is_clean: 是否自动清理
        :return:
        """
        if not enable:
            logger.remove()
            return False
        # 大于DEBUG级别的日志，不显示在控制台
        if level > 1:
            logger.remove()

        # 设置日志过滤等级
        logger.level(LogKit.get_level(level))

        if output is not None:
            if is_clean:
                LogKit._clean(os.path.dirname(output))
            output_path = os.path.join(os.path.dirname(output), "{time:YYYY-MM-DD}_" + os.path.basename(output))
            logger.add(sink=output_path, rotation="daily")
        return True

    @staticmethod
    def _clean(folder: str):
        localtime = time.localtime()
        begin_time_str = time.strftime('%Y-%m-%d', localtime)
        # 每2个月清理1个月的日志
        now_month = int(begin_time_str.split('-')[1])
        # 使用os.walk()遍历文件夹
        for root, dirs, files in os.walk(folder):
            for filename in files:
                if not filename.endswith('.log'):
                    continue
                file_path = os.path.join(root, filename)
                month = int(file_path.split('-')[1])

                diff = now_month - month
                if diff < 0:
                    diff = now_month + 12 - month

                if diff >= 2:
                    os.remove(file_path)
