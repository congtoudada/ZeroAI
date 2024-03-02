import os
import time
from loguru import logger
from enum import Enum


class LogKit:
    class Level(Enum):
        TRACE = "trace"
        DEBUG = "debug"
        INFO = "info"
        SUCCESS = "success"
        WARNING = "warning"
        ERROR = "error"
        CRITICAL = "critical"

    @staticmethod
    def load(output: str = None, level: Level = Level.TRACE, is_clean=False):
        """
        加载日志配置
        :param output: 日志输出路径（文件名自带日期前缀）
        :param level: 日志等级，小于该等级的日志不打印
        :param is_clean: 是否自动清理
        :return:
        """
        if is_clean:
            LogKit._clean(os.path.dirname(output))
        # 大于DEBUG级别的日志，不显示在控制台
        if level > LogKit.Level.DEBUG:
            logger.remove()
        # 设置日志过滤等级
        logger.level(level.value)
        output_path = os.path.join(os.path.dirname(output), "{time:YYYY-MM-DD}_" + os.path.basename(output) + ".log")
        logger.add(sink=output_path, rotation="daily")

    @staticmethod
    def _clean(folder: str):
        localtime = time.localtime()
        begin_time_str = time.strftime('%Y-%m-%d', localtime)
        # 每2个月清理1个月的日志
        now_month = int(begin_time_str.split('-')[1])
        # 使用os.walk()遍历文件夹
        for root, dirs, files in os.walk(folder):
            for filename in files:
                file_path = os.path.join(root, filename)
                month = int(file_path.split('-')[1])

                diff = now_month - month
                if diff < 0:
                    diff = now_month + 12 - month

                if diff >= 2:
                    os.remove(file_path)
