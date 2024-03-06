from enum import Enum


class CacheKey(Enum):
    """
    进程间临时数据的Key常量
    """
    """
    视频流
    """
    STREAM_WAIT_MAX = 0  # 视频流开始工作的最大等待数（取决于算法数量，初始化用）
