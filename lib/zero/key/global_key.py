from enum import Enum


class GlobalKey(Enum):
    """
    全局共享内存Key
    使用举例: Global.EVENT_ESC.name（非端口相关不用）
    """
    EVENT_ESC = 0  # 退出事件
    LAUNCH_COUNTER = 1  # 启动计数器（用于控制初始化各模块的顺序）
    ALL_READY = 2  # 全部脚本初始化完毕的信号
