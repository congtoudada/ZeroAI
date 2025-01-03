import os
import time
import traceback
from loguru import logger

from src.base.double_match_comp import DoubleMatchComponent
from zero.key.global_key import GlobalKey


class HelmetComponent(DoubleMatchComponent):
    """
    规范佩戴安全帽检测组件
    """

    def __init__(self, shared_memory, config_path: str):
        super().__init__(shared_memory, config_path)


def create_process(shared_memory, config_path: str):
    comp: HelmetComponent = HelmetComponent(shared_memory, config_path)  # 创建组件
    try:
        comp.start()  # 初始化
        # 初始化结束通知
        shared_memory[GlobalKey.LAUNCH_COUNTER.name] += 1
        while not shared_memory[GlobalKey.ALL_READY.name]:
            time.sleep(0.1)
        comp.update()  # 算法逻辑循环
    except KeyboardInterrupt:
        comp.on_destroy()
    except Exception as e:
        # 使用 traceback 打印堆栈信息
        logger.error(f"HelmetComponent: {e}")
        logger.error(traceback.format_exc())  # 打印完整的堆栈信息
        comp.on_destroy()
