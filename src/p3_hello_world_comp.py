import time
import traceback
from loguru import logger
from zero.key.global_key import GlobalKey

from zero.core.component import Component


class P3HelloWorldComponent(Component):
    def __init__(self, shared_memory, config_path: str):
        super().__init__(shared_memory)

    def on_start(self):
        print("hello world")
        logger.info("Hello World!")

    def on_update(self):
        print("hello world update!")

    def on_destroy(self):
        print("hello world destroy!")


def create_process(shared_memory, config_path: str):
    comp = P3HelloWorldComponent(shared_memory, config_path)  # 创建组件
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
        logger.error(f"P3HelloWorldComponent: {e}")
        logger.error(traceback.format_exc())  # 打印完整的堆栈信息
        comp.on_destroy()
