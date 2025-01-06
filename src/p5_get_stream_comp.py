import time
import traceback

from loguru import logger

from utility.config_kit import ConfigKit
from zero.core.based_stream_comp import BasedStreamComponent
from zero.info.based_stream_info import BasedStreamInfo
from zero.key.global_key import GlobalKey


class P5GetStreamComponent(BasedStreamComponent):

    def __init__(self, shared_memory, config_path: str):
        super().__init__(shared_memory)
        self.config: BasedStreamInfo = BasedStreamInfo(ConfigKit.load(config_path))

    def on_handle_stream(self, idx, frame, user_data) -> object:
        frame_id = self.frame_id_cache[idx]
        logger.info(f"frame_id: {frame_id}")

    def on_draw_vis(self, idx, frame, user_data):
        return frame


def create_process(shared_memory, config_path: str):
    comp = P5GetStreamComponent(shared_memory, config_path)  # 创建组件
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
        logger.error(f"P5GetStreamComponent: {e}")
        logger.error(traceback.format_exc())  # 打印完整的堆栈信息
        comp.on_destroy()
