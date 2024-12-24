import os
import time
import traceback

import cv2
from loguru import logger

from utility.config_kit import ConfigKit
from zero.core.based_stream_comp import BasedStreamComponent
from zero.info.based_stream_info import BasedStreamInfo
from zero.key.global_key import GlobalKey


class EmptyStreamComponent(BasedStreamComponent):
    """
    空白流组件，输出取流画面
    """

    def __init__(self, shared_memory, config_path: str):
        super().__init__(shared_memory)
        self.config: BasedStreamInfo = BasedStreamInfo(ConfigKit.load(config_path))
        self.pname = f"[ {os.getpid()}:EmptyStream ]"

    def on_handle_stream(self, idx, frame, user_data) -> object:
        """
        TODO: 获取frame做一些操作
        """
        frame_id = self.frame_id_cache[idx]
        if frame_id % self.config.update_fps == 0 and frame is not None:
            print(f"{frame_id}: {frame.shape}")
        return None

    def on_draw_vis(self, idx, frame, user_data):
        text_scale = 1
        text_thickness = 1
        cv2.putText(frame, 'Example1.Empty Stream',
                    (0, int(15 * text_scale)),
                    cv2.FONT_HERSHEY_PLAIN, text_scale,
                    (0, 0, 255), thickness=text_thickness)
        return frame


def create_process(shared_memory, config_path: str):
    comp = EmptyStreamComponent(shared_memory, config_path)  # 创建组件
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
        logger.error(f"YoloxComponent: {e}")
        logger.error(traceback.format_exc())  # 打印完整的堆栈信息
        comp.on_destroy()
