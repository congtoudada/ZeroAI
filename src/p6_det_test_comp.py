
import time
import traceback

from loguru import logger

from utility.config_kit import ConfigKit
from zero.core.based_stream_comp import BasedStreamComponent
from zero.info.based_stream_info import BasedStreamInfo
from zero.key.detection_key import DetectionKey
from zero.key.global_key import GlobalKey


class P6DetTestComponent(BasedStreamComponent):

    def __init__(self, shared_memory, config_path: str):
        super().__init__(shared_memory)
        self.config: BasedStreamInfo = BasedStreamInfo(ConfigKit.load(config_path))

    def on_get_stream(self, read_idx):
        frame, _ = super().on_get_stream(read_idx)  # 解析视频帧id+视频帧
        if frame is None:  # 没有有效帧
            return frame, None
        # 解析额外数据
        stream_package = self.read_dict[read_idx][self.config.input_ports[read_idx]]
        input_det = stream_package[DetectionKey.DET_PACKAGE_RESULT.name]  # 目标检测结果
        return frame, input_det

    def on_handle_stream(self, idx, frame, input_det) -> object:
        frame_id = self.frame_id_cache[idx]
        logger.info(f"frame_id: {frame_id}")
        if input_det is not None:
            for item in input_det:
                logger.info(f"ltrb:{item[:4]} cls:{item[5]}")

    def on_draw_vis(self, idx, frame, user_data):
        return frame


def create_process(shared_memory, config_path: str):
    comp = P6DetTestComponent(shared_memory, config_path)  # 创建组件
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
        logger.error(f"P6DetTestComponent: {e}")
        logger.error(traceback.format_exc())  # 打印完整的堆栈信息
        comp.on_destroy()
