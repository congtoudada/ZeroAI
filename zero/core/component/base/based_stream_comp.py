from loguru import logger
import numpy as np

from zero.core.component.base.base_comp import Component
from zero.core.key.shared_key import SharedKey


class BasedStreamComponent(Component):
    """
    基于视频流的算法组件
    """
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.width = 640
        self.height = 640
        self.channel = 3
        self.frame = None
        self.current_frame_id = 0

    def on_start(self):
        super().on_start()
        self.width = int(self.shared_data[SharedKey.STREAM_ORIGINAL_WIDTH])
        self.height = int(self.shared_data[SharedKey.STREAM_ORIGINAL_HEIGHT])
        self.channel = int(self.shared_data[SharedKey.STREAM_ORIGINAL_CHANNEL])

    def on_update(self) -> bool:
        # 只有不同帧才有必要计算
        frame_info = self.shared_data[SharedKey.STREAM_FRAME_INFO]
        if frame_info is not None and self.current_frame_id != int(frame_info[SharedKey.STREAM_FRAME_ID]):
            self.current_frame_id = int(frame_info[SharedKey.STREAM_FRAME_ID])
            self.frame = np.reshape(np.ascontiguousarray(np.copy(frame_info[SharedKey.STREAM_FRAME])),
                                    (self.height, self.width, self.channel))
            self.analysis(self.shared_data[SharedKey.STREAM_FRAME_INFO][SharedKey.STREAM_FRAME_ID])  # 打印性能分析报告
            return True
        return False

    def start(self):
        self.on_start()
        logger.info(f"{self.pname} 成功初始化！")
        # 只要基于视频流的组件，在初始化结束都必须通知给流进程
        self.shared_data[SharedKey.STREAM_WAIT_COUNTER] += 1
