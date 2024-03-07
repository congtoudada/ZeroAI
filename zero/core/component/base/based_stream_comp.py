import time

from loguru import logger
import numpy as np

from zero.core.component.base.base_comp import Component
from zero.core.key.shared_key import SharedKey
from zero.utility.timer_kit import TimerKit


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
        self.update_fps = -1
        self.update_timer = TimerKit()  # 计算两帧时间
        self._tic = False

    def on_start(self):
        super().on_start()
        self.width = int(self.shared_data[SharedKey.STREAM_ORIGINAL_WIDTH])
        self.height = int(self.shared_data[SharedKey.STREAM_ORIGINAL_HEIGHT])
        self.channel = int(self.shared_data[SharedKey.STREAM_ORIGINAL_CHANNEL])
        self.update_fps = self.shared_data[SharedKey.STREAM_UPDATE_FPS]

    def on_resolve_stream(self) -> bool:
        # 只有不同帧才有必要计算
        frame_info = self.shared_data[SharedKey.STREAM_FRAME_INFO]
        if frame_info is not None and self.current_frame_id != int(frame_info[SharedKey.STREAM_FRAME_ID]):
            self.current_frame_id = int(frame_info[SharedKey.STREAM_FRAME_ID])
            self.frame = np.reshape(np.ascontiguousarray(np.copy(frame_info[SharedKey.STREAM_FRAME])),
                                    (self.height, self.width, self.channel))
            self.analysis(self.shared_data[SharedKey.STREAM_FRAME_INFO][SharedKey.STREAM_FRAME_ID])  # 打印性能分析报告
            return True
        else:
            return False

    def on_update(self) -> bool:
        super().on_update()
        if self.update_fps > 0:
            time.sleep(1.0 / self.update_fps)
        ret = self.on_resolve_stream()
        if ret:
            if not self._tic:
                self.update_timer.tic()
            else:
                self.update_timer.toc()
            self._tic = not self._tic
        return ret

    def start(self):
        super().start()
        logger.info(f"{self.pname} 成功初始化！")
        # 只要基于视频流的组件，在初始化结束都必须通知给流进程
        self.shared_data[SharedKey.STREAM_WAIT_COUNTER] += 1
