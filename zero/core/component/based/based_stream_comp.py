import time
from typing import List

from loguru import logger
import numpy as np

from zero.core.component.base.component import Component
from zero.core.component.helper.feature.save_video_helper_comp import SaveVideoHelperComponent
from zero.core.info.based.based_stream_info import BasedStreamInfo
from zero.core.key.shared_key import SharedKey
from zero.utility.timer_kit import TimerKit


class BasedStreamComponent(Component):
    """
    基于视频流的算法组件
    """
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.config: BasedStreamInfo = None  # 由子类完成初始化
        self.stream_width = []
        self.stream_height = []
        self.stream_channel = []
        self.stream_fps = []
        self.stream_url = []
        self.stream_cam_id = []
        self.update_fps = []
        self.current_frame_id = []
        self.frame = None
        self.update_timer = TimerKit()  # 计算两帧时间
        self._tic = False
        self.cur_stream_idx = 0  # 当前取流索引

    def on_start(self):
        super().on_start()
        for i in range(len(self.config.input_port)):
            self.stream_width.append(int(self.shared_data[self.config.STREAM_ORIGINAL_WIDTH[i]]))
            self.stream_height.append(int(self.shared_data[self.config.STREAM_ORIGINAL_HEIGHT[i]]))
            self.stream_channel.append(int(self.shared_data[self.config.STREAM_ORIGINAL_CHANNEL[i]]))
            self.stream_fps.append(self.shared_data[self.config.STREAM_ORIGINAL_FPS[i]])
            self.stream_url.append(self.shared_data[self.config.STREAM_URL[i]])
            self.stream_cam_id.append(self.shared_data[self.config.STREAM_CAMERA_ID[i]])
            self.update_fps.append(self.shared_data[self.config.STREAM_UPDATE_FPS[i]])
            self.current_frame_id.append(0)

    def on_resolve_stream(self) -> bool:
        # 只有不同帧才有必要计算
        self.cur_stream_idx = (self.cur_stream_idx + 1) % len(self.config.STREAM_FRAME_INFO)
        for i, info_key in enumerate(self.config.STREAM_FRAME_INFO):
            if i == self.cur_stream_idx:
                frame_info = self.shared_data[info_key]
                if frame_info is not None and self.current_frame_id[i] != int(frame_info[SharedKey.STREAM_FRAME_ID]):
                    self.current_frame_id[i] = int(frame_info[SharedKey.STREAM_FRAME_ID])
                    self.frame = np.reshape(np.ascontiguousarray(np.copy(frame_info[SharedKey.STREAM_FRAME])),
                                            (self.stream_height[i], self.stream_width[i], self.stream_channel[i]))
                    self.analysis(frame_info[SharedKey.STREAM_FRAME_ID])  # 打印性能分析报告
                    return True
        return False

    def on_update(self) -> bool:
        super().on_update()
        if self.update_fps[self.cur_stream_idx] > 0:
            time.sleep(1.0 / self.update_fps[self.cur_stream_idx])
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
        # 在初始化结束通知给流进程
        self.shared_data[SharedKey.STREAM_WAIT_COUNTER] += 1
