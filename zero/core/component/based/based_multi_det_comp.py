import numpy as np

from zero.core.component.based.based_stream_comp import BasedStreamComponent
from zero.core.info.based.based_multi_det_info import BasedMultiDetInfo
from zero.core.key.shared_key import SharedKey


class BasedMultiDetComponent(BasedStreamComponent):
    """
    基于目标检测的算法组件（同视频流不同检测模型）
    """
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.config: BasedMultiDetInfo = None  # 由子类加载
        self.input_det = []  # 目标检测算法的输出作为该组件的输入（多个）
        self.ports_len = 0

    def on_start(self):
        super().on_start()
        self.ports_len = len(self.config.input_port)
        self.input_det = [None] * self.ports_len
        # list -> int
        self.stream_width = int(self.stream_width[0])
        self.stream_height = int(self.stream_height[0])
        self.stream_channel = int(self.stream_channel[0])
        self.stream_fps = self.stream_fps[0]
        self.stream_url = self.stream_url[0]
        self.stream_cam_id = self.stream_cam_id[0]
        self.current_frame_id = self.current_frame_id[0]


    def on_resolve_stream(self) -> bool:
        """
        # output shape: [n, 6]
        # n: n个对象
        # [0,1,2,3]: tlbr bboxes (基于视频流分辨率)
        #   [0]: x1
        #   [1]: y1
        #   [2]: x2
        #   [3]: y2
        # [4]: 置信度
        # [5]: 类别 (下标从0开始)
        """
        # 取的是同一个视频流，不同检测结果的帧差距可以忽略不计，因此在最后统一更新id和帧信息
        det_info = self.shared_data[self.config.DETECTION_INFO[self.ports_len - 1]]  # 取最后一个模型的信息用于判断是否更新id和frame
        if det_info is not None and self.current_frame_id != int(det_info[SharedKey.DETECTION_ID]):
            self.current_frame_id = int(det_info[SharedKey.DETECTION_ID])  # 更新帧id
            self.frame = np.reshape(np.ascontiguousarray(np.copy(det_info[SharedKey.DETECTION_FRAME])),
                                    (self.stream_height, self.stream_width, self.stream_channel))  # 更新帧
            for i in range(self.ports_len):
                det_output = self.shared_data[self.config.DETECTION_INFO[i]]  # 取出每一个目标检测算法的检测结果
                self.input_det[i] = det_output[SharedKey.DETECTION_OUTPUT]  # 更新结果
            return True
        else:
            return False
