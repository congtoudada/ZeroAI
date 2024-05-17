import numpy as np

from zero.core.component.based.based_stream_comp import BasedStreamComponent
from zero.core.info.based.based_det_info import BasedDetInfo
from zero.core.key.shared_key import SharedKey


class BasedDetComponent(BasedStreamComponent):
    """
    基于目标检测的算法组件
    """
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.config: BasedDetInfo = None  # 由子类加载
        self.input_det = None  # 目标检测算法的输出作为该组件的输入

    def on_start(self):
        super().on_start()
        # list -> int
        self.stream_width = int(self.stream_width[0])
        self.stream_height = int(self.stream_height[0])
        self.stream_channel = int(self.stream_channel[0])
        self.stream_fps = self.stream_fps[0]
        self.stream_url = self.stream_url[0]
        self.stream_cam_id = self.stream_cam_id[0]
        self.current_frame_id = self.current_frame_id[0]
        self.output_dir = self.output_dir[0]

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
        # 只有不同帧才有必要计算
        det_info = self.shared_data[self.config.DETECTION_INFO]
        if det_info is not None and self.current_frame_id != int(det_info[SharedKey.DETECTION_ID]):
            self.current_frame_id = int(det_info[SharedKey.DETECTION_ID])
            self.frame = np.reshape(np.ascontiguousarray(np.copy(det_info[SharedKey.DETECTION_FRAME])),
                                    (self.stream_height, self.stream_width, self.stream_channel))
            self.input_det = det_info[SharedKey.DETECTION_OUTPUT]
            return True
        else:
            return False
