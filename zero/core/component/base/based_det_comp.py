import numpy as np
from loguru import logger

from zero.core.component.base.based_stream_comp import BasedStreamComponent
from zero.core.key.shared_key import SharedKey


class BasedDetComponent(BasedStreamComponent):
    """
    基于目标检测的算法组件
    """
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.input_port = ""
        self.input_det = None  # 目标检测算法的输出作为该组件的输入

    def on_update(self) -> bool:
        # 只有不同帧才有必要计算
        det_info = self.shared_data[SharedKey.DETECTION_INFO.name + self.input_port]
        if det_info is not None and self.current_frame_id != int(det_info[SharedKey.DETECTION_ID]):
            self.current_frame_id = int(det_info[SharedKey.DETECTION_ID])
            self.frame = np.reshape(np.ascontiguousarray(np.copy(det_info[SharedKey.DETECTION_FRAME])),
                                    (self.height, self.width, self.channel))
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
            self.input_det = det_info[SharedKey.DETECTION_OUTPUT]
            return True
        return False
