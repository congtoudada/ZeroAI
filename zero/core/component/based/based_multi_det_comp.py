import numpy as np

from zero.core.component.based.based_stream_comp import BasedStreamComponent
from zero.core.info.based.based_multi_det_info import BasedMultiDetInfo
from zero.core.key.shared_key import SharedKey


class BasedMultiDetComponent(BasedStreamComponent):
    """
    基于目标检测的算法组件
    """
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.config: BasedMultiDetInfo = None  # 由子类加载
        self.input_det = []  # 目标检测算法的输出作为该组件的输入（多个）

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
        for i in range(len(self.config.DETECTION_INFO)):
            det_info = self.shared_data[self.config.DETECTION_INFO[i]]  # 取出每一个目标检测算法的检测结果
            if det_info is not None and self.current_frame_id[i] != int(det_info[SharedKey.DETECTION_ID]):
                self.current_frame_id[i] = int(det_info[SharedKey.DETECTION_ID])  # 更新帧id
                self.frame[i] = np.reshape(np.ascontiguousarray(np.copy(det_info[SharedKey.DETECTION_FRAME])),
                                        (self.stream_height, self.stream_width, self.stream_channel))  # 更新帧
                self.input_det[i] = det_info[SharedKey.DETECTION_OUTPUT]  # 更新结果
                return True
            else:
                return False
