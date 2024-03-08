import numpy as np

from zero.core.component.based.based_det_comp import BasedDetComponent
from zero.core.info.based.based_mot_info import BasedMOTInfo
from zero.core.key.shared_key import SharedKey


class BasedMOTComponent(BasedDetComponent):
    """
    基于多目标追踪的算法组件
    """
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.config: BasedMOTInfo = None  # 由子类加载
        self.input_mot = None  # 多目标追踪算法的输出作为该组件的输入

    def on_resolve_stream(self) -> bool:
        """
        # mot output shape: [n, 7]
        # n: n个对象
        # [0,1,2,3]: tlbr bboxes (基于视频流分辨率)
        #   [0]: x1
        #   [1]: y1
        #   [2]: x2
        #   [3]: y2
        # [4]: 置信度
        # [5]: 类别 (下标从0开始)
        # [6]: id
        """
        # 只有不同帧才有必要计算
        mot_info = self.shared_data[self.config.MOT_INFO]
        if mot_info is not None and self.current_frame_id != int(mot_info[SharedKey.MOT_ID]):
            self.current_frame_id = int(mot_info[SharedKey.MOT_ID])
            self.frame = np.reshape(np.ascontiguousarray(np.copy(mot_info[SharedKey.MOT_FRAME])),
                                    (self.stream_height, self.stream_width, self.stream_channel))
            self.input_mot = mot_info[SharedKey.MOT_OUTPUT]
            return True
        else:
            return False
