import numpy as np

from zero.core.component.base.based_stream_comp import BasedStreamComponent
from zero.core.key.shared_key import SharedKey


class BasedMOTComponent(BasedStreamComponent):
    """
    基于多目标追踪的算法组件
    """
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.input_port = ""
        self.input_mot = None  # 多目标追踪算法的输出作为该组件的输入

    def on_start(self):
        super().on_start()

    def on_update(self) -> bool:
        # 只有不同帧才有必要计算
        mot_info = self.shared_data[SharedKey.MOT_INFO.name + self.input_port]
        if mot_info is not None and self.current_frame_id != int(mot_info[SharedKey.MOT_ID]):
            self.current_frame_id = int(mot_info[SharedKey.MOT_ID])
            self.frame = np.reshape(np.ascontiguousarray(np.copy(mot_info[SharedKey.MOT_FRAME])),
                                    (self.height, self.width, self.channel))
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
            self.input_mot = mot_info[SharedKey.MOT_OUTPUT]
            return True
        return False
