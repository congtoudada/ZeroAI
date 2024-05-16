import numpy as np

from zero.core.component.based.based_det_comp import BasedDetComponent
from zero.core.component.based.based_multi_det_comp import BasedMultiDetComponent
from zero.core.info.based.based_mot_info import BasedMOTInfo
from zero.core.info.based.based_multi_mot_info import BasedMultiMOTInfo
from zero.core.key.shared_key import SharedKey


class BasedMultiMOTComponent(BasedMultiDetComponent):
    """
    基于多目标追踪的算法组件
    """
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.config: BasedMultiMOTInfo = None  # 由子类加载
        self.ports_len = 0
        self.input_mot = []  # 多目标追踪算法的输出作为该组件的输入

    def on_start(self):
        super().on_start()
        self.ports_len = len(self.config.input_port)
        self.input_mot = [None] * self.ports_len

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
        # 取的是同一个视频流，不同检测结果的帧差距可以忽略不计，因此在最后统一更新id和帧信息
        mot_info = self.shared_data[self.config.MOT_INFO[self.ports_len - 1]]  # 取最后一个模型的信息用于判断是否更新id和frame
        if mot_info is not None and self.current_frame_id != int(mot_info[SharedKey.MOT_ID]):
            self.current_frame_id = int(mot_info[SharedKey.MOT_ID])
            self.frame = np.reshape(np.ascontiguousarray(np.copy(mot_info[SharedKey.MOT_FRAME])),
                                    (self.stream_height, self.stream_width, self.stream_channel))
            for i in range(self.ports_len):
                mot_output = self.shared_data[self.config.MOT_INFO[i]]  # 取出每一个目标检测算法的检测结果
                if mot_output is not None:
                    self.input_mot[i] = mot_output[SharedKey.DETECTION_OUTPUT]  # 更新结果
            return True
        else:
            return False
