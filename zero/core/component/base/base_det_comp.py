import numpy as np

from zero.core.component.base.based_stream_comp import BasedStreamComponent
from zero.core.key.shared_key import SharedKey


class BaseDetComponent(BasedStreamComponent):
    """
    TODO: 目标检测算法基类（时间有限，未完待续）
    """
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.output_port = ""  # 输出端口
        self.inference_outputs = None  # 推理结果
        self.output_detect_info = {}   # 进程间共享检测信息

    def on_start(self):
        super().on_start()
        self.shared_data[SharedKey.DETECTION_INFO.name + self.output_port] = None

    def resolve_output(self, inference_outputs):
        self.output_detect_info[SharedKey.DETECTION_ID] = self.current_frame_id
        self.output_detect_info[SharedKey.DETECTION_FRAME] = self.frame
        self.output_detect_info[SharedKey.DETECTION_OUTPUT] = self.on_resolve_output(inference_outputs)
        self.shared_data[SharedKey.DETECTION_INFO.name + self.output_port] = self.output_detect_info  # 填充输出

    def on_resolve_output(self, inference_outputs):
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
        :return:
        """
        return None
