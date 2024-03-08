from zero.core.component.based.based_stream_comp import BasedStreamComponent
from zero.core.info.base.base_det_info import BaseDetInfo
from zero.core.key.shared_key import SharedKey


class BaseDetComponent(BasedStreamComponent):
    """
    TODO: 目标检测算法基类
    """
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.config: BaseDetInfo = None
        self.inference_outputs = None  # 推理结果
        self.output_detect_info = {}   # 进程间共享检测信息

    def on_start(self):
        super().on_start()
        for info_key in self.config.DETECTION_INFO:
            self.shared_data[info_key] = None

    def resolve_output(self, inference_outputs):
        self.output_detect_info.clear()
        self.output_detect_info[SharedKey.DETECTION_ID] = self.current_frame_id[self.cur_stream_idx]
        self.output_detect_info[SharedKey.DETECTION_FRAME] = self.frame  # 理论上需要拷贝一份
        self.output_detect_info[SharedKey.DETECTION_OUTPUT] = self.on_resolve_output(inference_outputs)
        output_port = self.config.DETECTION_INFO[self.cur_stream_idx]
        self.shared_data[output_port] = self.output_detect_info  # 填充输出

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
