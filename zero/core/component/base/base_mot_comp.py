from zero.core.component.based.based_det_comp import BasedDetComponent
from zero.core.info.base.base_mot_info import BaseMOTInfo
from zero.core.key.shared_key import SharedKey


class BaseMOTComponent(BasedDetComponent):
    """
    TODO: 多目标追踪算法基类
    """
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.config: BaseMOTInfo = None
        self.inference_outputs = None  # 推理结果
        self.output_mot_info = {}   # 进程间共享检测信息

    def on_start(self):
        super().on_start()
        self.shared_data[self.config.MOT_INFO] = None

    def resolve_output(self, inference_outputs):
        self.output_mot_info.clear()
        self.output_mot_info[SharedKey.MOT_ID] = self.current_frame_id
        self.output_mot_info[SharedKey.MOT_FRAME] = self.frame.flatten()
        self.output_mot_info[SharedKey.MOT_OUTPUT] = self.on_resolve_output(inference_outputs)
        self.shared_data[self.config.MOT_INFO] = self.output_mot_info  # 填充输出

    def get_color(self, idx):
        idx = idx * 3
        color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
        return color

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
        # [6]: id
        :return:
        """
        return None
