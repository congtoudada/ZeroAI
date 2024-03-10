from zero.core.info.based.based_stream_info import BasedStreamInfo
from zero.core.key.shared_key import SharedKey


class BaseDetInfo(BasedStreamInfo):
    def __init__(self, data: dict = None):
        self.output_port = ""  # 输出端口
        self.detection_labels = []  # 类别标签
        super().__init__(data)
        # ----------------------------------------- output -----------------------------------------
        # 算法输出结果 shape: [n, 6]
        # n: n个对象
        # [0,1,2,3]: ltrb bboxes (基于视频流分辨率)
        #   [0]: x1
        #   [1]: y1
        #   [2]: x2
        #   [3]: y2
        # [4]: 置信度
        # [5]: 类别 (下标从0开始)
        self.DETECTION_INFO = []  # 检测算法信息 (package) 输出端口 eg.SharedKey.STREAM_FRAME_INFO+yolox+door1
        self.DETECTION_TEST_SIZE = []  # 目标检测输入尺寸 (暂时没用)
        for input_port in self.input_port:
            self.DETECTION_INFO.append(f"{SharedKey.DETECTION_INFO.name}-{input_port}-{self.output_port}")
            self.DETECTION_TEST_SIZE.append(
                f"{SharedKey.DETECTION_TEST_SIZE.name}-{input_port}-{self.output_port}")
