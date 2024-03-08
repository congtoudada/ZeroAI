from zero.core.info.based.based_det_info import BasedDetInfo
from zero.core.key.shared_key import SharedKey


class BaseMOTInfo(BasedDetInfo):
    def __init__(self, data: dict = None):
        self.output_port = ""  # 输出端口
        super().__init__(data)
        # ----------------------------------------- output -----------------------------------------
        self.MOT_INFO = ""
        # 算法输出结果 shape: [n, 7]
        # n: n个对象
        # [0,1,2,3]: ltrb bboxes (基于视频流分辨率)
        #   [0]: x1
        #   [1]: y1
        #   [2]: x2
        #   [3]: y2
        # [4]: 置信度
        # [5]: 类别 (下标从0开始)
        # [6]: id
        self.MOT_INFO = f"{SharedKey.MOT_INFO.name}-{self.output_port}"  # 多目标追踪算法信息 (package)

