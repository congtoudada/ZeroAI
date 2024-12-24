from enum import Enum


class DetectionKey(Enum):
    """
        目标检测共享内存Key
        使用举例: Global.EVENT_ESC.name（非端口相关不用）
        # output shape: [n, 6]
        # n: n个对象
        # [0,1,2,3]: ltrb bboxes (基于视频流分辨率)
        #   [0]: x1
        #   [1]: y1
        #   [2]: x2
        #   [3]: y2
        # [4]: 置信度
        # [5]: 类别 (下标从0开始)
    """
    DET_PACKAGE_RESULT = 0  # 目标检测推理结果
