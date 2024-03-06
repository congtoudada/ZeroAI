from enum import Enum


class SharedKey(Enum):
    """
    单个摄像头进程内，摄像头与算法共享数据的Key常量
    """
    """
    全局
    """
    EVENT_ESC = 0  # 退出事件
    """
    视频流
    """
    STREAM_WAIT_COUNTER = 100  # 视频流初始化等待计数器（取决于算法数量，初始化用）
    STREAM_WAIT_MAX = 101  # 视频流开始工作的最大等待数（取决于算法数量，初始化用）
    # ---
    STREAM_FRAME_INFO = 102  # 视频流信息 (ID,Frame信息)
    STREAM_FRAME_ID = 103  # 原始图像ID（每次成功读取新的FRAME都会更新ID，避免算法重复处理相同帧）
    STREAM_FRAME = 104  # 原始图像
    # ---
    STREAM_ORIGINAL_WIDTH = 105  # 原始图像宽
    STREAM_ORIGINAL_HEIGHT = 106  # 原始图像高
    STREAM_ORIGINAL_CHANNEL = 107  # 原始图像通道数
    STREAM_ORIGINAL_FPS = 108  # 原始视频图像帧率
    STREAM_URL = 109  # 摄像头取流地址
    STREAM_CAMERA_ID = 110  # 摄像头id
    """
    目标检测
    """
    # ---
    DETECTION_INFO = 200  # 检测算法信息
    DETECTION_ID = 201  # 当前帧ID（每次成功读取新的FRAME都会更新ID，避免算法重复处理相同帧）
    DETECTION_FRAME = 202  # 读取的检测图像
    # 算法输出结果 shape: [n, 6]
    # n: n个对象
    # [0,1,2,3]: ltrb bboxes (基于视频流分辨率)
    #   [0]: x1
    #   [1]: y1
    #   [2]: x2
    #   [3]: y2
    # [4]: 置信度
    # [5]: 类别 (下标从0开始)
    DETECTION_OUTPUT = 203
    # ---
    DETECTION_TEST_SIZE = 204  # 目标检测输入尺寸 (暂时没用)
    """
    多目标追踪
    """
    # ---
    MOT_INFO = 300  # 多目标追踪算法信息
    MOT_ID = 301  # 当前帧ID（每次成功读取新的FRAME都会更新ID，避免算法重复处理相同帧）
    MOT_FRAME = 302  # 读取的检测图像
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
    MOT_OUTPUT = 303
    # ---


if __name__ == '__main__':
    print(SharedKey.EVENT_ESC)
    print(SharedKey.EVENT_ESC.name + "1")
