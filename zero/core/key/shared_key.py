from enum import Enum


class SharedKey(Enum):
    """
    单个摄像头进程内，摄像头与算法共享数据的Key常量
    """
    """
    全局
    """
    EVENT_ESC = 0  # 退出事件
    WAIT_COUNTER = 1  # 服务初始化等待计数器（取决于服务数量，初始化用）
    CAMERAS = 2  # 相机dict列表
    LOCK = 3  # 锁
    # ------------------ 以下Key对global_shared_data无效 ------------------
    """
    视频流
    """
    STREAM_GLOBAL = 100  # 全局共享内存的引用
    STREAM_WAIT_COUNTER = 101  # 视频流初始化等待计数器
    STREAM_WAIT_COUNTER_MAX = 102  # 视频流初始化等待计数器最大值（取决于算法数量，初始化用）
    # ---
    STREAM_FRAME_INFO = 103  # 视频流信息 (package)
    STREAM_FRAME_ID = 104  # 原始图像ID（每次成功读取新的FRAME都会更新ID，避免算法重复处理相同帧）
    STREAM_FRAME = 105  # 原始图像
    # STREAM_FRAME_TIME = 113  # Time测试
    # ---
    STREAM_ORIGINAL_WIDTH = 106  # 原始图像宽
    STREAM_ORIGINAL_HEIGHT = 107  # 原始图像高
    STREAM_ORIGINAL_CHANNEL = 108  # 原始图像通道数
    STREAM_ORIGINAL_FPS = 109  # 原始视频图像帧率
    STREAM_URL = 110  # 摄像头取流地址
    STREAM_CAMERA_ID = 111  # 摄像头id
    STREAM_UPDATE_FPS = 112  # 算法最小update间隔
    """
    目标检测
    """
    # ---
    DETECTION_INFO = 200  # 检测算法信息 (package)
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
    MOT_INFO = 300  # 多目标追踪算法信息 (package)
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
