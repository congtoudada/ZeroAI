from enum import Enum


class StreamKey(Enum):
    """
    视频流共享内存Key
    使用举例: StreamKey.STREAM_PORT.name（非端口相关不用）
    """
    STREAM_PACKAGE_FRAME_ID = 0  # 帧ID
    STREAM_PACKAGE_FRAME = 1  # 帧图像
    STREAM_CAM_ID = 2  # 摄像头id
    STREAM_WIDTH = 3  # 摄像头宽
    STREAM_HEIGHT = 4  # 摄像头高
    STREAM_FPS = 5  # 摄像头帧率

