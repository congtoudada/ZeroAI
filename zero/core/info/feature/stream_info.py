from zero.core.info.base.base_info import BaseInfo
from zero.core.key.shared_key import SharedKey


class StreamInfo(BaseInfo):
    def __init__(self, data: dict = None):
        self.stream_cam_id = 0  # 摄像头编号
        self.stream_url = ""  # 取流地址
        self.stream_output_port = ""  # 输出端口 eg.SharedKey.STREAM_FRAME_INFO+door1
        self.stream_algorithm = []  # 算法组
        self.stream_delay = False  # 是否以delay模式运行
        self.stream_delay_speed = 1  # delay加速
        self.stream_frame_history_capacity = 0  # 回放视频帧缓存上限 (决定了抓拍视频从之前帧开始的程度，该值越大显存占用越大）
        self.stream_drop_interval = 0  # 手动丢帧数（越大性能越差，准确率高） eg.2: 每2帧里面丢1帧
        self.stream_width = 640  # 取流重缩放宽
        self.stream_height = 640  # 取流重缩放高
        self.stream_channel = 3  # 取流通道数
        self.stream_update_fps = -1  # 算法更新帧率
        super().__init__(data)   # 前面是声明，一定要最后调用这段赋值
        # ----------------------------------------- output -----------------------------------------
        self.STREAM_FRAME_INFO = f"{SharedKey.STREAM_FRAME_INFO.name}-{self.stream_output_port}"  # 视频流信息 (package)
        self.STREAM_ORIGINAL_WIDTH = f"{SharedKey.STREAM_ORIGINAL_WIDTH.name}-{self.stream_output_port}"  # 原始图像宽
        self.STREAM_ORIGINAL_HEIGHT = f"{SharedKey.STREAM_ORIGINAL_HEIGHT.name}-{self.stream_output_port}"  # 原始图像高
        self.STREAM_ORIGINAL_CHANNEL = f"{SharedKey.STREAM_ORIGINAL_CHANNEL.name}-{self.stream_output_port}"  # 原始图像通道数
        self.STREAM_ORIGINAL_FPS = f"{SharedKey.STREAM_ORIGINAL_FPS.name}-{self.stream_output_port}"  # 原始视频图像帧率
        self.STREAM_URL = f"{SharedKey.STREAM_URL.name}-{self.stream_output_port}"  # 摄像头取流地址
        self.STREAM_CAMERA_ID = f"{SharedKey.STREAM_CAMERA_ID.name}-{self.stream_output_port}"  # 摄像头id
        self.STREAM_UPDATE_FPS = f"{SharedKey.STREAM_UPDATE_FPS.name}-{self.stream_output_port}"  # 算法最小update间隔
