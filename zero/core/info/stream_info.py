from zero.core.info.base_info import BaseInfo


class StreamInfo(BaseInfo):
    def __init__(self, data: dict = None):
        self.stream_cam_id = 0  # 摄像头编号
        self.stream_url = ""  # 取流地址
        self.stream_algorithm = []  # 算法组
        self.stream_delay = False  # 是否以delay模式运行
        self.stream_delay_speed = 1  # delay加速
        self.stream_frame_history_capacity = 0  # 回放视频帧缓存上限 (决定了抓拍视频从之前帧开始的程度，该值越大显存占用越大）
        self.stream_drop_interval = 0  # 手动丢帧数（越大性能越差，准确率高） eg.2: 每2帧里面丢1帧
        self.stream_width = 640  # 取流重缩放宽
        self.stream_height = 640  # 取流重缩放高
        self.stream_channel = 3  # 取流通道数
        self.stream_update_fps = -1  # 算法更新帧率
        self.face_helper = ""
        super().__init__(data)   # 前面是声明，一定要最后调用这段赋值
