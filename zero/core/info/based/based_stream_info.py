from zero.core.info.base.base_info import BaseInfo
from zero.core.key.shared_key import SharedKey


class BasedStreamInfo(BaseInfo):
    def __init__(self, data: dict = None):
        self.input_port = []  # 输入端口 (可能是数组，也可能是字符串）
        self.stream_output_dir = "output/stream"  # 输出目录
        self.stream_save_video_enable = False  # 是否存储视频
        self.stream_save_video_width = 640  # 存储视频宽
        self.stream_save_video_height = 480  # 存储视频高
        self.stream_save_video_fps = 24  # 存储视频帧率
        self.stream_draw_vis_enable = False  # 是否可视化
        super().__init__(data)
        # ----------------------------------------- input -----------------------------------------
        self.STREAM_FRAME_INFO = []
        self.STREAM_ORIGINAL_WIDTH = []
        self.STREAM_ORIGINAL_HEIGHT = []
        self.STREAM_ORIGINAL_CHANNEL = []
        self.STREAM_ORIGINAL_FPS = []
        self.STREAM_URL = []
        self.STREAM_CAMERA_ID = []
        self.STREAM_UPDATE_FPS = []
        if isinstance(self.input_port, str):  # 单个视频流也初始化成数组
            temp_port = self.input_port
            self.input_port = []
            self.input_port.append(temp_port)
            camera = self.input_port[0].split('-')[0]
            self.STREAM_FRAME_INFO.append(f"{SharedKey.STREAM_FRAME_INFO.name}-{camera}")
            self.STREAM_ORIGINAL_WIDTH.append(f"{SharedKey.STREAM_ORIGINAL_WIDTH.name}-{camera}")
            self.STREAM_ORIGINAL_HEIGHT.append(f"{SharedKey.STREAM_ORIGINAL_HEIGHT.name}-{camera}")
            self.STREAM_ORIGINAL_CHANNEL.append(f"{SharedKey.STREAM_ORIGINAL_CHANNEL.name}-{camera}")
            self.STREAM_ORIGINAL_FPS.append(f"{SharedKey.STREAM_ORIGINAL_FPS.name}-{camera}")
            self.STREAM_URL.append(f"{SharedKey.STREAM_URL.name}-{camera}")
            self.STREAM_CAMERA_ID.append(f"{SharedKey.STREAM_CAMERA_ID.name}-{camera}")
            self.STREAM_UPDATE_FPS.append(f"{SharedKey.STREAM_UPDATE_FPS.name}-{camera}")
        else:
            for input_port in self.input_port:
                camera = input_port.split('-')[0]
                self.STREAM_FRAME_INFO.append(f"{SharedKey.STREAM_FRAME_INFO.name}-{camera}")
                self.STREAM_ORIGINAL_WIDTH.append(f"{SharedKey.STREAM_ORIGINAL_WIDTH.name}-{camera}")
                self.STREAM_ORIGINAL_HEIGHT.append(f"{SharedKey.STREAM_ORIGINAL_HEIGHT.name}-{camera}")
                self.STREAM_ORIGINAL_CHANNEL.append(f"{SharedKey.STREAM_ORIGINAL_CHANNEL.name}-{camera}")
                self.STREAM_ORIGINAL_FPS.append(f"{SharedKey.STREAM_ORIGINAL_FPS.name}-{camera}")
                self.STREAM_URL.append(f"{SharedKey.STREAM_URL.name}-{camera}")
                self.STREAM_CAMERA_ID.append(f"{SharedKey.STREAM_CAMERA_ID.name}-{camera}")
                self.STREAM_UPDATE_FPS.append(f"{SharedKey.STREAM_UPDATE_FPS.name}-{camera}")
