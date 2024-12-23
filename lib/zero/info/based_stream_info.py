from zero.info.base_info import BaseInfo


class BasedStreamInfo(BaseInfo):
    def __init__(self, data: dict = None):
        self.input_ports = []  # 输入端口 (eg. camera1, camera2, ...)
        self.output_ports = []  # 输出端口，如果不为[]需要和输入端口一一对应 (eg. xxx1, xxx2, ...)
        self.stream_output_dir = "output/stream"  # 输出目录
        self.stream_output_absolute = False  # 是否导出绝对路径
        self.stream_save_video_enable = False  # 是否存储视频
        self.stream_save_video_resize = False  # 是否resize写入视频（需拷贝一次图像）
        self.stream_save_video_width = 640  # 存储视频宽
        self.stream_save_video_height = 480  # 存储视频高
        self.stream_save_video_fps = 24  # 存储视频帧率
        self.stream_save_video_filename = "exp"  # 导出视频文件名（最终会拼上cam_id）
        self.stream_draw_vis_enable = False  # 是否可视化
        self.stream_draw_vis_resize = False  # 是否缩放可视化分辨率（需要拷贝一次图像）
        self.stream_draw_vis_width = 960  # 可视化宽
        self.stream_draw_vis_height = 540  # 可视化高
        self.stream_export_img_enable = False  # 是否导出图片
        self.stream_web_enable = False  # 是否与web交互
        self.stream_rtsp_enable = False  # 是否推rtsp流
        self.stream_rtsp_url = ""  # rtsp推流路径
        self.stream_mot_config = ""  # mot配置路径
        self.stream_http_config = ""  # http请求帮助类配置路径
        super().__init__(data)
