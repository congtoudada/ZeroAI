from zero.core.info.base.base_info import BaseInfo
from zero.core.key.shared_key import SharedKey


class PrintStreamInfo(BaseInfo):  # 通过继承相应配置文件父类，可以减少字段的编写
    def __init__(self, data: dict = None):
        # 默认值声明通常与配置文件一致，这里为了演示配置重载，设置为系统默认值
        self.input_port = None  # 输入端口
        self.print_stream_show_title = ""
        self.print_stream_show_width = 0
        self.print_stream_show_height = 0
        # 下面是对 conf/global/output.yaml 配置的声明，主要用于存储视频
        # 参考 zero/core/info/based/based_stream_info.py
        self.stream_output_dir = "output/stream"  # 输出目录
        self.stream_save_video_enable = False  # 是否存储视频
        self.stream_save_video_width = 640  # 存储视频宽
        self.stream_save_video_height = 480  # 存储视频高
        self.stream_save_video_fps = 24  # 存储视频帧率
        self.stream_draw_vis_enable = False  # 是否可视化
        super().__init__(data)  # super调用要放在声明之后，本质是由super调用重载方法
        # 调用super之后，所有值均为配置文件值
        # ----------------------------------------- input -----------------------------------------
        # 组件的输入需要依赖视频流，视频流需要从共享内存中获取，这里可以声明SharedKey，方便访问共享内存
        # 这里参考 zero/core/info/based/based_stream_info.py 可以对相应SharedKey进行声明
        # 注意：这里只取一个流，而不是像BasedStreamInfo一样取多个流
        camera = self.input_port  # camera就是输入端口，也是视频流的输出端口
        self.STREAM_FRAME_INFO = f"{SharedKey.STREAM_FRAME_INFO.name}-{camera}"
        # STREAM_FRAME_INFO是视频流组件输出的一个包，每帧获取。
        # 从SharedKey源码可以看到包内含有帧序号和帧图像两个信息
        # ---
        # STREAM_FRAME_INFO = 103  # 视频流信息 (package)
        # STREAM_FRAME_ID = 104  # 原始图像ID（每次成功读取新的FRAME都会更新ID，避免算法重复处理相同帧）
        # STREAM_FRAME = 105  # 原始图像
        # ---
        # 同时，视频流进程还携带有一些摄像头配置（篇幅有限，具体可以参考zero/core/key/shared_key.py）
        self.STREAM_ORIGINAL_WIDTH = f"{SharedKey.STREAM_ORIGINAL_WIDTH.name}-{camera}"
        self.STREAM_ORIGINAL_HEIGHT = f"{SharedKey.STREAM_ORIGINAL_HEIGHT.name}-{camera}"
        self.STREAM_ORIGINAL_CHANNEL = f"{SharedKey.STREAM_ORIGINAL_CHANNEL.name}-{camera}"
        self.STREAM_ORIGINAL_FPS = f"{SharedKey.STREAM_ORIGINAL_FPS.name}-{camera}"
        self.STREAM_URL = f"{SharedKey.STREAM_URL.name}-{camera}"
        self.STREAM_CAMERA_ID = f"{SharedKey.STREAM_CAMERA_ID.name}-{camera}"
        self.STREAM_UPDATE_FPS = f"{SharedKey.STREAM_UPDATE_FPS.name}-{camera}"
