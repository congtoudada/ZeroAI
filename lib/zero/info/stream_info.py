from zero.info.base_info import BaseInfo


class StreamInfo(BaseInfo):
    def __init__(self, data: dict = None):
        self.stream_cam_id = 0  # 摄像头编号
        self.output_port = ""  # 输出端口
        self.stream_runtime_enable = False  # 是否以实时流运行（如果是，会在初始化后连续丢帧）
        self.stream_runtime_drop_count = 10  # 实时流初始化后连丢n帧再运行
        self.stream_url = ""  # 取流地址
        self.stream_read_frequency = 2  # 读取频率（越大性能越好，准确率越低） eg.2: 每2帧里面读1帧
        self.stream_reduce_scale = 2  # 画面缩小倍数
        super().__init__(data)   # 前面是声明，一定要最后调用这段赋值

