from zero.info.based_stream_info import BasedStreamInfo


class YoloxInfo(BasedStreamInfo):
    def __init__(self, data: dict = None):
        self.detection_reid_enable = False  # 是否支持reid存图(若支持，会定期对第0个类别发送reid存图请求)
        self.detection_reid_interval = 30  # 间隔多少帧发送一次reid存图请求
        self.yolox_args_expn = ""  # 实验名称
        self.yolox_args_path = None  # 取流路径（为None则通过内部框架取流）
        self.yolox_args_exp_file = ""  # 通过.py选择模型
        self.yolox_args_ckpt = ""  # 推理用模型权重文件
        self.yolox_args_conf = 0.67  # test conf
        self.yolox_args_name = None  # 通过模型名选择预置模型（建议使用exp_file自定义选择）
        self.yolox_args_camid = 0  # webcam demo camera id（含摄像头设备才需要，一般不设置）
        self.yolox_args_device = "gpu"  # 运行设备
        self.yolox_args_nms = 0.7  # test nms threshold
        self.yolox_args_tsize = 640  # test img size
        self.yolox_args_fp16 = False  # Adopting mix precision evaluating.
        self.yolox_args_fuse = False  # Fuse conv and bn for testing.
        self.yolox_args_trt = False  # Using TensorRT model for testing.
        self.detection_labels = []  # 类别标签（仅用于可视化）
        super().__init__(data)  # 前面是声明，一定要最后调用这段赋值


