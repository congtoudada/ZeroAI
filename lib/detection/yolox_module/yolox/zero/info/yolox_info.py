from zero.core.info.base.base_det_info import BaseDetInfo


class YoloxInfo(BaseDetInfo):
    def __init__(self, data: dict = None):
        self.yolox_vis = False  # 是否使用opencv可视化（测试用）
        self.yolox_args_expn = ""  # 实验名称
        self.yolox_args_path = None  # 取流路径（为None则通过内部框架取流）
        # self.yolox_args_save_result = False  # 是否存储视频
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
        self.yolox_save_video = False  # 是否存视频
        self.yolox_output_dir = False  # 输出目录
        super().__init__(data)  # 前面是声明，一定要最后调用这段赋值


