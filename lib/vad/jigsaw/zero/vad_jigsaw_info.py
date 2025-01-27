from zero.info.base_info import BaseInfo


class VadJigsawInfo(BaseInfo):
    def __init__(self, data: dict = None):
        self.vad_jigsaw_ckpt = "pretrained/vad/jigsaw/avenue_92.18.pth"  # ckpt目录(结尾勿带/)
        self.vad_jigsaw_device = "cuda"  # 运行设备
        # self.vad_jigsaw_sample_num = 7  # 单对象匹配帧数  (放到vad_core)
        # self.vad_jigsaw_filter_ratio = 0.8  # 置信度过滤阈值  (放到vad_core)
        self.vad_jigsaw_static_threshold = 0.1  # 静止阈值，低于该阈值视为无运动发生
        self.vad_jigsaw_align_size = [64, 64]  # roi_align提取的特征图尺寸
        super().__init__(data)  # 前面是声明，一定要最后调用这段赋值


