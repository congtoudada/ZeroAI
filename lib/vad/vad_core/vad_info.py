from zero.info.based_stream_info import BasedStreamInfo


class VadInfo(BasedStreamInfo):
    def __init__(self, data: dict):
        # ------------------- 帧级别异常检测 -------------------
        self.vad_frame_enable = False  # 是否支持帧级异常
        self.vad_frame_batch_optimize = True  # 是否优化batch
        self.vad_frame_config = ""  # 帧级别异常检测配置
        self.vad_frame_resize = [320, 640]  # (h,w)
        self.vad_frame_threshold = 0.33  # 帧级别异常阈值 高于该阈值视为异常
        self.vad_frame_times = 1  # 异常倍数
        self.vad_frame_valid = 3  # 连续几帧检测到异常才视为异常 (受异常倍数影响)
        self.vad_frame_nums = 5  # 帧级别异常单次执行至少需要的图片数
        self.vad_frame_interval = 5  # 每隔n帧进行一次异常检测，如果检测到异常则n帧都视为异常
        # ------------------- 对象级别异常检测 -------------------
        self.vad_obj_enable = True  # 是否支持对象级异常
        self.vad_obj_config = ""  # 对象级异常检测配置
        self.vad_obj_spatial_threshold = 0.95  # 空间异常阈值  高于该阈值视为异常
        self.vad_obj_temporal_threshold = 0.9  # 时间异常阈值  高于该阈值视为异常
        self.vad_obj_s_times = 2  # 空间异常倍数
        self.vad_obj_t_times = 3  # 空间异常倍数
        self.vad_obj_valid = 6  # 连续几帧检测到异常才视为异常 (受异常倍数影响)
        self.vad_obj_det_conf = 0.8  # 置信度
        self.vad_obj_nums = 7  # 对象级别异常单次执行至少需要的图片数 (也是sample_nums)
        self.vad_obj_interval = 5  # 每隔n帧进行一次异常检测，如果检测到异常则n帧都视为异常
        super().__init__(data)
