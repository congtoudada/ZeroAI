from zero.info.base_info import BaseInfo


class BytetrackInfo(BaseInfo):
    def __init__(self, data: dict = None):
        self.bytetrack_args_fps = 30
        self.bytetrack_args_thresh = 0.5
        self.bytetrack_args_buffer = 30
        self.bytetrack_args_match_thresh = 0.8
        self.bytetrack_args_aspect_ratio_thresh = 5
        self.bytetrack_args_min_box_area = 5
        self.bytetrack_args_mot20 = False
        self.detection_labels = []
        # reid存图支持
        self.bytetrack_reid_enable = False  # 是否支持reid存图(若支持，会定期对第0个类别发送reid存图请求)
        self.bytetrack_reid_conf = 0.7  # reid存图阈值
        self.bytetrack_reid_min_interval = 30  # reid存图最小间隔(支持reid存图才有效)
        self.bytetrack_reid_max_interval = 300  # reid存图最大间隔(支持reid存图才有效)
        self.bytetrack_reid_lost_frames = 180  # 对象消失多少帧则销毁
        self.bytetrack_reid_cull_up_y = 0.2  # 向上剔除百分比
        self.bytetrack_reid_cull_down_y = 0.2  # 向下剔除百分比
        self.bytetrack_reid_camera_gallery = "output/service/clip_reid/camera_gallery"  # 存图路径(支持reid存图才有效)
        super().__init__(data)  # 前面是声明，一定要最后调用这段赋值


