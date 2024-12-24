from zero.core.info.base_info import BaseInfo


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
        super().__init__(data)  # 前面是声明，一定要最后调用这段赋值


