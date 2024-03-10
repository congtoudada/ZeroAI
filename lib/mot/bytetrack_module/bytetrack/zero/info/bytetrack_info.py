from zero.core.info.base.base_mot_info import BaseMOTInfo
from zero.core.key.shared_key import SharedKey


class BytetrackInfo(BaseMOTInfo):
    def __init__(self, data: dict = None):
        self.bytetrack_args_fps = 30
        self.bytetrack_args_thresh = 0.5
        self.bytetrack_args_buffer = 30
        self.bytetrack_args_match_thresh = 0.8
        self.bytetrack_args_aspect_ratio_thresh = 5
        self.bytetrack_args_min_box_area = 5
        self.bytetrack_args_mot20 = False
        super().__init__(data)  # 前面是声明，一定要最后调用这段赋值


