from zero.core.info.based.based_mot_info import BasedMOTInfo


class HelmetInfo(BasedMOTInfo):
    def __init__(self, data: dict = None):
        self.helmet_valid_count = 5  # 对象有效帧阈值
        self.helmet_lost_frame = 60  # 消失多少帧丢弃
        super().__init__(data)  # 前面是声明，一定要最后调用这段赋值
