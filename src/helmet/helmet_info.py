from zero.info.based_stream_info import BasedStreamInfo


class HelmetInfo(BasedStreamInfo):
    def __init__(self, data: dict = None):
        self.detection_labels = []  # 安全帽检测类别
        self.helmet_valid_count = 5  # 对象有效帧阈值
        self.helmet_lost_frame = 60  # 消失多少帧丢弃
        self.helmet_y_sort = False  # 分配id前进行y轴排序（时间换精度）
        self.helmet_zone = []  # 安全帽检测区域 ltrb百分比 <-> [0][1][2][3]
        super().__init__(data)  # 前面是声明，一定要最后调用这段赋值
