from zero.info.based_stream_info import BasedStreamInfo


class CardInfo(BasedStreamInfo):
    def __init__(self, data: dict = None):
        # self.card_area = []  # 刷卡点矩形区域，格式为 [x1, y1, x2, y2]
        # self.gate_line = []  # 大门线，格式为 [x1, y1, x2, y2]
        self.card_red = []  # 点集1
        self.card_green = []  # 点集2
        self.card_lost_frames = 60  # card区的obj消失多少帧则丢弃,根据从car区到gate区所需时间确定
        self.card_item_base = 0  # 检测基准 0:包围盒中心点 1:包围盒左上角
        self.card_item_valid_frames = 1  # 对象稳定出现多少帧，才开始计算
        self.card_item_lost_frames = 60  # 对象消失多少帧则丢弃 (业务层)
        self.card_warning_frame = 10  # 警告信息显示的帧数
        self.card_valid_zone = [0, 0, 1, 1]  # 包围盒有效运动区域
        # self.card_ABC = 0
        super().__init__(data)  # 前面是声明，一定要最后调用这段赋值
