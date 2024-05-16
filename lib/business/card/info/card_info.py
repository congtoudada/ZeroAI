from zero.core.info.based.based_mot_info import BasedMOTInfo


class CardInfo(BasedMOTInfo):
    def __init__(self, data: dict = None):
        self.count_input_port = ""  # 输入端口
        self.count_output_port = ""  # 输出端口
        # self.card_area = []  # 刷卡点矩形区域，格式为 [x1, y1, x2, y2]
        # self.gate_line = []  # 大门线，格式为 [x1, y1, x2, y2]
        self.card_red = []  # 点集1
        self.card_green = []  # 点集2
        self.item_lost_frames = 60  # 对象消失多少帧则丢弃 (业务层)
        self.card_lost_frames = 60
        self.item_valid_frames = 5  #
        self.item_base = 0  # 检测基准 0:包围盒中心点 1:包围盒左上角
        self.draw_warning_time = 10  # 警告信息显示的帧数
        # self.card_ABC = 0
        super().__init__(data)  # 前面是声明，一定要最后调用这段赋值
