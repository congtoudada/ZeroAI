from zero.info.based_stream_info import BasedStreamInfo


class CountInfo(BasedStreamInfo):
    def __init__(self, data: dict = None):
        self.count_base = 1  # 检测基准 0:包围盒中心点 1:包围盒左上角
        self.count_reverse = False  # 默认从上到下为进入，从下到上为离开
        self.count_filter = 0  # 方向过滤 0:不过滤，双向检测 1:过滤进方向 2:过滤出方向
        self.count_lost_frames = 60  # 对象消失多少帧则丢弃
        self.count_valid_frames = 5  # 对象稳定出现多少帧，才开始计算
        self.count_red = []  # 点集1
        self.count_green = []  # 点集2
        super().__init__(data)  # 前面是声明，一定要最后调用这段赋值
