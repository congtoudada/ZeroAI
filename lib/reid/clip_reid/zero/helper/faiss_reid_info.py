from zero.info.base_info import BaseInfo


class FaissReidInfo(BaseInfo):
    def __init__(self, data: dict = None):
        self.dimension = 1280  # 特征维度
        self.refresh_interval = 30 * 60 * 30  # 每隔refresh_interval帧刷新一次半区 ( 帧 秒 分钟 )
        super().__init__(data)  # 前面是声明，一定要最后调用这段赋值
