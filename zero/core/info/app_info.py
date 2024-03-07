from zero.core.info.base_info import BaseInfo


class AppInfo(BaseInfo):
    def __init__(self, data: dict = None):
        self.cam_list = []
        self.service = []
        super().__init__(data)  # 前面是声明，一定要最后调用这段赋值
