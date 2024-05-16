from zero.core.info.based.based_mot_info import BasedMOTInfo


class HelmetInfo(BasedMOTInfo):
    def __init__(self, data: dict = None):
        super().__init__(data)  # 前面是声明，一定要最后调用这段赋值