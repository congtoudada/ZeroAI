from zero.core.info.based.based_multi_mot_info import BasedMultiMOTInfo


class PhoneInfo(BasedMultiMOTInfo):
    def __init__(self, data: dict = None):
        # 声明组件需要的参数，由外部配置文件配置
        self.warning_path = "output/business/phone/warning"
        self.timing_path = "output/business/phone/timing"
        self.timing_enable = True
        super().__init__(data)  # 前面是声明，一定要最后调用这段赋值
