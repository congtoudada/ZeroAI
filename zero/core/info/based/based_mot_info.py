from zero.core.info.based.based_det_info import BasedDetInfo
from zero.core.key.shared_key import SharedKey


class BasedMOTInfo(BasedDetInfo):
    def __init__(self, data: dict = None):
        self.input_port = []  # 输入端口
        super().__init__(data)
        # ----------------------------------------- input -----------------------------------------
        self.MOT_INFO = f"{SharedKey.MOT_INFO.name}-{self.input_port[0]}"
