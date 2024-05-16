from zero.core.info.based.based_stream_info import BasedStreamInfo
from zero.core.key.shared_key import SharedKey


class BasedMultiMOTInfo(BasedStreamInfo):
    def __init__(self, data: dict = None):
        self.input_port = []  # 输入端口
        super().__init__(data)
        # ----------------------------------------- input -----------------------------------------
        # self.MOT_INFO = f"{SharedKey.MOT_INFO.name}-{self.input_port[0]}"
        self.MOT_INFO = []
        for port in self.input_port:
            self.MOT_INFO.append(f"{SharedKey.MOT_INFO.name}-{port}")