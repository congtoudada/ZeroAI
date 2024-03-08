from zero.core.info.based.based_stream_info import BasedStreamInfo
from zero.core.key.shared_key import SharedKey


class BasedMOTInfo(BasedStreamInfo):
    def __init__(self, data: dict = None):
        self.input_port = []  # 输入端口
        super().__init__(data)
        # ----------------------------------------- input -----------------------------------------
        self.MOT_INFO = f"{SharedKey.MOT_INFO.name}-{self.input_port[0]}"
