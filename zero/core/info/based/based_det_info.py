from zero.core.info.based.based_stream_info import BasedStreamInfo
from zero.core.key.shared_key import SharedKey


class BasedDetInfo(BasedStreamInfo):
    def __init__(self, data: dict = None):
        self.input_port = []  # 输入端口
        super().__init__(data)
        # ----------------------------------------- input -----------------------------------------
        self.DETECTION_INFO = f"{SharedKey.DETECTION_INFO.name}-{self.input_port[0]}"
        self.DETECTION_TEST_SIZE = f"{SharedKey.DETECTION_TEST_SIZE.name}-{self.input_port[0]}"


