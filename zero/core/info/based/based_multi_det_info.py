from zero.core.info.based.based_stream_info import BasedStreamInfo
from zero.core.key.shared_key import SharedKey


class BasedMultiDetInfo(BasedStreamInfo):
    def __init__(self, data: dict = None):
        self.input_port = []  # 输入端口
        self.detection_labels = []  # 类别标签
        super().__init__(data)
        # ----------------------------------------- input -----------------------------------------
        self.DETECTION_INFO = []
        self.DETECTION_TEST_SIZE = []
        for port in self.input_port:
            self.DETECTION_INFO.append(f"{SharedKey.DETECTION_INFO.name}-{port}")
            self.DETECTION_TEST_SIZE.append(f"{SharedKey.DETECTION_TEST_SIZE.name}-{port}")


