from zero.info.based_stream_info import BasedStreamInfo


class P9ReidTestInfo(BasedStreamInfo):
    def __init__(self, data: dict = None):
        self.reid_helper_config = "conf/dev/service/reid/helper/fast_reid_helper.yaml"
        super().__init__(data)
