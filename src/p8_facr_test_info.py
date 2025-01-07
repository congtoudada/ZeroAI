from zero.info.based_stream_info import BasedStreamInfo


class P8FacrTestInfo(BasedStreamInfo):
    def __init__(self, data: dict = None):
        self.face_helper_config = "conf/dev/service/facr/insight/face_helper.yaml"
        super().__init__(data)
