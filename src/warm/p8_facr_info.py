from zero.info.based_stream_info import BasedStreamInfo


class P8FacrInfo(BasedStreamInfo):
    def __init__(self, data: dict = None):
        self.facr_config = "conf/dev/service/facr/insight/face_helper.yaml"
        # self.val_dict = {"key1": "value1", "key2": "value3"}
        super().__init__(data)
