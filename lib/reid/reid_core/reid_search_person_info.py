from zero.info.base_info import BaseInfo


class ReidSearchPersonInfo(BaseInfo):
    def __init__(self, data: dict):
        self.reid_sp_helper_config = "conf/dev/service/reid/helper/search_person_helper.yaml"  # helper配置路径
        super().__init__(data)
