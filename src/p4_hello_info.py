from zero.info.base_info import BaseInfo


class P4HelloInfo(BaseInfo):
    def __init__(self, data: dict = None):
        self.val_int = 100
        self.val_float = 3.14
        self.val_str = "haha"
        self.val_bool = True
        self.val_list = [1, 2, 3]
        # self.val_dict = {"key1": "value1", "key2": "value3"}
        super().__init__(data)
