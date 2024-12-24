
class HelmetItem:
    def __init__(self):
        self.obj_id = 0  # 目标id
        self.last_update_id = 0  # 上次更新帧
        self.valid_count = 0  # 有效更新阈值（连续检测多少帧报警，才报警）
        self.cls: int = 1  # 0: wrong 1:right 2:no_hat
        self.has_warn = False  # 是否已经报警

    def init(self, obj_id, cls, last_update_id):
        self.obj_id = obj_id
        self.last_update_id = last_update_id
        self.valid_count = 0
        # 错误佩戴算正确佩戴
        if cls == 0:
            cls = 1
        self.cls = cls
        self.has_warn = False

    # def update(self, last_update_id):
    #     self.last_update_id = last_update_id

    def cls_update(self, cls):
        # 错误佩戴算正确佩戴
        if cls == 0:
            cls = 1
        if not self.has_warn:
            # print(f"now_cls: {self.cls}, predict: {cls} valid_count: {self.valid_count}")
            if cls == self.cls:  # 当前检测类别和记录的类别相同，递增
                self.valid_count += 1
            else:  # 不同则重置
                self.valid_count = 0
                self.cls = cls

    def get_valid_count(self):
        return self.valid_count

