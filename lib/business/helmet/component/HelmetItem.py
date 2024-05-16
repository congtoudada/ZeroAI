
class HelmetItem:
    def __init__(self):
        self.obj_id = 0  # 目标id
        self.last_update_id = 0  # 上次更新帧
        self.valid_count = 0  # 有效更新阈值（到达该阈值的Item才有效，避免抖动开销）
        self.cls = 0  # 0: wrong 1:right 2:no_hat
        self.has_warn = False

    def init(self, obj_id, cls, last_update_id):
        self.obj_id = obj_id
        self.last_update_id = last_update_id
        self.valid_count = 0
        self.cls = cls
        self.has_warn = False

    def update(self, last_update_id, cls):
        self.last_update_id = last_update_id
        if cls == self.cls:  # 当前检测类别和记录的类别相同，递增
            self.valid_count += 1
        else:
            self.valid_count = 0

    def get_valid_count(self):
        return self.valid_count

