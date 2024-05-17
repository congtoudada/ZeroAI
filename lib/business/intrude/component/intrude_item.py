
class IntrudeItem:
    def __init__(self):
        self.obj_id = 0  # 目标id
        self.last_update_id = 0  # 上次更新帧
        self.valid_count = 0  # 有效报警值（到达该阈值的Item才算报警）
        self.has_warn = False

    def init(self, obj_id, last_update_id):
        self.obj_id = obj_id
        self.last_update_id = last_update_id
        self.valid_count = 0
        self.has_warn = False

    def update(self, last_update_id, in_warn):
        self.last_update_id = last_update_id
        if in_warn:  # 当前处于报警区域，递增
            self.valid_count += 1
        else:
            self.valid_count = 0

    def get_valid_count(self):
        return self.valid_count

