
class HelmetItem:
    def __init__(self):
        self.obj_id = 0  # 目标id
        self.last_update_id = 0  # 上次更新帧
        self.valid_count = 5  # 有效更新阈值（到达该阈值的Item才有效，避免抖动开销）
