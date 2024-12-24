
class IntrudeItem:
    def __init__(self):
        self.obj_id = 0  # 目标id
        self.last_update_id = 0  # 上次更新帧
        self.valid_count = 0  # 有效报警值（到达该阈值的Item才算报警）
        self.has_warn = False
        # ------- 2024.09.20新增 -------
        self.enable = False
        self.base_x = 0  # base_x 百分比
        self.base_y = 0  # base_y 百分比
        self.per_id = 1  # 默认为陌生人
        self.score = 0  # 置信度分数
        self.retry = 0  # 识别重试次数
        self.last_send_req = 0  # 上次发送请求的帧id
        self.ltrb = (0, 0, 0, 0)  # 包围盒（实际像素位置）

    def init(self, obj_id, last_update_id):
        self.obj_id = obj_id
        self.last_update_id = last_update_id
        self.valid_count = 0
        self.has_warn = False
        self.enable = False
        self.base_x = 0  # base_x 百分比
        self.base_y = 0  # base_y 百分比
        self.per_id = 1  # 默认为陌生人
        self.score = 0  # 置信度分数
        self.retry = 0  # 识别重试次数
        self.last_send_req = 0  # 上次发送请求的帧id
        self.ltrb = (0, 0, 0, 0)  # 包围盒（实际像素位置）

    def update(self, last_update_id, in_warn, base_x, base_y, ltrb):
        self.last_update_id = last_update_id
        self.ltrb = ltrb  # 包围盒（实际像素位置）
        self.base_x = base_x  # base_x 百分比
        self.base_y = base_y  # base_y 百分比
        if in_warn:  # 当前处于报警区域，递增
            self.valid_count += 1
        else:
            self.valid_count = 0

    def get_valid_count(self):
        return self.valid_count

    def reset_update(self):
        self.ltrb = (0, 0, 0, 0)  # 包围盒（实际像素位置）
        self.enable = False
        self.base_x = 0  # base_x 百分比
        self.base_y = 0  # base_y 百分比
