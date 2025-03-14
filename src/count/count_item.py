
class CountItem:
    def __init__(self):
        self.obj_id = 0  # 目标id
        self.red_cur = -1  # 当前红值
        self.green_cur = -1  # 当前绿值
        self.red_seq = []  # 红线结果序列
        self.green_seq = []  # 绿线结果序列
        self.last_update_id = 0  # 上次更新帧
        self.ltrb = (0, 0, 0, 0)  # 包围盒（实际像素位置）
        self.base_x = 0  # base_x 百分比
        self.base_y = 0  # base_y 百分比
        self.valid_count = 5  # 有效更新阈值（到达该阈值的Item才有效，避免抖动开销）
        self._update_count = 0  # 累计更新次数
        self.enable = False  # 是否为有效点
        self.person_score = 0  # 当前帧人置信度

    def init(self, obj_id, valid_count):
        self.obj_id = obj_id  # 目标id
        self.red_cur = -1
        self.green_cur = -1
        self.red_seq.clear()
        self.green_seq.clear()
        self.last_update_id = 0  # 上次更新帧
        self.ltrb = (0, 0, 0, 0)  # 包围盒（实际像素位置）
        self.base_x = 0  # base_x 百分比
        self.base_y = 0  # base_y 百分比
        self.valid_count = valid_count
        self._update_count = 0  # 累计更新次数
        self.enable = False
        self.person_score = 0  # 当前帧人置信度

    def update(self, last_update_id, base_x, base_y, ltrb, person_score):
        self._update_count += 1
        if self._update_count >= self.valid_count:
            self.enable = True
        self.last_update_id = last_update_id
        self.ltrb = ltrb  # 包围盒（实际像素位置）
        self.base_x = base_x  # base_x 百分比
        self.base_y = base_y  # base_y 百分比
        self.person_score = person_score

    def update_red(self, red_cur):
        if red_cur == -1:
            return
        self.red_cur = red_cur

    def update_green(self, green_cur):
        if green_cur == -1:
            return
        self.green_cur = green_cur

    def reset_update(self):
        self.enable = False
        self.ltrb = (0, 0, 0, 0)  # 包围盒（实际像素位置）
        self.base_x = 0  # base_x 百分比
        self.base_y = 0  # base_y 百分比
        self.person_score = 0  # 人分数
        # self.red_cur = -1
        # self.green_cur = -1



