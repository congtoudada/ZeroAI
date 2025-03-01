from src.count.count_item import CountItem


class RenlianItem(CountItem):
    def __init__(self):
        super().__init__()
        self.per_id = 1  # 默认为陌生人
        self.score = 0  # 置信度分数
        self.has_save_reid = False  # 是否已经reid存图

    def init(self, obj_id, valid_count):
        super().init(obj_id, valid_count)
        self.per_id = 1  # 默认为陌生人
        self.score = 0  # 置信度分数
        self.has_save_reid = False  # 是否已经reid存图
