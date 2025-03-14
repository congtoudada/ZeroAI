from src.count.count_item import CountItem


class RenlianItem(CountItem):
    def __init__(self):
        super().__init__()
        self.per_id = 1  # 默认为陌生人
        self.score = 0  # 人脸置信度分数
        self.has_save_reid = False  # 是否已经reid存图
        self.best_person_image = None  # 最优人全身图片
        self.best_person_score = 0  # 最优人的分数

    def init(self, obj_id, valid_count):
        super().init(obj_id, valid_count)
        self.per_id = 1  # 默认为陌生人
        self.score = 0  # 置信度分数
        self.has_save_reid = False  # 是否已经reid存图
        self.best_person_image = None  # 最优人全身图片
        self.best_person_score = 0  # 最优分数
