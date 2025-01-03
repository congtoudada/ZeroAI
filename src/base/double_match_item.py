from src.base.double_match_info import DoubleMatchInfo


class DoubleMatchItem:
    """
    生命周期随次要个体(sub)
    """
    def __init__(self):
        # 通用属性
        self.last_update_id = 0  # 上次更新帧
        self.valid_count = 0  # 有效更新阈值（连续检测多少帧报警，才报警）
        self.has_warn = False  # 是否已经报警
        self.config: DoubleMatchInfo = None
        # 主体属性(通常是手机、安全帽等)
        self.main_cls = 0   # 主体类别
        self.main_ltrb = (0, 0, 1, 1)  # 主体ltrb
        self.main_valid = False  # 主体是否有效（由于主体和次体未必同时出现，因此需要限制）
        self.main_score = 0  # 主体置信度
        # 次要个体属性(通常是人)
        self.sub_obj_id = 0  # 追踪目标id
        self.sub_ltrb = (0, 0, 1, 1)  # 次体ltrb
        self.sub_per_id = 1  # 陌生人

    def init(self, obj_id, last_update_id, sub_ltrb):
        # 通用属性
        self.last_update_id = last_update_id
        self.valid_count = 0
        self.has_warn = False
        # 主体属性
        self.main_cls = 0
        self.main_ltrb = (0, 0, 1, 1)
        self.main_valid = False
        self.main_score = 0
        # 次体属性
        self.sub_obj_id = obj_id
        self.sub_ltrb = sub_ltrb

    def main_update(self, main_cls, main_ltrb, main_score):
        if not self.has_warn:
            self.main_ltrb = main_ltrb
            self.main_score = main_score
            self.main_valid = True
            if self.main_cls == main_cls:  # 当前检测类别和记录的类别相同，递增
                self.valid_count += 1
            else:  # 不同则重置
                self.valid_count = 0
                self.main_cls = main_cls

    def common_update(self, last_update_id, sub_ltrb):
        self.last_update_id = last_update_id
        self.sub_obj_id = sub_ltrb

