from src.count.count_item import CountItem


class RenlianItem(CountItem):
    def __init__(self):
        super().__init__()
        self.per_id = 1  # 默认为陌生人
        self.score = 0  # 置信度分数
        self.retry = 0  # 识别重试次数
        self.last_send_req = 0  # 上次发送请求的帧id

    def init(self, obj_id, valid_count):
        super().init(obj_id, valid_count)
        self.per_id = 1  # 默认为陌生人
        self.score = 0  # 置信度分数
        self.retry = 0  # 识别重试次数
        self.last_send_req = 0  # 上次发送请求的帧id
