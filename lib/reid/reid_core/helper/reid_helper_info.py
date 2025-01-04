from zero.info.base_info import BaseInfo


class ReidHelperInfo(BaseInfo):
    def __init__(self, data: dict = None):
        # self.reid_ports = []  # 请求端口
        self.reid_face_gallery_dir = "output/service/clip_reid/face_gallery"  # 由人脸识别捕捉的带id的人像gallery
        self.reid_min_send_interval = 30  # 最快每多少帧发送一次人脸请求（小于0为不限）
        self.reid_max_retry = 5  # 单人最大重试次数
        self.reid_lost_frames = 300  # 超过一定时间未销毁的对象自动销毁
        self.reid_quest_method = []  # 支持的请求方式 1:存图请求(可不配置) 2:Reid请求 3:找人请求
        # 暂未实装
        super().__init__(data)   # 前面是声明，一定要最后调用这段赋值
