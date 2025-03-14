from src.count.count_info import CountInfo


class RenlianInfo(CountInfo):
    def __init__(self, data: dict = None):
        self.count_face_config = ""  # 人脸识别配置文件，发送人脸识别请求相关
        self.reid_enable = True  # 是否开启reid存图
        self.reid_path = "output/service/clip_reid/face_gallery"  # reid存图路径
        self.reid_best_enable = False  # 是否启用最优匹配
        self.reid_best_aspect = 0.8  # 长宽比小于该aspect的图才有资格成为最优
        super().__init__(data)  # 前面是声明，一定要最后调用这段赋值
