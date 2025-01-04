from zero.info.based_stream_info import BasedStreamInfo


class DetectionInfo(BasedStreamInfo):
    def __init__(self, data: dict = None):
        self.detection_reid_enable = False  # 是否支持reid存图(若支持，会定期对第0个类别发送reid存图请求)
        self.detection_reid_interval = 24  # reid存图间隔(支持reid存图才有效)
        self.detection_reid_camera_gallery = "output/service/clip_reid/camera_gallery"  # 存图路径(支持reid存图才有效)
        self.detection_labels = []  # 类别标签（仅用于可视化）
        super().__init__(data)  # 前面是声明，一定要最后调用这段赋值
