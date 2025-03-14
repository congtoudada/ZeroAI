from zero.info.based_stream_info import BasedStreamInfo


class IntrudeInfo(BasedStreamInfo):
    def __init__(self, data: dict = None):
        self.intrude_valid_count = 5  # 对象有效报警帧阈值
        self.intrude_lost_frame = 60  # 消失多少帧丢弃
        self.intrude_zone = []  # 区域点
        self.intrude_reid_enable = False  # 是否开启reid（优先级大于人脸，二者只会开启一个）
        self.intrude_reid_config = ""  # reid helper config
        self.intrude_face_enable = True  # 是否开启人脸识别
        self.intrude_face_config = ""  # 人脸识别配置文件，发送人脸识别请求相关
        super().__init__(data)  # 前面是声明，一定要最后调用这段赋值
