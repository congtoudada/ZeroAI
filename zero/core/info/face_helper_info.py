from zero.core.info.base_info import BaseInfo


class FaceHelperInfo(BaseInfo):
    def __init__(self, data: dict = None):
        self.face_ports = []  # 请求端口
        self.face_cull_y = 0.2  # 上下剔除百分比，只有当对象进入中间区域才识别
        self.face_min_send_interval = 60  # 最快每多少帧发送一次人脸请求（小于0为不限）
        # 暂未实装
        self.face_success_thresh = 2  # 人脸识别成功次数阈值 (只有当识别结果次数>=该阈值，才认定人脸检测成功，否则返回陌生人1）
        self.face_optimal_matching = False  # 是否开启最优匹配（如果识别成功次数小于count_thresh，则选择最有可能的一项结果作为识别结果）
        super().__init__(data)   # 前面是声明，一定要最后调用这段赋值
