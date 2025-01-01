from zero.info.base_info import BaseInfo


class ClipReidInfo(BaseInfo):
    def __init__(self, data: dict):
        self.clip_reid_dimension = 1280  # VIT特征维度
        self.clip_reid_threshold = 0.86  # reid相似度阈值
        # self.data_shape = 3072  # RN50特征维度
        self.clip_reid_config_file = "lib/reid/clip_reid/configs/person/vit_clipreid.yml"  # 模型配置文件
        self.clip_reid_face_gallery_dir = "output/service/clip_reid/face_gallery"  # 由人脸识别捕捉的带id的人像gallery
        self.clip_reid_camera_gallery_dir = "output/service/clip_reid/camera_gallery"  # 由摄像头定期截图的gallery
        self.clip_reid_refresh_mode = 0  # 刷新模式 0:基于时间刷新 1:基于特征数量刷新 2:综合刷新(满足其中一项就刷新)
        self.clip_reid_refresh_interval = 54000  # 经过n帧刷新一次特征库（30fps * 60s * 30min = 54000 frame）
        self.clip_reid_refresh_count = 10000  # 达到n条数据刷新一次特征库
        self.clip_reid_cull_mode = 0  # 0:全部进行reid 1:只开启白名单 2:只开启黑名单 3:开启黑、白名单，黑名单优先级更高
        self.clip_reid_white_list = [1, 2, 3]  # 只有当开启白名单制时有效，在白名单的摄像头id会参与reid计算，其他不会
        self.clip_reid_black_list = [1, 2, 3]  # 只有当开启黑名单制时有效，在黑名单的摄像头id会不参与reid计算，其他都会
        self.clip_reid_debug_enable = True  # 是否debug
        self.clip_reid_debug_output = "output/service/clip_reid"  # debug输出路径
        super().__init__(data)
