from zero.core.info.base.base_info import BaseInfo


class InsightInfo(BaseInfo):
    def __init__(self, data: dict = None):
        self.insight_port = ""  # 收集请求的端口名
        self.insight_vis = False  # 可视化
        self.insight_force_rebuild = False  # 每次运行无论本地是否有缓存都重新建库
        self.insight_update_fps = -1  # 算法update帧率, <=0 表示无限制
        self.insight_det_thresh = 0.4  # 人脸检测阈值，小于该阈值的检测框会被剔除
        self.insight_det_detector = "RFB"  # ['RFB', 'MTCNN']
        self.insight_det_ckpt = ""  # 人脸检测权重文件
        self.insight_rec_thresh = 0.45  # 人脸识别阈值，小于该阈值的人脸识别结果为unknown，表示未知
        self.insight_rec_feature = "mobilenet_v2"  # ['resnet50', 'resnet18', 'mobilenet_v2']
        self.insight_rec_input_size = 112  # 输入缩放到该尺寸
        self.insight_rec_embedding_size = 512  # 编码特征向量维度
        self.insight_rec_ckpt = ""  # 人脸特征提取权重文件
        self.insight_database = "res/images/face/database"  # 特征库位置
        self.insight_reconstruct_file = "bin/runtime/face.pkl"  # 人脸标识文件路径（该文件用于通知重建特征库）
        super().__init__(data)  # 前面是声明，一定要最后调用这段赋值
