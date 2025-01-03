from zero.info.based_stream_info import BasedStreamInfo


class DoubleMatchInfo(BasedStreamInfo):
    def __init__(self, data: dict = None):
        self.detection_labels = []  # 主体检测类别
        self.dm_valid_count = 5  # 对象有效帧阈值
        self.dm_lost_frame = 60  # 消失多少帧丢弃
        self.dm_y_sort = False  # 分配id前进行y轴排序（时间换精度）
        self.dm_zone = []  # 次体合法检测区域 ltrb百分比 <-> [0][1][2][3]
        self.dm_match_method = 0  # 0:里外包围盒匹配  1:基于L2匹配
        self.dm_warn_type = 2  # 报警类型 (1:phone 2:helmet 3:card 4:intrude)
        super().__init__(data)  # 前面是声明，一定要最后调用这段赋值
        # 特殊处理
        # 0:像素容忍值(通常为10) 1:像素**2容忍值(通常为10000)
        self.dm_match_tolerance = 10 if self.dm_match_method == 0 else 10000
        # 绘制类别(安全帽只绘制未佩戴的2, 手机则绘制0)
        self.dm_draw_cls = [2] if self.dm_warn_type == 2 else [0]
        # 指定类别映射到0类别（在main_update时需要）
        # 安全帽未标准佩戴和正常佩戴为0，其他类别不映射
        self.dm_aggregate_cls = [0, 1] if self.dm_warn_type == 2 else [0]
        # 指定报警类别（在最后计算结果时需要）
        # 安全帽未佩戴为2，其他类别默认为0
        self.dm_anomaly_cls = [2] if self.dm_warn_type == 2 else [0]
