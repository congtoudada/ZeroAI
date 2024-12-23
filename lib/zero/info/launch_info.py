from zero.info.base_info import BaseInfo


class LaunchInfo(BaseInfo):
    def __init__(self, data: dict = None):
        # ---------------------- 成员变量声明（会被配置文件覆盖，不可访问） ----------------------
        self.single_mode = True  # 是否为单核模式（根据设备进程数和算法规模选择）
        self.stream_list = []  # 视频流列表 per element: { cam_id: 摄像头id conf: 视频流配置路径 }
        self.algorithm_list = []  # 算法列表，会根据顺序依次启动算法 per element: { py: python脚本路径 conf: 算法配置文件路径 }
        self.service_list = []  # 服务列表，会根据顺序依次启动服务 per element: { py: python脚本路径 conf: 服务配置文件路径 }
        self.app_running_file = ""  # 框架运行时标识路径，运行时生成，删除该文件可以关闭算法端
        self.app_analysis_enable = True  # 是否打印性能分析报告
        self.app_analysis_interval = 10  # 每隔x秒打印一次性能分析报告
        # ---------------------------------- End ----------------------------------
        super().__init__(data)

