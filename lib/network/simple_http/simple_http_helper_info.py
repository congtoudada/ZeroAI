from zero.info.base_info import BaseInfo


class SimpleHttpHelperInfo(BaseInfo):
    def __init__(self, data: dict = None):
        self.output_port = "SimpleHttp"  # http请求端口
        self.debug_enable = False  # 是否启用debug模式（会打印发送的消息）
        super().__init__(data)   # 前面是声明，一定要最后调用这段赋值
