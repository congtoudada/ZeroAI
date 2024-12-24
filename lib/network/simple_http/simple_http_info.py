from zero.info.base_info import BaseInfo


class SimpleHttpInfo(BaseInfo):
    def __init__(self, data: dict = None):
        self.input_port = "SimpleHttp"
        self.http_web_address = "localhost:8080"  # 请求地址前缀
        self.http_delay_enable = False  # 是否开启消息延迟发送（该处主要是为了让后端能够访问到本地写入的图片）
        self.http_delay_frame = 30  # Http通常为30fps
        super().__init__(data)  # 前面是声明，一定要最后调用这段赋值
