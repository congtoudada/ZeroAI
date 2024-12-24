
class SimpleHttpTask:
    def __init__(self):
        self.url = ""
        self.method = 0
        self.content = None
        self._delay_flag = 0

    def init(self, url, method, content):
        self.url = url
        self.method = method
        self.content = content
        self._delay_flag = 0

    def update(self):
        self._delay_flag += 1

    @property
    def delay_flag(self):
        return self._delay_flag
