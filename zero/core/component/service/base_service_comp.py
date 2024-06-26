from zero.core.component.base.component import Component
from loguru import logger

from zero.core.key.shared_key import SharedKey


class BaseServiceComponent(Component):
    def __init__(self, shared_data):
        super().__init__(shared_data)

    def on_start(self):
        super().on_start()
        logger.info(f"{self.pname} 成功初始化！")
        # 只要是服务组件，在初始化结束都必须通知给主进程
        self.shared_data[SharedKey.WAIT_COUNTER] += 1
