import os
import time
from multiprocessing import Process
from loguru import logger

from zero.core.component.helper.base_helper_comp import BaseHelperComponent
from zero.core.info.app_info import AppInfo
from zero.core.key.shared_key import SharedKey


class ServiceGroupComponent(BaseHelperComponent):
    """
    算法管理组件
    """
    def __init__(self, global_shared_data, app_config: AppInfo):
        super().__init__(global_shared_data)
        self.config: AppInfo = app_config
        self.pname = f"[ {os.getpid()}:app service ]"
        self.global_shared_data = self.shared_data

    def on_start(self):
        super().on_start()
        lock = -1
        for comp in self.config.service:
            # 存在依赖关系，必须顺序初始化
            while lock == self.global_shared_data[SharedKey.WAIT_COUNTER]:
                time.sleep(1)
            lock = self.global_shared_data[SharedKey.WAIT_COUNTER]
            logger.info(f"{self.pname} 启动服务: {comp['name']}")
            # TODO: 可以用工厂模式 + 反射优化该部分代码
            if comp['name'] == "insight":
                from insight.zero.component.insight_comp import create_insight_process
                Process(target=create_insight_process,
                        args=(self.global_shared_data, comp['conf']),
                        daemon=False).start()




