import os
import time
from idlelib.debugger_r import DictProxy
from multiprocessing import Process

from loguru import logger

from zero.core.component.base.base_comp import Component
from zero.core.info.stream_info import StreamInfo
from zero.core.key.shared_key import SharedKey


class AlgorithmGroupComponent(Component):
    """
    算法管理组件
    """
    def __init__(self, shared_data, stream_config: StreamInfo):
        super().__init__(shared_data)
        self.config: StreamInfo = stream_config
        self.pname = f"[ {os.getpid()}:camera{self.config.stream_cam_id} ]"

    def on_start(self):
        super().on_start()
        lock = -1
        for comp in self.config.stream_algorithm:
            # 存在依赖关系，必须顺序初始化
            while lock == self.shared_data[SharedKey.STREAM_WAIT_COUNTER]:
                time.sleep(1)
            lock = self.shared_data[SharedKey.STREAM_WAIT_COUNTER]
            logger.info(f"{self.pname} 启动算法: {comp['name']}")
            # TODO: 可以用工厂模式 + 反射优化该部分代码
            if comp['name'] == "yolox":
                from yolox.zero.component.yolox_comp import create_yolox_process
                Process(target=create_yolox_process,
                        args=(self.shared_data, comp['conf']),
                        daemon=True).start()
            elif comp['name'] == "bytetrack":
                from bytetrack.zero.component.bytetrack_comp import create_bytetrack_process
                Process(target=create_bytetrack_process,
                        args=(self.shared_data, comp['conf']),
                        daemon=True).start()
            elif comp['name'] == "count":
                from count.component.count_comp import create_count_process
                Process(target=create_count_process,
                        args=(self.shared_data, comp['conf']),
                        daemon=True).start()



