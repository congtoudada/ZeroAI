import importlib
import os
import sys
import time
from multiprocessing import Process

from loguru import logger

from zero.core.component.helper.base_helper_comp import BaseHelperComponent
from zero.core.info.feature.stream_info import StreamInfo
from zero.core.key.shared_key import SharedKey


class AlgorithmGroupComponent(BaseHelperComponent):
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
            logger.info(f"{self.pname} 启动算法: {os.path.basename(comp['path']).split('.')[0]}")
            module_file = comp['path']
            if os.path.exists(module_file):
                sys.path.append(os.path.dirname(module_file))
                module = importlib.import_module(os.path.basename(module_file).split(".")[0])
                if module.__dict__.__contains__("create_process"):
                    Process(target=module.create_process,
                            args=(self.shared_data, comp['conf']),
                            daemon=False).start()
                else:
                    logger.error(f"{self.pname} 算法启动失败！没有实现create_process函数: {os.path.basename(module_file)}")
            else:
                logger.error(f"{self.pname} 算法启动失败！找不到py脚本: {module_file}")


if __name__ == '__main__':
    pass
    # exp_file = "script/helloworld.py"
    # sys.path.append(os.path.dirname(exp_file))
    # current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])
    # print(current_exp.__dict__.__contains__("print_hello"))
    # print(current_exp.__dict__.__contains__("print_hello1"))
    # func = current_exp.__getattribute__("print_hello")
    # # exp = current_exp.print_hello('congtou')
    # func('congtoudada')


