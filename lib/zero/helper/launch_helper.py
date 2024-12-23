import importlib
import os
import sys
import time
from multiprocessing import Process
from loguru import logger

from zero.key.global_key import GlobalKey


class LaunchHelper:
    """
    启动器类，利用Python反射启动
    调用.py脚本的create_process方法
    launch_list格式: [{path: xxx, conf: xxx}, ...]
    """

    def __init__(self, shared_memory):
        self.shared_memory = shared_memory
        self.pname = f"[ {os.getpid()}:service helper ]"

    def execute(self, launch_list):
        lock = -1
        for comp in launch_list:
            # 存在依赖关系，必须顺序初始化
            while lock == self.shared_memory[GlobalKey.LAUNCH_COUNTER.name]:
                time.sleep(1)
            lock = self.shared_memory[GlobalKey.LAUNCH_COUNTER.name]
            logger.info(f"{self.pname} 启动python脚本: {os.path.basename(comp['path']).split('.')[0]}")
            module_file = comp['path']
            if os.path.exists(module_file):
                sys.path.append(os.path.dirname(module_file))
                module = importlib.import_module(os.path.basename(module_file).split(".")[0])
                if module.__dict__.__contains__("create_process"):
                    logger.info(f"{self.pname} 配置文件路径: {comp['conf']}")
                    Process(target=module.create_process,
                            args=(self.shared_memory, comp['conf']),
                            daemon=False).start()
                else:
                    logger.error(
                        f"{self.pname} python脚本启动失败！没有实现create_process: {os.path.basename(module_file)}")
            else:
                logger.error(f"{self.pname} 服务启动失败！找不到python脚本: {module_file}")
