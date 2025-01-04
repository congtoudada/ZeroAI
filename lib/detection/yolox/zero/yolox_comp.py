import time
import traceback
from loguru import logger

from detection_core.detection_comp import DetectionComponent
from yolox.exp import get_exp
from yolox.zero.predictor import create_zero_predictor
from yolox.zero.yolox_info import YoloxInfo
from zero.key.global_key import GlobalKey
from utility.config_kit import ConfigKit


class YoloxComponent(DetectionComponent):
    def __init__(self, shared_memory, config_path: str):
        super().__init__(shared_memory, config_path)
        self.config: YoloxInfo = YoloxInfo(ConfigKit.load(config_path))
        self.pname = f"[ {self.pid}:yolox for {self.config.yolox_args_expn}]"

    def on_start(self):
        """
        初始化时调用一次
        :return:
        """
        super().on_start()
        # 初始化yolox
        exp = get_exp(self.config.yolox_args_exp_file, self.config.yolox_args_name)
        # 创建zero框架版的yolox目标检测器
        self.predictor = create_zero_predictor(self.config, exp, self.pname)
        self.conf = self.config.yolox_args_conf


def create_process(shared_memory, config_path: str):
    comp = YoloxComponent(shared_memory, config_path)  # 创建组件
    try:
        comp.start()  # 初始化
        # 初始化结束通知
        shared_memory[GlobalKey.LAUNCH_COUNTER.name] += 1
        while not shared_memory[GlobalKey.ALL_READY.name]:
            time.sleep(0.1)
        comp.update()  # 算法逻辑循环
    except KeyboardInterrupt:
        comp.on_destroy()
    except Exception as e:
        logger.error(f"YoloxComponent: {e}")
        logger.error(traceback.format_exc())  # 打印完整的堆栈信息
        comp.on_destroy()
