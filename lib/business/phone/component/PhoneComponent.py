import os
from typing import List
import cv2
import numpy as np
from loguru import logger

from bytetrack.zero.tracker.byte_tracker import BYTETracker, STrack
from phone.info.PhoneInfo import PhoneInfo
from zero.core.component.based.based_multi_det_comp import BasedMultiDetComponent
from zero.core.component.based.based_multi_mot_comp import BasedMultiMOTComponent
from zero.core.key.shared_key import SharedKey
from zero.utility.config_kit import ConfigKit
from zero.utility.timer_kit import TimerKit


class PhoneComponent(BasedMultiMOTComponent):
    def __init__(self, shared_data, config_path: str):
        super().__init__(shared_data)
        self.config: PhoneInfo = PhoneInfo(ConfigKit.load(config_path))
        self.pname = f"[ {os.getpid()}:phone ]"

    def on_update(self) -> bool:
        """
        # mot output shape: [n, 7]
        # n: n个对象
        # [0,1,2,3]: tlbr bboxes (基于视频流分辨率)
        #   [0]: x1
        #   [1]: y1
        #   [2]: x2
        #   [3]: y2
        # [4]: 置信度
        # [5]: 类别 (下标从0开始)
        # [6]: id
        """
        if super().on_update() and self.input_mot is not None:
            for i in range(len(self.input_mot)):  # 遍历内一个检测模型的输出
                if self.input_mot[i] is not None:
                    for j in range(len(self.input_mot[i])):  # 遍历每一个对象
                        logger.info(f"来自{self.config.input_port[i]}端口，检测类别: {self.input_mot[i][j][5]} obj_id: {self.input_mot[i][j][6]}")
        return False

    def on_analysis(self):
        logger.info(f"{self.pname} video fps: {1. / max(1e-5, self.update_timer.average_time):.2f}"
                    f" inference fps: {1. / max(1e-5, self.timer.average_time):.2f}")


def create_process(shared_data, config_path: str):
    phoneComp: PhoneComponent = PhoneComponent(shared_data, config_path)  # 创建组件
    phoneComp.start()  # 初始化
    phoneComp.update()  # 算法逻辑循环
