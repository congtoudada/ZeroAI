import os
from typing import Dict

from helmet.info.HelmetInfo import HelmetInfo
from zero.core.component.based.based_mot_comp import BasedMOTComponent
from zero.utility.config_kit import ConfigKit
from loguru import logger


class HelmetComponent(BasedMOTComponent):
    def __init__(self, shared_data, config_path: str):
        super().__init__(shared_data)
        self.config: HelmetInfo = HelmetInfo(ConfigKit.load(config_path))
        self.pname = f"[ {os.getpid()}:helmet for {self.config.count_input_port}]"
        # key: obj_id value: cls
        self.data_dict: Dict[int, int] = None

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
            for obj in self.input_mot:
                ltrb = obj[:4]
                conf = obj[4]
                cls = int(obj[5])
                obj_id = int(obj[6])
                if not self.data_dict.__contains__(obj_id):  # 没有被记录过
                    self.data_dict[obj_id] = cls
                    if cls == 0 or cls == 2:
                        # 报警
                        logger.info("首次安全帽佩戴异常")
                else:  # 已经记录过
                    if self.data_dict[obj_id] == 1:
                        if cls == 0 or cls == 2:
                            self.data_dict[obj_id] = cls
                            # 报警
                            logger.info("后期安全帽佩戴异常")
            return True
        return False


