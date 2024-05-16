import os
from typing import Dict

from helmet.component.HelmetItem import HelmetItem
from helmet.info.HelmetInfo import HelmetInfo
from zero.core.component.based.based_mot_comp import BasedMOTComponent
from zero.utility.config_kit import ConfigKit
from loguru import logger


class HelmetComponent(BasedMOTComponent):
    def __init__(self, shared_data, config_path: str):
        super().__init__(shared_data)
        self.config: HelmetInfo = HelmetInfo(ConfigKit.load(config_path))
        self.pname = f"[ {os.getpid()}:helmet for {self.config.input_port[0]}]"
        # key: obj_id value: cls
        self.data_dict: Dict[int, HelmetItem] = {}

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
            self.preprocess()
            for obj in self.input_mot:
                ltrb = obj[:4]
                conf = obj[4]
                cls = int(obj[5])
                obj_id = int(obj[6])
                if not self.data_dict.__contains__(obj_id):  # 没有被记录过
                    item = HelmetItem(obj_id, cls)
                    self.data_dict[obj_id] = item
                    # if cls == 0 or cls == 2:
                    #     # 报警
                    #     logger.info("首次安全帽佩戴异常")
                else:  # 已经记录过
                    self.data_dict[obj_id].update(self.current_frame_id, cls)
                    # if self.data_dict[obj_id] == 1:
                    #     if cls == 0 or cls == 2:
                    #         self.data_dict[obj_id] = cls
                    #         # 报警
                    #         logger.info("后期安全帽佩戴异常")
                self.postprocess_item(self.data_dict[obj_id])
            return True
        return False

    def postprocess_item(self, helmet_item: HelmetItem):
        if not helmet_item.has_warn and helmet_item.get_valid_count() >= self.config.helmet_valid_count:
            if helmet_item.cls == 0 or helmet_item.cls == 2:
                logger.info("安全帽佩戴异常")
                helmet_item.has_warn = True

    def preprocess(self):
        # 清空长期未更新点
        clear_keys = []
        for key, item in self.data_dict.items():
            if self.current_frame_id - item.last_update_id > self.config.helmet_lost_frame:
                clear_keys.append(key)
        for key in clear_keys:
            self.data_dict.pop(key)  # 从字典中移除item


def create_process(shared_data, config_path: str):
    helmetComp: HelmetComponent = HelmetComponent(shared_data, config_path)  # 创建组件
    helmetComp.start()  # 初始化
    helmetComp.update()  # 算法逻辑循环
