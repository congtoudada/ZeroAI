import os
from typing import Dict
from UltraDict import UltraDict
from loguru import logger

from insight.zero.component.face_process_helper import FaceProcessHelper
from insight.zero.component.insight_comp import InsightComponent
from insight.zero.info.face_helper_info import FaceHelperInfo
from utility.config_kit import ConfigKit


class FaceHelper:
    """
    人脸识别帮助类，由用户持有
    """
    def __init__(self, config, cam_id, callback):
        if isinstance(config, str):
            self.config: FaceHelperInfo = FaceHelperInfo(ConfigKit.load(config))
        else:
            self.config: FaceHelperInfo = config
        self.pname = f"[ {os.getpid()}:face_helper ]"
        self.face_shared_memory = UltraDict(name=InsightComponent.SHARED_MEMORY_NAME)
        self.handler = FaceProcessHelper(self.face_shared_memory, self.config, cam_id, self.face_callback)
        # key: obj_id
        # value: { "per_id": 1, "score": 0 }
        self.face_dict: Dict[int, dict] = {}  # 人脸识别结果集
        self.callback = callback

    def tick(self):
        """
        tick: 用于轮询响应队列
        :return:
        """
        self.handler.tick()

    def try_send(self, obj_id, image, diff, per_y, retry) -> bool:
        """
        是否可以发送
        :param obj_id:
        :param image:
        :param diff:
        :param per_y:
        :param retry:
        :return:
        """
        if retry > self.config.face_max_retry:  # 大于重试次数，不发送
            return False
        if self.face_dict.__contains__(obj_id) and self.face_dict[obj_id]['per_id'] != 1:  # 不是陌生人，不发送
            return False
        if not diff > self.config.face_min_send_interval:  # 不满足发送间隔，不发送
            return False
        if not self.config.face_cull_up_y < per_y < 1.0 - self.config.face_cull_down_y:  # 在剔除区域，不发送
            return False
        # 尝试发送人脸识别请求（内部可能还会判断）
        return self.handler.send(obj_id, image)

    def destroy_obj(self, obj_id):
        """
        清除对象
        :param obj_id:
        :return:
        """
        if self.face_dict.__contains__(obj_id):
            self.face_dict.pop(obj_id)

    def face_callback(self, obj_id, per_id, score):
        logger.info(f"{self.pname} 收到人脸响应: {obj_id} {per_id} {score}")
        # 添加到结果集缓存
        self.face_dict[obj_id] = {
            "per_id": per_id,
            "score": score
        }
        # 触发外界回调函数
        if self.callback is not None:
            self.callback(obj_id, per_id, score)
