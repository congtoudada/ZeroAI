import os
from typing import Dict
from UltraDict import UltraDict
from loguru import logger

from insight.zero.component.face_process_helper import FaceProcessHelper
from insight.zero.component.insight_comp import InsightComponent
from insight.zero.info.face_helper_info import FaceHelperInfo
from utility.config_kit import ConfigKit
from utility.img_kit import ImgKit
from zero.core.global_constant import GlobalConstant


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
        self.face_shared_memory = UltraDict(name=InsightComponent.SHARED_MEMORY_NAME, shared_lock=GlobalConstant.LOCK_MODE)
        self.handler = FaceProcessHelper(self.face_shared_memory, self.config, cam_id, self.face_callback)
        # key: obj_id
        # value: { "per_id": 1, "score": 0 }
        self.face_dict: Dict[int, dict] = {}  # 人脸识别结果集
        self.callback = callback

    def tick(self, now):
        """
        tick: 用于轮询响应队列
        :return:
        """
        self.handler.tick()
        # 清除长期未使用对象
        clear_keys = []
        for key, item in self.face_dict.items():
            if now - item["last_time"] > self.config.face_lost_frames:
                clear_keys.append(key)
        clear_keys.reverse()
        for key in clear_keys:
            self.face_dict.pop(key)  # 从字典中移除item

    def try_send(self, now, frame, ltrb, obj_id, diff, per_y, retry) -> bool:
        """
        尝试发送识别请求
        :return:
        """
        # 保温
        if self.face_dict.__contains__(obj_id):
            self.face_dict[obj_id].update({
                "last_time": now
            })

        if retry > self.config.face_max_retry:  # 大于重试次数，不发送
            return False
        if self.face_dict.__contains__(obj_id) and self.face_dict[obj_id]['per_id'] != 1:  # 不是陌生人，不发送
            return False
        if not diff > self.config.face_min_send_interval:  # 不满足发送间隔，不发送
            return False
        if not self.config.face_cull_up_y < per_y < 1.0 - self.config.face_cull_down_y:  # 在剔除区域，不发送
            return False
        # 尝试发送人脸识别请求（内部可能还会判断）
        if frame is not None:
            image = ImgKit.crop_img(frame, ltrb)
            if image is not None:
                ret = self.handler.send(obj_id, image)
                if ret:
                    self.face_dict[obj_id] = {
                        "last_time": now,
                        "per_id": 1,
                        "score": 0
                    }
                    return True
        return False

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
        if self.face_dict.__contains__(obj_id):
            # 添加到结果集缓存
            self.face_dict[obj_id].update({
                "per_id": per_id,
                "score": score
            })
            # 触发外界回调函数
            if self.callback is not None:
                self.callback(obj_id, per_id, score)
