import os
from typing import Dict
from loguru import logger

from zero.core.component.helper.base_helper_comp import BaseHelperComponent
from zero.core.component.helper.face_process_helper_comp import FaceProcessHelperComponent
from zero.core.info.feature.face_helper_info import FaceHelperInfo
from zero.utility.config_kit import ConfigKit


class FaceHelperComponent(BaseHelperComponent):
    def __init__(self, shared_data, config, cam_id, callback):
        super().__init__(shared_data)
        if isinstance(config, str):
            self.config: FaceHelperInfo = FaceHelperInfo(ConfigKit.load(config))
        else:
            self.config: FaceHelperInfo = config
        self.pname = f"[ {os.getpid()}:face_helper ]"
        self.handler = FaceProcessHelperComponent(shared_data, self.config, cam_id, self.face_callback)
        # key: obj_id
        # value: { "per_id": 1, "score": 0 }
        self.face_dict: Dict[int, dict] = {}  # 人脸识别结果集
        self.callback = callback

    def on_start(self):
        super().on_start()
        self.handler.start()

    def on_update(self) -> bool:
        if super().on_update():
            self.handler.update()  # Helper是特殊update
        return False

    def can_send(self, obj_id, diff, per_y) -> bool:
        if self.face_dict.__contains__(obj_id) and self.face_dict[obj_id]['per_id'] != 1:  # 不是陌生人，不发送
            return False
        if not self.handler.can_send(obj_id):  # 处于响应阶段，不发送
            return False
        if not diff > self.config.face_min_send_interval:  # 不满足发送间隔，不发送
            return False
        if not self.config.face_cull_up_y < per_y < 1.0 - self.config.face_cull_down_y:  # 在剔除区域，不发送
            return False
        return True

    def send(self, obj_id, image):
        self.handler.send(obj_id, image)

    def destroy_obj(self, obj_id):
        if self.face_dict.__contains__(obj_id):
            self.face_dict.pop(obj_id)

    def get_face_dict(self) -> dict:
        return self.face_dict

    def face_callback(self, obj_id, per_id, score):
        logger.info(f"{self.pname} 人脸响应: {obj_id} {per_id} {score}")
        self.face_dict[obj_id] = {
            "per_id": per_id,
            "score": score
        }
        if self.callback is not None:
            self.callback(obj_id, per_id, score)

    def on_destroy(self):
        self.handler.destroy()
        super().on_destroy()
