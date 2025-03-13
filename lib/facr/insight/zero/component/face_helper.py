import os
import sys
from typing import Dict
from UltraDict import UltraDict
from loguru import logger

from insight.zero.component.face_process_helper import FaceProcessHelper
from insight.zero.component.insight_comp import InsightComponent
from insight.zero.info.face_helper_info import FaceHelperInfo
from utility.config_kit import ConfigKit
from utility.img_kit import ImgKit
from utility.object_pool import ObjectPool
from zero.core.global_constant import GlobalConstant


class FaceHelper:
    """
    人脸识别帮助类，由用户持有
    """
    def __init__(self, config, callback):
        if isinstance(config, str):
            self.config: FaceHelperInfo = FaceHelperInfo(ConfigKit.load(config))
        else:
            self.config: FaceHelperInfo = config
        self.pname = f"[ {os.getpid()}:face_helper ]"
        self.face_shared_memory = UltraDict(name=InsightComponent.SHARED_MEMORY_NAME, shared_lock=GlobalConstant.LOCK_MODE)
        self.handler = FaceProcessHelper(self.face_shared_memory, self.config, self.face_callback)
        self.dict_pool = ObjectPool(20, dict)
        # key: obj_id
        # value: { "last_time": 0, "last_send_req": 0, "retry": 0, "per_id": 1, "score": 0 }
        self.face_dict: Dict[int, dict] = {}
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

    def try_send(self, now, frame, ltrb, obj_id, obj_x=-1, obj_y=-1, cam_id=0) -> bool:
        """
        尝试发送识别请求
        :param now: 当前时间
        :param frame: 当前帧图像
        :param ltrb: 包围框
        :param obj_id: 对象id
        :param obj_x: 对象y轴百分比，用于限定有效识别区域
        :param obj_y: 对象y轴百分比，用于限定有效识别区域
        :param cam_id: 请求摄像头id (仅调试)
        :return:
        """
        # 新建
        if not self.face_dict.__contains__(obj_id):
            face_item: dict = self.dict_pool.pop()
            # face_item.clear()  # 手动赋值
            face_item["last_time"] = now
            face_item["last_send_req"] = 0
            face_item["retry"] = 0
            face_item["per_id"] = 1
            face_item["score"] = 0
            self.face_dict[obj_id] = face_item
            req_diff = sys.maxsize
        else:  # 保温
            req_diff = now - self.face_dict[obj_id]["last_send_req"]
            self.face_dict[obj_id]["last_time"] = now

        retry = self.face_dict[obj_id]["retry"]
        if retry > self.config.face_max_retry:  # 大于重试次数，不发送
            return False
        per_id = self.face_dict[obj_id]["per_id"]
        if per_id != 1:  # 不是陌生人，不发送
            return False
        if req_diff < self.config.face_min_send_interval:  # 小于发送间隔，不发送
            return False
        if obj_y != -1 and not self.config.face_cull_up_y < obj_y < 1.0 - self.config.face_cull_down_y:  # 不在检测区域，不发送
            return False
        if obj_x != -1 and not self.config.face_cull_left_x < obj_x < 1.0 - self.config.face_cull_right_x:  # 不在检测区域，不发送
            return False
        # 尝试发送人脸识别请求（内部可能还会判断）
        if frame is not None:
            image = ImgKit.crop_img(frame, ltrb)
            if image is not None:
                if self.handler.send(obj_id, image, cam_id):
                    self.face_dict[obj_id]["last_send_req"] = now
                    return True
        return False

    def destroy_obj(self, obj_id):
        """
        清除对象
        :param obj_id:
        :return:
        """
        if self.face_dict.__contains__(obj_id):
            self.dict_pool.push(self.face_dict[obj_id])
            self.face_dict.pop(obj_id)

    def face_callback(self, obj_id, per_id, score):
        logger.info(f"{self.pname} Receive Face: {obj_id} {per_id} {score}")
        if self.face_dict.__contains__(obj_id):
            # 添加到结果集缓存
            self.face_dict[obj_id]["per_id"] = per_id
            self.face_dict[obj_id]["score"] = score
            self.face_dict[obj_id]["retry"] += 1
            # 触发外界回调函数
            if self.callback is not None:
                self.callback(obj_id, per_id, score)
