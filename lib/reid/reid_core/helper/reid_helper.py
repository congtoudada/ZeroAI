import multiprocessing
import os
import sys
from typing import Dict

from UltraDict import UltraDict
from loguru import logger
import numpy as np
from PIL import Image

from reid_core.reid_comp import ReidComponent
from reid_core.helper.reid_helper_info import ReidHelperInfo
from reid_core.reid_key import ReidKey
from utility.config_kit import ConfigKit
from utility.object_pool import ObjectPool
from zero.core.global_constant import GlobalConstant


class ReidHelper:
    """
    人脸识别帮助类，由用户持有
    """

    def __init__(self, config=None, reid_callback=None, search_person_callback=None):
        if config is not None and isinstance(config, str):
            self.config: ReidHelperInfo = ReidHelperInfo(ConfigKit.load(config))
        else:
            self.config: ReidHelperInfo = config
        self.pid = os.getpid()
        self.pname = f"[ {self.pid}:reid_helper ]"
        self.req_lock: set = set()  # 请求队列
        self.reid_helper_memory = UltraDict(name=ReidComponent.SHARED_MEMORY_NAME, shared_lock=GlobalConstant.LOCK_MODE)
        if self.config is not None:
            # FastReid
            self.rsp_queue = None
            self.rsp_key = ReidKey.REID_RSP.name + str(self.pid)
            self.dict_pool = ObjectPool(20, dict)
            # key: obj_id
            # value: { "per_id": 1, "score": 0, "retry": 1, "last_time": 0, "last_send_req": 0 }
            self.reid_dict: Dict[int, dict] = {}  # Reid结果集
            self.reid_callback = reid_callback

            # Search Person
            self.rsp_sp_queue = None
            self.rsp_sp_key = ReidKey.REID_RSP_SP.name + str(self.pid)
            self.search_per_id = 0
            self.search_person_callback = search_person_callback

            self.start()

    def start(self):
        if 2 in self.config.reid_quest_method:  # 支持FastReid
            self.rsp_queue = multiprocessing.Manager().Queue()  # Fast Reid接收队列
            self.reid_helper_memory[self.rsp_key] = self.rsp_queue
        if 3 in self.config.reid_quest_method:
            self.rsp_sp_queue = multiprocessing.Manager().Queue()  # 找人接收队列
            self.reid_helper_memory[self.rsp_sp_key] = self.rsp_sp_queue

    def tick(self, now=0):
        if self.config is None:
            return
        # 处理响应队列 FastReid
        if self.rsp_queue is not None:
            while not self.rsp_queue.empty():
                data = self.rsp_queue.get()
                obj_id = data[ReidKey.REID_RSP_OBJ_ID.name]
                per_id = data[ReidKey.REID_RSP_PER_ID.name]
                score = data[ReidKey.REID_RSP_SCORE.name]
                self.reid_callback(obj_id, per_id, score)  # 触发回调事件
                self.req_lock.remove(obj_id)  # 解锁对象，使其可以再次发送请求
        if self.rsp_sp_queue is not None:
            while not self.rsp_sp_queue.empty():
                data = self.rsp_sp_queue.get()
                package = data[ReidKey.REID_RSP_SP_PACKAGE.name]
                self.search_person_callback(package)  # 触发回调事件
                self.req_lock.remove(self.search_per_id)  # 解锁对象，使其可以再次发送请求
        # 清除长期未使用对象
        clear_keys = []
        for key, item in self.reid_dict.items():
            if now - item["last_time"] > self.config.reid_lost_frames:
                clear_keys.append(key)
        clear_keys.reverse()
        for key in clear_keys:
            self.reid_dict.pop(key)  # 从字典中移除item

    def send_save_timing(self, cam_id, pid, obj_id, image):
        """
        存图请求
        """
        req_package = ReidHelper.make_package(cam_id, pid, obj_id, image, 1)
        self.reid_helper_memory[ReidKey.REID_REQ.name].put(req_package)

    def try_send_reid(self, now, image, obj_id, cam_id, status=-1) -> bool:
        """
        Reid请求: 在face shot中匹配
        """
        if self.config is None:
            return
        # 新建
        if not self.reid_dict.__contains__(obj_id):
            reid_item: dict = self.dict_pool.pop()
            # face_item.clear()  # 手动赋值
            reid_item["last_time"] = now
            reid_item["last_send_req"] = 0
            reid_item["retry"] = 0
            reid_item["per_id"] = 1
            reid_item["score"] = 0
            self.reid_dict[obj_id] = reid_item
            req_diff = sys.maxsize
        else:  # 保温
            req_diff = now - self.reid_dict[obj_id]["last_send_req"]
            self.reid_dict[obj_id]["last_time"] = now

        # 图像为None
        if image is None:
            return False
        # 如果正在请求队列，不发送
        if self.req_lock.__contains__(obj_id):
            return False
        # 如果已经识别出非陌生人，不发送
        if self.reid_dict[obj_id]["per_id"] != 1:
            return False
        # 不满足发送间隔
        if req_diff < self.config.reid_min_send_interval:
            return False
        # 超出重试次数
        retry = self.reid_dict[obj_id]["retry"]
        if retry > self.config.reid_max_retry:
            return False
        # 发送
        self.reid_dict[obj_id]["last_send_req"] = now
        req_package = ReidHelper.make_package(cam_id, self.pid, obj_id, image, 2, status)
        logger.info(f"{self.pname} 发送Fast Reid请求: obj_id is {obj_id}")
        self.req_lock.add(obj_id)
        if self.reid_helper_memory is not None:
            self.reid_helper_memory[ReidKey.REID_REQ.name].put(req_package)
        return True

    def try_send_search_person(self, per_id):
        """
        找人
        """
        if self.config is None:
            return
        # 如果正在请求队列，不发送
        if self.req_lock.__contains__(per_id):
            return False, None
        img_path = self.find_first_file(per_id)
        if img_path is None:
            return False, None
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)[..., ::-1]  # RGB->BGR
        req_package = ReidHelper.make_package(0, self.pid, per_id, img_np, 3)
        logger.info(f"{self.pname} 发送Search Person请求: per_id is {per_id}")
        if self.reid_helper_memory is not None:
            self.reid_helper_memory[ReidKey.REID_REQ.name].put(req_package)
            self.search_per_id = per_id
            self.req_lock.add(self.search_per_id)
        return True, img_path

    def find_first_file(self, prefix):
        for filename in os.listdir(self.config.reid_face_gallery_dir):
            if filename.startswith(str(prefix)):
                return os.path.join(self.config.reid_face_gallery_dir, filename)
        return None

    @staticmethod
    def make_package(cam_id, pid, obj_id, image, method, status=-1):
        req_package = {
            ReidKey.REID_REQ_CAM_ID.name: cam_id,
            ReidKey.REID_REQ_PID.name: pid,
            ReidKey.REID_REQ_OBJ_ID.name: obj_id,
            ReidKey.REID_REQ_IMAGE.name: image,
            ReidKey.REID_REQ_METHOD.name: method,  # 方式2
            ReidKey.REID_REQ_STATUS.name: status
        }
        return req_package

    def destroy_obj(self, obj_id):
        """
        清除对象
        :param obj_id:
        :return:
        """
        if self.reid_dict.__contains__(obj_id):
            self.dict_pool.push(self.reid_dict[obj_id])
            self.reid_dict.pop(obj_id)

    def reid_callback(self, obj_id, per_id, score):
        # logger.info(f"{self.pname} 收到reid响应: {obj_id} {per_id} {score}")
        # 添加到结果集缓存
        if self.reid_dict.__contains__(obj_id):
            self.reid_dict[obj_id]["per_id"] = per_id
            self.reid_dict[obj_id]["score"] = score
            self.reid_dict[obj_id]["retry"] += 1
            re =  self.reid_dict[obj_id]["retry"]
            # 触发外界回调函数
            if self.reid_callback is not None:
                self.reid_callback(obj_id, per_id, score)
