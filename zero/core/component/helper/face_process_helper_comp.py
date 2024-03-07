import multiprocessing
import os
import random
import time
from loguru import logger

from zero.core.component.base.base_comp import Component
from zero.core.info.face_helper_info import FaceHelperInfo
from zero.core.key.face_key import FaceKey
from zero.core.key.shared_key import SharedKey
from zero.utility.config_kit import ConfigKit


class FaceProcessHelperComponent(Component):
    def __init__(self, shared_data, config, cam_id, callback):
        """
        人脸识别帮助类（主要负责请求和响应）
        :param shared_data:
        :param port:
        :param callback: func(obj_id, per_id, score)
        """
        super().__init__(shared_data)
        if isinstance(config, str):
            self.config: FaceHelperInfo = FaceHelperInfo(ConfigKit.load(config))
        else:
            self.config: FaceHelperInfo = config
        self.pname = f"[ {os.getpid()}:face_helper ]"
        self.send_buffer = []  # 临时消息发送队列，用于延迟发送
        self.recv_queues = []  # 接收队列（强引用）
        self.send_lock: set = set()  # 已发送的数据，只有接收到响应后才可以再次发送
        self.ports = self.config.face_ports
        # 多个请求端口，请求不同人脸识别进程（由人脸识别进程维护）
        self.req_keys = [FaceKey.REQ.name + port for port in self.ports]
        # 多个接收端口，收集来自不同人脸识别进程的响应（由自身维护）
        # FaceKey.RSP.name + self.config.insight_port + str(pid)
        self.rsp_keys = [FaceKey.RSP.name + port + str(os.getpid()) for port in self.ports]
        self.cam_id = cam_id
        self.callback = callback
        self.global_shared_data = shared_data[SharedKey.STREAM_GLOBAL]

    def on_start(self):
        super().on_start()
        # 创建接收队列
        for i, key in enumerate(self.rsp_keys):
            queue = multiprocessing.Manager().Queue()
            self.recv_queues.append(queue)
            self.shared_data[key] = queue

    def send_buffer(self, obj_id, image):
        self.send_buffer.append({
            FaceKey.REQ_CAM_ID: self.cam_id,
            FaceKey.REQ_PID: os.getpid(),
            FaceKey.REQ_OBJ_ID: obj_id,
            FaceKey.REQ_IMAGE: image
        })

    def can_send(self, obj_id):
        if not self.send_lock.__contains__(obj_id):
            return True
        else:
            return False

    def send(self, obj_id, image):
        if self.can_send(obj_id):
            logger.info(f"{self.pname} 发送人脸识别请求: {obj_id}")
            self.send_lock.add(obj_id)
            self.global_shared_data[self._get_random_req_key()].put({
                FaceKey.REQ_CAM_ID: self.cam_id,
                FaceKey.REQ_PID: os.getpid(),
                FaceKey.REQ_OBJ_ID: obj_id,
                FaceKey.REQ_IMAGE: image
            })

    def on_update(self) -> bool:
        # 处理发送队列
        if super().on_update():
            for data in self.send_buffer:
                self.send(data[FaceKey.REQ_OBJ_ID], data[FaceKey.REQ_IMAGE])
                # self.global_shared_data[self._get_random_req_key()].put(data)
            for queue in self.recv_queues:
                while not queue.empty():
                    data = queue.get()
                    obj_id = data[FaceKey.RSP_OBJ_ID]
                    per_id = data[FaceKey.RSP_PER_ID]
                    score = data[FaceKey.RSP_SCORE]
                    self.callback(obj_id, per_id, score)
                    self.send_lock.remove(obj_id)  # 该对象可以再次发送请求
        return False

    def update(self):
        self.on_update()

    def _get_random_req_key(self):
        if len(self.req_keys) == 1:
            return self.req_keys[0]
        return self.req_keys[random.randint(0, len(self.req_keys) - 1)]


if __name__ == '__main__':
    myset: set = {1, 2, 3}
    print(myset.__contains__(4))
    myset.add(4)
    print(myset.__contains__(4))
    # from zero.core.component.feature.launcher_comp import LauncherComponent
    # launcher = LauncherComponent("conf/application-dev.yaml")
    # launcher.start()
    # launcher.update()
