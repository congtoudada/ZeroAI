import multiprocessing
import os
from loguru import logger

from insight.zero.info.face_helper_info import FaceHelperInfo
from insight.zero.key.face_key import FaceKey
from utility.config_kit import ConfigKit


class FaceProcessHelper:
    """
    人脸识别进程级帮助类（主要负责进程间请求和响应）
    """
    def __init__(self, face_shared_memory, config, cam_id, callback):
        self.face_shared_memory = face_shared_memory
        if isinstance(config, str):
            self.config: FaceHelperInfo = FaceHelperInfo(ConfigKit.load(config))
        else:
            self.config: FaceHelperInfo = config
        self.pname = f"[ {os.getpid()}:face_process_helper ]"
        self.rsp_queue = multiprocessing.Manager().Queue()  # 接收队列
        self.send_lock: set = set()  # 已发送的数据，只有接收到响应后才可以再次发送
        self.ports = self.config.face_ports
        # 多个接收端口，收集来自不同人脸识别进程的响应（由自身维护）
        self.rsp_key = FaceKey.FACE_RSP.name + str(os.getpid())
        self.cam_id = cam_id
        self.callback = callback
        self.start()

    def start(self):
        # 创建接收队列
        self.face_shared_memory[self.rsp_key] = self.rsp_queue

    def can_send(self, obj_id):
        if not self.send_lock.__contains__(obj_id):
            return True
        else:
            return False

    def send(self, obj_id, image):
        """
        发送人脸识别请求（进程级）
        :param obj_id: 对象id
        :param image: 对象截图
        :return:
        """
        if self.can_send(obj_id):
            logger.info(f"{self.pname} 发送人脸识别请求: {obj_id}")
            self.send_lock.add(obj_id)
            self.face_shared_memory[FaceKey.FACE_REQ.name].put({
                FaceKey.FACE_REQ_CAM_ID.name: self.cam_id,
                FaceKey.FACE_REQ_PID.name: os.getpid(),
                FaceKey.FACE_REQ_OBJ_ID.name: obj_id,
                FaceKey.FACE_REQ_IMAGE.name: image
            })
            return True
        else:
            return False

    def tick(self):
        # 处理响应队列
        while not self.rsp_queue.empty():
            data = self.rsp_queue.get()
            obj_id = data[FaceKey.FACE_RSP_OBJ_ID.name]
            per_id = data[FaceKey.FACE_RSP_PER_ID.name]
            score = data[FaceKey.FACE_RSP_SCORE.name]
            self.callback(obj_id, per_id, score)  # 触发回调事件
            self.send_lock.remove(obj_id)  # 解锁对象，使其可以再次发送请求


if __name__ == '__main__':
    myset: set = {1, 2, 3}
    print(myset.__contains__(4))
    myset.add(4)
    print(myset.__contains__(4))
    # from zero.core.component.feature.launcher_comp import LauncherComponent
    # launcher = LauncherComponent("conf/application-dev.yaml")
    # launcher.start()
    # launcher.update()
