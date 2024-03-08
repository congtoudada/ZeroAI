import multiprocessing
import os
import sys
import time

import cv2
from loguru import logger
import numpy as np

from insight.zero.component.face_recognizer import FaceRecognizer
from insight.zero.info.insight_info import InsightInfo
from zero.core.component.service.base_service_comp import BaseServiceComponent
from zero.core.key.face_key import FaceKey
from zero.core.key.shared_key import SharedKey
from zero.utility.config_kit import ConfigKit
from zero.utility.timer_kit import TimerKit


class InsightComponent(BaseServiceComponent):
    def __init__(self, shared_data, config_path: str):
        super().__init__(shared_data)
        self.config: InsightInfo = InsightInfo(ConfigKit.load(config_path))
        self.pname = f"[ {os.getpid()}:insight_face ]"
        self.req_queue = None
        self.req_key = FaceKey.REQ.name + self.config.insight_port
        self.database_file = os.path.join(os.path.dirname(self.config.insight_database),
                                          "database-{}.json".format(self.config.insight_rec_feature))
        self.face_model: FaceRecognizer = FaceRecognizer(self.config)
        self.update_count = 0
        self.timer = TimerKit()

    def on_start(self):
        super().on_start()
        # 初始化请求缓存
        self.req_queue = multiprocessing.Manager().Queue()
        self.shared_data[self.req_key] = self.req_queue

    def on_update(self) -> bool:
        if super().on_update():
            if self.config.insight_update_fps > 0:
                time.sleep(1.0 / self.config.insight_update_fps)

            self.update_count = (self.update_count + 1) % sys.maxsize

            # 检查特征库是否需要重建
            if self.update_count % 2024 == 0:
                if not os.path.exists(self.database_file):
                    self.face_model.create_database(self.config.insight_database)
            # 处理请求
            while not self.req_queue.empty():
                # 处理请求
                req = self.req_queue.get()
                cam_id = req[FaceKey.REQ_CAM_ID]  # 请求的摄像头id
                pid = req[FaceKey.REQ_PID]  # 请求的进程
                obj_id = req[FaceKey.REQ_OBJ_ID]  # 请求的对象id
                face_image = np.ascontiguousarray(np.copy(req[FaceKey.REQ_IMAGE]))  # 请求的图片
                # 人脸识别处理
                self.timer.tic()
                per_id, score = self.face_model.search_face_image(face_image, self.config.insight_vis)
                self.timer.toc()

                # per_id, score = 1, 0
                # 响应输出结果
                rsp_key = FaceKey.RSP.name + self.config.insight_port + str(pid)
                if self.shared_data.__contains__(rsp_key):
                    self.shared_data[rsp_key].put({
                        FaceKey.RSP_OBJ_ID: obj_id,
                        FaceKey.RSP_PER_ID: per_id,
                        FaceKey.RSP_SCORE: score
                    })
                break  # 每次最多处理一个响应
        return False

    def on_analysis(self):
        logger.info(f"{self.pname} face reg fps: {1. / max(1e-5, self.timer.average_time):.2f}")

    def on_destroy(self):
        self.face_model.save()  # 保存数据库
        super().on_destroy()


def create_process(shared_data, config_path: str):
    insightComp: InsightComponent = InsightComponent(shared_data, config_path)  # 创建组件
    insightComp.start()  # 初始化
    insightComp.update()  # 算法逻辑循环


if __name__ == '__main__':
    img = cv2.imread('res/images/face/database/48-0001.jpg')
    config: InsightInfo = InsightInfo(ConfigKit.load("conf/algorithm/face/insight/insight.yaml"))
    face_model: FaceRecognizer = FaceRecognizer(config)
    begin = time.time()
    print("开始预测")
    per_id, score = face_model.search_face_image(img, config.insight_vis)
    print(f"{per_id} {score}")
    logger.info(f"耗时: {time.time() - begin}")
