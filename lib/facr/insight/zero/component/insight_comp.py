import multiprocessing
import os
import sys
import traceback

import cv2
from UltraDict import UltraDict
from loguru import logger

from insight.zero.component.face_recognizer import FaceRecognizer
from insight.zero.info.insight_info import InsightInfo
from insight.zero.key.face_key import FaceKey
from utility.img_kit import ImgKit
from zero.core.component import Component
from zero.core.global_constant import GlobalConstant
from zero.helper.analysis_helper import AnalysisHelper
from zero.key.global_key import GlobalKey
from utility.config_kit import ConfigKit
from utility.timer_kit import TimerKit


class InsightComponent(Component):
    """
    Insight人脸识别服务:
        1.所有请求会发送到一个Req Queue，由Insight服务轮询处理。举例: Ultradict['FACE_REQ'].put({请求数据(含pid)})
        2.每个请求方需主动开辟一块共享内存作为Rsp Queue，Insight会把处理后的结果根据请求pid放到相应位置。举例: Ultradict['FACE_RSP'+pid].put({响应数据})
    """
    SHARED_MEMORY_NAME = "insight_face"

    def __init__(self, shared_memory, config_path: str):
        super().__init__(shared_memory)
        self.config: InsightInfo = InsightInfo(ConfigKit.load(config_path))  # 配置文件内容
        self.pname = f"[ {os.getpid()}:insight_face ]"
        self.face_shared_memory = UltraDict(name=InsightComponent.SHARED_MEMORY_NAME, shared_lock=GlobalConstant.LOCK_MODE)
        self.req_queue = None  # 人脸请求队列
        self.database_file = os.path.join(os.path.dirname(self.config.insight_database),
                                          "database-{}.json".format(self.config.insight_rec_feature))  # 人脸特征库配置文件路径
        if os.path.exists(self.database_file):
            os.remove(self.database_file)
        self.face_model: FaceRecognizer = FaceRecognizer(self.config)  # 人脸识别模型（含人脸检测、人脸关键点检测、特征对齐、人脸识别）
        self.time_flag = 0  # 时间标识，用于检查是否重建特征库
        self.check_database_time = max(self.config.update_fps * 30, 1800)  # 半分钟检查一次
        self.database_timer = TimerKit()  # 特征数据库构建计时器
        self.infer_timer = TimerKit()  # 推理计时器

    def on_start(self):
        # 初始化请求缓存
        self.req_queue = multiprocessing.Manager().Queue()
        self.face_shared_memory[FaceKey.FACE_REQ.name] = self.req_queue
        if self.config.insight_debug_enable:
            if not os.path.exists(self.config.insight_debug_output):
                os.makedirs(self.config.insight_debug_output, exist_ok=True)

    def on_update(self):
        # 检查特征库是否需要重建
        self.time_flag = (self.time_flag + 1) % sys.maxsize
        if self.time_flag % self.check_database_time == 0:
            if not os.path.exists(self.database_file):
                self.database_timer.tic()
                self.face_model.create_database(self.config.insight_database)
                self.database_timer.toc()
                # 记录构建特征库平均耗时
                if self.config.log_analysis:
                    AnalysisHelper.refresh("Face Database Reconstruction", self.database_timer.average_time * 1000, 9999)

        # 处理请求
        while not self.req_queue.empty():
            self.infer_timer.tic()
            req_package = self.req_queue.get()
            cam_id = req_package[FaceKey.FACE_REQ_CAM_ID.name]  # 请求的摄像头id
            pid = req_package[FaceKey.FACE_REQ_PID.name]  # 请求的进程
            obj_id = req_package[FaceKey.FACE_REQ_OBJ_ID.name]  # 请求的对象id
            face_image = req_package[FaceKey.FACE_REQ_IMAGE.name]  # 请求的图片
            # 人脸识别处理
            per_id, score = self.face_model.search_face_image(face_image, self.config.insight_vis)
            # if per_id != 1:
            #     logger.info(f"{self.pname} 识别成功! cam_id: {cam_id}, obj_id: {obj_id}, per_id: {per_id}, score: {score}")
            # debug输出
            if self.config.insight_debug_enable and face_image is not None and face_image.size != 0:
                img_path = os.path.join(self.config.insight_debug_output,
                                        f"facr_cam{cam_id}_per{per_id}_score{score:.2f}.jpg")
                cv2.imwrite(img_path, face_image)
            # 响应输出结果
            rsp_key = FaceKey.FACE_RSP.name + str(pid)
            if self.face_shared_memory.__contains__(rsp_key):
                self.face_shared_memory[rsp_key].put({
                    FaceKey.FACE_RSP_OBJ_ID.name: obj_id,
                    FaceKey.FACE_RSP_PER_ID.name: per_id,
                    FaceKey.FACE_RSP_SCORE.name: score
                })
            self.infer_timer.toc()
            # break  # 每次最多处理一个响应
        # 记录推理平均耗时
        if self.config.log_analysis:
            AnalysisHelper.refresh("Face inference average time", self.infer_timer.average_time * 1000)

    def on_destroy(self):
        self.face_shared_memory.unlink()
        self.face_model.save()  # 保存数据库
        super().on_destroy()


def create_process(shared_memory, config_path: str):
    comp = InsightComponent(shared_memory, config_path)
    try:
        comp.start()
        shared_memory[GlobalKey.LAUNCH_COUNTER.name] += 1
        comp.update()
    except KeyboardInterrupt:
        comp.on_destroy()
    except Exception as e:
        logger.error(f"InsightComponent: {e}")
        logger.error(traceback.format_exc())  # 打印完整的堆栈信息
        comp.on_destroy()


if __name__ == '__main__':
    img = cv2.imread('res/images/face/database/48-0001.jpg')
    config: InsightInfo = InsightInfo(ConfigKit.load("conf/dev/modules/face/insight/insight.yaml"))
    face_model: FaceRecognizer = FaceRecognizer(config)
    timerKit = TimerKit()
    print("开始预测")
    for i in range(10):
        timerKit.tic()
        per_id, score = face_model.search_face_image(img, config.insight_vis)
        print(f"{per_id} {score}")
        timerKit.toc()
    logger.info(f"耗时: {timerKit.average_time:.6f}")
