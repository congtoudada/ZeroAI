import multiprocessing
import os
import sys
import time
import traceback
from typing import Dict
import cv2
import numpy as np
from PIL import Image
from UltraDict import UltraDict
from loguru import logger

from clip_reid.zero.component.clip_reid_wrapper import ClipReidWrapper
from clip_reid.zero.component.faiss_reid_helper import FaissReidHelper
from clip_reid.zero.info.clip_reid_info import ClipReidInfo
from clip_reid.zero.key.reid_key import ReidKey
from utility.file_modify_kit import FileModifyKit
from zero.core.component import Component
from zero.helper.analysis_helper import AnalysisHelper
from zero.key.global_key import GlobalKey
from utility.config_kit import ConfigKit
from utility.timer_kit import TimerKit


class ClipReidComponent(Component):
    """
    ClipReid服务:
        1.所有请求会发送到一个Req Queue，由ClipReid服务轮询处理。举例: Ultradict['REID_REQ'].put({请求数据(含pid)})
        2.每个请求方需主动开辟一块共享内存作为Rsp Queue，ClipReid会把处理后的结果根据请求pid放到相应位置。举例: Ultradict['REID_RSP'+pid].put({响应数据})
    """
    SHARED_MEMORY_NAME = "clip_reid"

    def __init__(self, shared_memory, config_path: str):
        super().__init__(shared_memory)
        self.config: ClipReidInfo = ClipReidInfo(ConfigKit.load(config_path))  # 配置文件内容
        self.pname = f"[ {os.getpid()}:clip_reid ]"
        self.reid_shared_memory = UltraDict(name=ClipReidComponent.SHARED_MEMORY_NAME)
        self.req_queue = None  # reid请求队列
        self.reid_model: ClipReidWrapper = ClipReidWrapper(self.config)  # clip_reid模型
        # self.faiss_dict: Dict[int, FaissReidHelper] = {}  # 根据cam id分类的faiss字典(根据不同摄像头找人，暂不实现)
        self.faiss_search_person: FaissReidHelper = None  # 根据cam id分类的faiss字典
        self.time_flag = 0  # 时间标识，用于检查是否刷新特征半区
        self.last_modify_time = {}  # 各文件上次修改时间，用于检查是否需要重建由人脸生成的身份识别特征库
        self.faiss_face_shot: FaissReidHelper = None  # face_shot 特征库
        self.face_shot_dict: Dict[int, int] = {}  # per_id : faiss_idx
        self.infer_timer = TimerKit()  # 推理计时器

    def on_start(self):
        super().on_start()
        # 初始化请求缓存
        self.req_queue = multiprocessing.Queue()
        self.reid_shared_memory[ReidKey.REID_REQ.name] = self.req_queue
        if self.config.clip_reid_debug_enable:
            if not os.path.exists(self.config.clip_reid_debug_output):
                os.makedirs(self.config.clip_reid_debug_output, exist_ok=True)
        self.faiss_search_person = FaissReidHelper(self.config.clip_reid_dimension,
                                                   self.config.clip_reid_refresh_mode,
                                                   self.config.clip_reid_refresh_interval,
                                                   self.config.clip_reid_refresh_count)

    def on_update(self) -> bool:
        # 处理请求
        while not self.req_queue.empty():
            self.infer_timer.tic()
            req_package = self.req_queue.get()
            cam_id = req_package[ReidKey.REID_REQ_CAM_ID.name]  # 请求的摄像头id
            # 摄像头剔除
            if self.config.clip_reid_cull_mode == 1:  # 只开启白名单
                if cam_id not in self.config.clip_reid_white_list:
                    continue
            elif self.config.clip_reid_cull_mode == 2:  # 只开启黑名单
                if cam_id in self.config.clip_reid_black_list:
                    continue
            elif self.config.clip_reid_cull_mode == 3:  # 同时开启黑、白名单
                # 优先判断是否在黑名单
                if cam_id in self.config.clip_reid_black_list:
                    continue
                if cam_id not in self.config.clip_reid_white_list:
                    continue
            pid = req_package[ReidKey.REID_REQ_PID.name]  # 请求的进程
            obj_id = req_package[ReidKey.REID_REQ_OBJ_ID.name]  # 请求的对象id
            reid_img = req_package[ReidKey.REID_REQ_IMAGE.name]  # 请求的图片
            reid_method = req_package[ReidKey.REID_REQ_METHOD.name]  # 请求方式

            # Reid抽特征
            feat = self.reid_model.extract_feature(reid_img)
            # if cam_id not in self.faiss_dict:  # 首次添加
            #     self.faiss_dict[cam_id] = FaissReidHelper(self.config.clip_reid_dimension,
            #                                               self.config.clip_reid_refresh_mode,
            #                                               self.config.clip_reid_refresh_interval,
            #                                               self.config.clip_reid_refresh_count)
            time_str = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
            extra_info = {"cam_id": cam_id, "time": time_str, "feat": feat}
            self.faiss_search_person.add(feat, extra_info)  # 将特征加入特征库

            if reid_method == 0:  # 1.普通存图请求
                pass  # 已经完事了
            elif reid_method == 1:  # 2.reid识别请求，需要跟人脸截到的人像配准
                # 更新本地特征库
                self.finetune_face_shot()
                k = 1  # topK的K值
                extra_info = self.faiss_face_shot.search(feat, k)  # xq为待检索向量，返回的I为每个待检索query最相似TopK的索引list，D为其对应的距离
                per_id = extra_info['per_id']
                score = extra_info['score']
                if score < self.config.clip_reid_threshold:
                    per_id = 1  # 分数低视为陌生人
                # debug输出
                if self.config.clip_reid_debug_enable:
                    if per_id != 1:
                        logger.info(f"{self.pname} 识别成功! cam_id: {cam_id}, obj_id: {obj_id}, per_id: {per_id}, score: {score}")
                    cv2.imwrite(os.path.join(self.config.clip_reid_debug_enable,
                                             f"reid_cam{cam_id}_per{per_id}_score{score:.2f}.jpg"), reid_img)
                # 响应输出结果
                rsp_key = ReidKey.REID_RSP.name + str(pid)
                if self.reid_shared_memory.__contains__(rsp_key):
                    self.reid_shared_memory[rsp_key].put({
                        ReidKey.REID_RSP_OBJ_ID.name: obj_id,
                        ReidKey.REID_RSP_PER_ID.name: per_id,
                        ReidKey.REID_RSP_SCORE.name: score
                    })
            elif reid_method == 3:  # 找人
                # TODO:找人逻辑！！！！！！！！！！！！！！！！
                pass
            else:
                logger.error(f"{self.pname} Not found reid method: {reid_method}")
            self.infer_timer.toc()
            # break  # 每次最多处理一个响应
        # 记录推理平均耗时
        if self.config.log_analysis:
            AnalysisHelper.refresh("Face inference average time", self.infer_timer.average_time * 1000)
        # tick faiss
        self.time_flag = (self.time_flag + 1) % sys.maxsize
        self.faiss_search_person.tick(self.time_flag)
        return False

    def finetune_face_shot(self):
        # 检查是否存在文件修改
        added, removed, modified, new_mtime = FileModifyKit.check_changes(self.config.clip_reid_face_path,
                                                                          self.last_modify_time)
        added.update(modified)
        for file in added:
            per_id = file.split('_')[0]  # 首位存per id
            if self.face_shot_dict.__contains__(per_id):  # 已经存在该特征
                faiss_idx = self.face_shot_dict[per_id]
                self.faiss_face_shot.remove(faiss_idx)
            img_path = os.path.join(self.config.clip_reid_face_path, file)
            # 打开图像并转换为RGB模式
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)
            extra_info = {"per_id": per_id, "img_path": img_path}
            feat = self.reid_model.extract_feature(img_np)
            faiss_idx = self.faiss_face_shot.add(feat, extra_info)  # 将特征加入特征库
            self.face_shot_dict[per_id] = faiss_idx
        # for file in removed: # 暂不考虑移除情况
        self.last_modify_time = new_mtime

    def on_destroy(self):
        self.reid_shared_memory.unlink()
        super().on_destroy()


def create_process(shared_memory, config_path: str):
    comp = ClipReidComponent(shared_memory, config_path)
    try:
        comp.start()
        shared_memory[GlobalKey.LAUNCH_COUNTER.name] += 1
        comp.update()
    except KeyboardInterrupt:
        comp.on_destroy()
    except Exception as e:
        logger.error(f"ClipReidComponent: {e}")
        logger.error(traceback.format_exc())  # 打印完整的堆栈信息
        comp.on_destroy()


if __name__ == '__main__':
    pass
