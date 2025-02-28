import multiprocessing
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Dict
import cv2
import numpy as np
from PIL import Image
from UltraDict import UltraDict
from loguru import logger

from clip_reid.zero.clip_reid_wrapper import ClipReidWrapper
from reid_core.i_reid_wrapper import IReidWrapper
from reid_core.reid_info import ReidInfo
from reid_core.reid_key import ReidKey
from utility.file_modify_kit import FileModifyKit
from utility.img_kit import ImgKit
from zero.core.component import Component
from zero.core.global_constant import GlobalConstant
from zero.helper.analysis_helper import AnalysisHelper
from zero.helper.faiss_helper import FaissHelper
from zero.key.global_key import GlobalKey
from utility.config_kit import ConfigKit
from utility.timer_kit import TimerKit
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class ReidComponent(Component):
    """
    ClipReid服务:
        1.所有请求会发送到一个Req Queue，由ClipReid服务轮询处理。举例: Ultradict['REID_REQ'].put({请求数据(含pid)})
        2.每个请求方需主动开辟一块共享内存作为Rsp Queue，ClipReid会把处理后的结果根据请求pid放到相应位置。举例: Ultradict['REID_RSP'+pid].put({响应数据})
    """
    SHARED_MEMORY_NAME = "reid"
    reid_helper_memory = UltraDict(name=SHARED_MEMORY_NAME, shared_lock=GlobalConstant.LOCK_MODE)

    def __init__(self, shared_memory, config_path: str):
        super().__init__(shared_memory)
        self.config: ReidInfo = ReidInfo(ConfigKit.load(config_path))  # 配置文件内容
        self.pname = f"[ {os.getpid()}:reid ]"
        self.reid_shared_memory = UltraDict(name=ReidComponent.SHARED_MEMORY_NAME, shared_lock=GlobalConstant.LOCK_MODE)
        self.req_queue = None  # 请求队列
        self.reid_model: IReidWrapper = ClipReidWrapper(self.config)  # clip_reid模型（理论上使用工厂模式更解耦，但我懒）
        # self.faiss_dict: Dict[int, FaissReidHelper] = {}  # 根据cam id分类的faiss字典(根据不同摄像头找人，暂不实现)
        self.camera_gallery: FaissHelper = None  # 根据cam id分类的faiss字典
        self.anomaly_gallery: FaissHelper = None  # 根据cam id分类的额外faiss字典，只存异常图
        self.time_flag = 0  # 时间标识，用于检查是否刷新特征半区
        self.last_modify_time = {}  # 各文件上次修改时间，用于检查是否需要重建由人脸生成的身份识别特征库
        self.face_gallery: FaissHelper = None  # face_shot 特征库
        self.face_gallery_dict: Dict[int, int] = {}  # per_id : faiss_idx
        self.infer_timer = TimerKit()  # 推理计时器

    def on_start(self):
        # 初始化请求缓存
        self.req_queue = multiprocessing.Manager().Queue()
        self.reid_shared_memory[ReidKey.REID_REQ.name] = self.req_queue
        if self.config.reid_debug_enable:
            if not os.path.exists(self.config.reid_debug_output):
                os.makedirs(self.config.reid_debug_output, exist_ok=True)
        # if os.path.exists(self.config.reid_face_gallery_dir):
        #     shutil.rmtree(self.config.reid_face_gallery_dir)
        # os.makedirs(self.config.reid_face_gallery_dir, exist_ok=True)
        # 删除前一天的reid结果
        yesterday = time.time() - 24 * 60 * 60  # 24小时之前的时间戳
        # 检查指定的目录是否存在
        if os.path.exists(self.config.reid_face_gallery_dir):
            # 遍历目录中的所有文件
            for filename in os.listdir(self.config.reid_face_gallery_dir):
                file_path = os.path.join(self.config.reid_face_gallery_dir, filename)
                # 获取文件的最后修改时间
                file_mtime = os.path.getmtime(file_path)
                # 如果文件的修改时间是前一天的文件，进行删除
                if file_mtime < yesterday:
                    if os.path.isfile(file_path):
                        os.remove(file_path)  # 删除文件
        os.makedirs(self.config.reid_face_gallery_dir, exist_ok=True)
        if os.path.exists(self.config.reid_camera_gallery_dir):
            shutil.rmtree(self.config.reid_camera_gallery_dir)
        os.makedirs(self.config.reid_camera_gallery_dir, exist_ok=True)
        if os.path.exists(self.config.reid_anomaly_gallery_dir):
            shutil.rmtree(self.config.reid_anomaly_gallery_dir)
        os.makedirs(self.config.reid_anomaly_gallery_dir, exist_ok=True)

        self.camera_gallery = FaissHelper(self.config.reid_dimension,
                                          self.config.reid_refresh_mode,
                                          self.config.reid_refresh_interval,
                                          self.config.reid_refresh_count,
                                          ReidComponent.remove_feat,
                                          self.config.reid_debug_enable,
                                          self.config.log_analysis)
        self.anomaly_gallery = FaissHelper(self.config.reid_dimension,
                                           1,   # 异常库基于特征数量刷新
                                           self.config.reid_refresh_interval,
                                           self.config.reid_refresh_count,
                                           ReidComponent.remove_feat,
                                           self.config.reid_debug_enable,
                                           self.config.log_analysis)
        self.face_gallery = FaissHelper(self.config.reid_dimension,
                                        self.config.reid_refresh_mode,
                                        self.config.reid_refresh_interval,
                                        self.config.reid_refresh_count,
                                        ReidComponent.remove_feat,
                                        self.config.reid_debug_enable,
                                        self.config.log_analysis)

    def on_update(self):
        # 处理请求
        while not self.req_queue.empty():
            req_package = self.req_queue.get()
            self.process_request(req_package)  # 处理每一个请求数据
            # break  # 每次最多处理一个响应
        # 记录推理平均耗时
        if self.config.log_analysis:
            AnalysisHelper.refresh("Reid Model inference average time", self.infer_timer.average_time * 1000)
            AnalysisHelper.refresh("Reid Model inference max time", self.infer_timer.max_time * 1000)
        # tick faiss
        self.time_flag = (self.time_flag + 1) % sys.maxsize
        self.camera_gallery.tick(self.time_flag)
        self.anomaly_gallery.tick(self.time_flag)

    def process_request(self, req_package):
        cam_id = req_package[ReidKey.REID_REQ_CAM_ID.name]  # 请求的摄像头id
        # 摄像头剔除
        if self.config.reid_cull_mode == 1:  # 只开启白名单
            if cam_id not in self.config.reid_white_list:
                logger.info(f"{self.pname} cam_id is not in white list: {self.config.reid_white_list}")
                return
        elif self.config.reid_cull_mode == 2:  # 只开启黑名单
            if cam_id in self.config.reid_black_list:
                logger.info(f"{self.pname} cam_id is in black list: {self.config.reid_black_list}")
                return
        elif self.config.reid_cull_mode == 3:  # 同时开启黑、白名单
            # 优先判断是否在黑名单
            if cam_id in self.config.reid_black_list:
                logger.info(f"{self.pname} cam_id is in black list: {self.config.reid_black_list}")
                return
            if cam_id not in self.config.reid_white_list:
                logger.info(f"{self.pname} cam_id is not in white list: {self.config.reid_white_list}")
                return
        pid = req_package[ReidKey.REID_REQ_PID.name]  # 请求的进程
        obj_id = req_package[ReidKey.REID_REQ_OBJ_ID.name]  # 请求的对象id
        reid_img = req_package[ReidKey.REID_REQ_IMAGE.name].copy()  # 请求的图片
        if reid_img is None or reid_img.shape[0] == 0 or reid_img.shape[1] == 0:
            logger.error(f"{self.pname} reid_img is None!")
            return
        reid_method = req_package[ReidKey.REID_REQ_METHOD.name]  # 请求方式

        # Reid抽特征 BGR-->RGB
        reid_rgb_img = reid_img[..., ::-1]  # 反转通道顺序
        self.infer_timer.tic()
        feat = self.reid_model.inference(reid_rgb_img)
        self.infer_timer.toc()
        # if cam_id not in self.faiss_dict:  # 首次添加
        #     self.faiss_dict[cam_id] = FaissReidHelper(self.config.reid_dimension,
        #                                               self.config.reid_refresh_mode,
        #                                               self.config.reid_refresh_interval,
        #                                               self.config.reid_refresh_count)
        if reid_method == 1:  # 1.普通存图请求
            # 将图片写入本地磁盘
            time_str = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
            img_path = os.path.join(self.config.reid_camera_gallery_dir, f"{obj_id}_{time_str}_{cam_id}.jpg")
            cv2.imwrite(img_path, reid_img)
            extra_info = {"img_path": img_path}
            self.camera_gallery.add(feat, extra_info)  # 将特征加入特征库
            n = self.camera_gallery.get_total()
            if self.config.reid_debug_enable and n % 50 == 0:
                logger.info(f"{self.pname} 当前camera gallery有效特征数: {n}")
        elif reid_method == 2:  # 2.reid识别请求，需要跟人脸截到的人像配准
            # 更新本地face shot特征库
            self.update_face_shot()
            k = 1  # topK的K值
            extra_info = self.face_gallery.search(feat, k, self.config.reid_threshold)
            per_id = 1
            score = 0
            if len(extra_info) == 0:
                logger.info(
                    f"{self.pname} Reid failed to fast reid: pid:{pid} cam_id:{cam_id} obj_id:{obj_id}")
            else:
                per_id = extra_info[0]['per_id']  # 从人脸gallery构建会添加per_id
                score = extra_info[0]['score']
            # debug输出
            if self.config.reid_debug_enable:
                if per_id != 1:
                    logger.info(
                        f"{self.pname} 识别成功! cam_id: {cam_id}, obj_id: {obj_id}, per_id: {per_id}, score: {score}")
                # reid_img = cv2.cvtColor(reid_img, cv2.COLOR_RGB2BGR)
                output_path = os.path.join(self.config.reid_debug_output,
                                           f"reid_cam{cam_id}_per{per_id}_score{score:.2f}.jpg")
                cv2.imwrite(output_path, reid_img)
            # 响应输出结果
            rsp_key = ReidKey.REID_RSP.name + str(pid)  # KEY: REID_RSP
            if self.reid_shared_memory.__contains__(rsp_key):
                self.reid_shared_memory[rsp_key].put({
                    ReidKey.REID_RSP_OBJ_ID.name: obj_id,
                    ReidKey.REID_RSP_PER_ID.name: per_id,
                    ReidKey.REID_RSP_SCORE.name: score
                })
                logger.info(
                    f"{self.pname} 响应Reid请求成功: pid:{pid} cam_id:{cam_id} obj_id:{obj_id} per_id:{per_id} score:{score:.2f}")
            # 额外存图(通常是异常检测到的对象)
            time_str = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
            img_path = os.path.join(self.config.reid_anomaly_gallery_dir, f"{obj_id}_{time_str}_{cam_id}.jpg")
            cv2.imwrite(img_path, reid_img)
            extra_info = {"img_path": img_path}
            self.anomaly_gallery.add(feat, extra_info)  # 将特征加入特征库
            if self.config.reid_debug_enable:
                logger.info(f"{self.pname} 当前anomaly gallery有效特征数: {self.anomaly_gallery.get_total()}")
        elif reid_method == 3:  # 找人
            k = 3  # topK的K值
            # 优先从异常库找
            extra_info = self.anomaly_gallery.search(feat, k, self.config.reid_search_person_threshold)
            k = k - len(extra_info)
            # 剩余的从普通库找
            if k > 0:
                extra_info = extra_info + self.camera_gallery.search(feat, k, self.config.reid_search_person_threshold)
            if len(extra_info) == 0:
                # method3中 obj_id是per_id
                logger.info(
                    f"{self.pname} Reid failed to search person: pid:{pid} cam_id:{cam_id} per_id:{obj_id}")
            rsp_key = ReidKey.REID_RSP_SP.name + str(pid)  # KEY: REID_RSP_SP
            if self.reid_shared_memory.__contains__(rsp_key):
                for item in extra_info:
                    item.pop("index")
                self.reid_shared_memory[rsp_key].put({
                    ReidKey.REID_RSP_SP_PACKAGE.name: extra_info,
                })
                logger.info(
                    f"{self.pname} 响应找人请求成功，匹配结果数: {len(extra_info)}")
        else:
            logger.error(f"{self.pname} Not found reid method: {reid_method}")

    def update_face_shot(self):
        # 检查是否存在文件修改
        added, removed, modified, new_mtime = FileModifyKit.check_changes(self.config.reid_face_gallery_dir,
                                                                          self.last_modify_time)
        added.update(modified)
        logger.info(f"{self.pname} update face gallery, modified num: {len(added)}")
        for file in added:
            per_id = file.split('_')[0]  # 首位存per id
            if self.face_gallery_dict.__contains__(per_id):  # 已经存在该特征
                faiss_idx = self.face_gallery_dict[per_id]
                self.face_gallery.remove(faiss_idx)
            img_path = os.path.join(self.config.reid_face_gallery_dir, file)
            # 打开图像并转换为RGB模式
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)
            extra_info = {"per_id": per_id, "img_path": img_path}
            feat = self.reid_model.inference(img_np)
            faiss_idx = self.face_gallery.add(feat, extra_info)  # 将特征加入特征库
            self.face_gallery_dict[per_id] = faiss_idx
        # for file in removed: # 暂不考虑移除情况
        self.last_modify_time = new_mtime

    @staticmethod
    def remove_feat(info_dict):
        if info_dict is None or info_dict is {}:
            return
        img_path = info_dict['img_path']
        if os.path.exists(img_path):
            os.remove(img_path)

    def on_destroy(self):
        # ReidComponent.reid_helper_memory.unlink()
        self.reid_shared_memory.unlink()  # 读写共享内存
        super().on_destroy()


def create_process(shared_memory, config_path: str):
    comp = ReidComponent(shared_memory, config_path)
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
    shared_memory = UltraDict(name="global", shared_lock=GlobalConstant.LOCK_MODE)
    shared_memory[GlobalKey.EVENT_ESC.name] = multiprocessing.Manager().Event()
    reid_comp = ReidComponent(shared_memory, config_path="conf/dev/service/reid/clip_reid/clip_reid.yaml")
    reid_comp.start()

    print('---------------------------- 测试请求方式1: 存图 ----------------------------')
    # 指定文件夹路径
    folder_path = Path("res/images/reid/gallery")

    # 遍历文件夹内的所有文件，拼接成相对路径
    img_database = [
        str(file)  # 转为字符串格式
        for file in folder_path.iterdir()
        if file.is_file()  # 只保留文件
    ]
    shutil.rmtree(reid_comp.config.reid_camera_gallery_dir)
    os.makedirs(reid_comp.config.reid_camera_gallery_dir, exist_ok=True)
    for i, img_path in enumerate(img_database):
        img = Image.open(img_path).convert('RGB')
        img_ndarray = np.array(img)[..., ::-1]  # RGB->BGR
        req_package = {
            ReidKey.REID_REQ_CAM_ID.name: 1,
            ReidKey.REID_REQ_PID.name: 991101,
            ReidKey.REID_REQ_OBJ_ID.name: i+1,
            ReidKey.REID_REQ_IMAGE.name: img_ndarray,
            ReidKey.REID_REQ_METHOD.name: 1  # 方式1
        }
        reid_comp.process_request(req_package)

    # 测试特征库刷新
    reid_comp.camera_gallery.tick(1)
    reid_comp.camera_gallery.tick(reid_comp.config.reid_refresh_interval+2)  # 切换半区
    # reid_comp.camera_gallery.tick(reid_comp.config.reid_refresh_interval*2 + 3)  # 切换半区

    print('---------------------------- 测试请求方式2: Fast Reid ----------------------------')
    query_path = "res/images/reid/query/0002_000_01_02.jpg"
    query_path = "output/service/clip_reid/tmp/Snipaste_2025-01-02_18-47-20.png"
    img = Image.open(query_path).convert('RGB')
    img_ndarray = np.array(img)[..., ::-1]  # RGB->BGR
    req_package = {
        ReidKey.REID_REQ_CAM_ID.name: 1,
        ReidKey.REID_REQ_PID.name: 991101,
        ReidKey.REID_REQ_OBJ_ID.name: 888,
        ReidKey.REID_REQ_IMAGE.name: img_ndarray,
        ReidKey.REID_REQ_METHOD.name: 2  # 方式2
    }
    reid_comp.process_request(req_package)

    print('---------------------------- 测试请求方式3: Search Person ----------------------------')
    # query_path = "res/images/reid/query/0002_000_01_02.jpg"
    query_path = "output/service/clip_reid/face_gallery/4_2025-01-01_23-53-48_9.png"
    img = Image.open(query_path).convert('RGB')
    img_ndarray = np.array(img)[..., ::-1]  # RGB->BGR
    req_package = {
        ReidKey.REID_REQ_CAM_ID.name: 1,
        ReidKey.REID_REQ_PID.name: 991101,
        ReidKey.REID_REQ_OBJ_ID.name: 888,
        ReidKey.REID_REQ_IMAGE.name: img_ndarray,
        ReidKey.REID_REQ_METHOD.name: 3  # 方式3
    }
    reid_comp.process_request(req_package)
    print(f"process average time: {reid_comp.infer_timer.average_time * 1000}ms")
    print(f"process max time: {reid_comp.infer_timer.max_time * 1000}ms")

    # print('---------------------------- 其他: 理论上保留1张图片在camera gallery ----------------------------')
    # req_package[ReidKey.REID_REQ_METHOD.name] = 1
    # reid_comp.process_request(req_package)
    # reid_comp.camera_gallery.tick(reid_comp.config.reid_refresh_interval*2 + 3)  # 切换半区
