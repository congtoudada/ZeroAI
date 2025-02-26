import os
import time
import traceback
from typing import Dict

import cv2
import numpy as np
from loguru import logger

from jigsaw.zero.vad_jigsaw_helper import VadJigsawHelper
from simple_http.simple_http_helper import SimpleHttpHelper
from utility.config_kit import ConfigKit
from vad.vad_core.vad_info import VadInfo
from vad.vad_core.vad_item import VadItem
from vad_core.i_vad_frame_wrapper import IVadFrameWrapper
from vad_core.i_vad_obj_wrapper import IVadObjWrapper
from vad_mae.zero.vad_mae_helper import VadMaeHelper
from zero.core.based_stream_comp import BasedStreamComponent
from zero.key.detection_key import DetectionKey
from zero.key.global_key import GlobalKey
from zero.key.stream_key import StreamKey


class VadComponent(BasedStreamComponent):
    """
    Vad组件: 支持多台摄像头的帧级异常和对象级异常
    """

    def __init__(self, shared_memory, config_path: str):
        super().__init__(shared_memory)
        self.config: VadInfo = VadInfo(ConfigKit.load(config_path))  # 配置文件内容
        self.pname = f"[ {os.getpid()}:vad ]"
        # key: 端口索引
        self.data_dict: Dict[int, VadItem] = {}
        self.http_helper = SimpleHttpHelper(self.config.stream_http_config)  # http帮助类
        self.vad_frame_helper: IVadFrameWrapper = VadMaeHelper(self.config.vad_frame_config)  # 帧级别异常检测器
        self.vad_obj_helper: IVadObjWrapper = VadJigsawHelper(self.config.vad_obj_config,
                                                              self.config.vad_obj_nums)  # 对象级异常检测器
        self.last_tick_id = 0
        # self.aspect_scale = []   # 期望尺寸 / 原始尺寸 (h, w)
        self.bbox_limit = 3000  # bbox h*w最小值 基于(360,640)计算
        self.bbox_limits = []  # 根据实际分辨率换算

    def on_start(self):
        super().on_start()
        for i, input_port in enumerate(self.config.input_ports):
            cam_id = self.read_dict[i][StreamKey.STREAM_CAM_ID.name]  # 摄像头id
            width = self.read_dict[i][StreamKey.STREAM_WIDTH.name]
            height = self.read_dict[i][StreamKey.STREAM_HEIGHT.name]
            self.data_dict[i] = VadItem()
            capacity = int(max(self.config.vad_obj_nums, self.config.vad_frame_nums))
            self.data_dict[i].init(cam_id, capacity)
            # self.aspect_scale.append((self.config.vad_frame_resize[0] / height,
            #                           self.config.vad_frame_resize[1] / width))
            real_limit = self.bbox_limit * height / 360.0
            self.bbox_limits.append(real_limit)

    def on_get_stream(self, read_idx):
        frame, _ = super().on_get_stream(read_idx)  # 解析视频帧id+视频帧
        if frame is None:  # 没有有效帧
            return frame, None
        # 解析额外数据
        stream_package = self.read_dict[read_idx][self.config.input_ports[read_idx]]
        input_det = None
        if stream_package.__contains__(DetectionKey.DET_PACKAGE_RESULT.name):
            input_det = stream_package[DetectionKey.DET_PACKAGE_RESULT.name]  # 目标检测结果
        return frame, input_det

    def on_handle_stream(self, idx, frame, input_det) -> object:
        if self.config.vad_empty_filter:
            if input_det is None:  # 什么都没检测到，可返回
                return None
        # 帧级别异常检测器 (无batch优化)
        # pass
        now = self.frame_id_cache[idx]
        # 对象级异常检测器
        if not self.config.vad_obj_enable:
            self.data_dict[idx].push(cv2.resize(frame, self.config.vad_frame_resize[::-1]), now)
        else:
            # 1.过滤出人
            # 2.过滤置信度
            input_det = input_det[input_det[:, 5] == 0]
            input_det = input_det[input_det[:, 4] > self.config.vad_obj_det_conf]
            # # 3.过滤掉小物体
            # # 包围盒大小
            # w = input_det[:, 2] - input_det[:, 0]
            # h = input_det[:, 3] - input_det[:, 1]
            # wh = w * h
            # # print(np.min(w * h))
            # input_det = input_det[wh > self.bbox_limits[idx]]
            # 推入最新帧
            # self.data_dict[idx].push(cv2.resize(frame, self.config.vad_frame_resize[::-1]), now, input_det)
            self.data_dict[idx].push(frame, now, input_det)
            item = self.data_dict[idx]
            if now - item.last_obj_id >= self.config.vad_obj_interval:
                batch = item.get_batch(self.config.vad_obj_nums)
                if batch is not None:
                    det_idx = int(self.config.vad_obj_nums / 2) + 1  # 7-->4
                    middle_det = list(self.data_dict[idx].det_queue)[-det_idx]  # original scale
                    # input_det_scale[:, [0, 2]] *= self.aspect_scale[idx][1]  # width scale
                    # input_det_scale[:, [1, 3]] *= self.aspect_scale[idx][0]  # height scale
                    scores = self.vad_obj_helper.inference(batch, middle_det)
                    item.last_obj_id = now
                    print(f"{self.frame_id_cache[idx]}: {scores[0] * 100:.3f} | {scores[1] * 100:.3f}")
                    self.data_dict[idx].update_obj_score(1.0-scores[0], 1.0-scores[1],
                                                         self.config.vad_obj_spatial_threshold,
                                                         self.config.vad_obj_temporal_threshold,
                                                         self.config.vad_obj_s_times, self.config.vad_obj_t_times,
                                                         self.config.vad_obj_valid * 2)
        return input_det

    def on_update(self):
        super().on_update()
        # 基于cam0的帧更新所有逻辑
        frame_id = self.frame_id_cache[0]
        if self.last_tick_id == frame_id:
            return
        self.last_tick_id = frame_id
        # 帧级别异常检测器 (batch优化)
        if self.config.vad_frame_enable and self.config.vad_frame_batch_optimize:
            batches = []   # 所有满足帧异常检测的数据打包
            for k, v in self.data_dict.items():
                if v.frame_id - v.last_frame_id >= self.config.vad_frame_interval:
                    if not self.config.vad_obj_enable:
                        batch = v.get_batch(self.config.vad_frame_nums)
                    else:   # 开启对象级异常检测后需要手动缩放帧
                        batch = v.get_batch(self.config.vad_frame_nums, self.config.vad_frame_resize[::-1])
                    if batch is not None:
                        batches.append(batch)
                        v.last_frame_id = v.frame_id
            frame_scores = self.vad_frame_helper.inference_batch(batches)  # 各个摄像头帧List
            for i, score in enumerate(frame_scores):
                self.data_dict[i].update_frame_score(score, self.config.vad_frame_threshold,
                                                     self.config.vad_frame_times,
                                                     self.config.vad_frame_valid * 2)
                print(score)
            # if len(frame_scores) > 0:
            #     print(frame_scores)

    def on_draw_vis(self, idx, frame, input_mot):
        if len(self.data_dict) == 0:
            return frame
        frame = frame.copy()
        text_scale = 1
        text_thickness = 1
        # 标题线
        cv2.putText(frame, 'VAD threshold:%.3f score:%.3f | obj_s:%.3f | obj_t:%.3f' %
                    (self.config.vad_frame_threshold, self.data_dict[idx].frame_score,
                     self.data_dict[idx].obj_s_score, self.data_dict[idx].obj_t_score),
                    (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale,
                    (0, 0, 255), thickness=text_thickness)
        # 设置字体和大小
        if (self.data_dict[idx].frame_valid >= self.config.vad_frame_valid or
                self.data_dict[idx].obj_valid >= self.config.vad_obj_valid):
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = (0, 0, 255)  # 红色
            line_type = 3
            # 计算文本位置
            text = 'WARNING: Anomaly Detected!'
            text_size = cv2.getTextSize(text, font, font_scale, line_type)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = (frame.shape[0] + text_size[1]) // 2
            # 在帧上绘制文本
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, line_type)
        return frame


def create_process(shared_memory, config_path: str):
    comp = VadComponent(shared_memory, config_path)  # 创建组件
    try:
        comp.start()  # 初始化
        # 初始化结束通知
        shared_memory[GlobalKey.LAUNCH_COUNTER.name] += 1
        while not shared_memory[GlobalKey.ALL_READY.name]:
            time.sleep(0.1)
        comp.update()  # 算法逻辑循环
    except KeyboardInterrupt:
        comp.on_destroy()
    except Exception as e:
        logger.error(f"VadComponent: {e}")
        logger.error(traceback.format_exc())  # 打印完整的堆栈信息
        comp.on_destroy()
