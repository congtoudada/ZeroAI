import os
from abc import ABC
from typing import Dict, List

import cv2
import numpy as np
from loguru import logger

from bytetrack.zero.bytetrack_helper import BytetrackHelper
from simple_http.simple_http_helper import SimpleHttpHelper
from src.base.double_match_info import DoubleMatchInfo
from src.base.double_match_item import DoubleMatchItem
from src.utility.match_kit import DetectionRecord, MatchKit
from src.utility.warn_proxy import WarnProxy
from utility.config_kit import ConfigKit
from utility.img_kit import ImgKit
from utility.object_pool import ObjectPool
from zero.core.based_stream_comp import BasedStreamComponent
from zero.key.detection_key import DetectionKey
from zero.key.stream_key import StreamKey


class DoubleMatchComponent(BasedStreamComponent, ABC):
    """
    检测类别1 + 追踪类别2 (二者互相匹配)
    检测的类别1是我们重点关注，决定了任务的性质，我们称为主体(main)；追踪的类别2是次要关注的，我们称为次要个体(sub)
    * 安全帽任务中: 检测安全帽(main) + 追踪人(sub)
    * 手机任务中: 检测手机(main) + 追踪人(sub)
    """

    Warn_Desc = {
        1: "打电话异常",
        2: "安全帽异常"
    }

    def __init__(self, shared_memory, config_path: str):
        super().__init__(shared_memory)
        self.config: DoubleMatchInfo = DoubleMatchInfo(ConfigKit.load(config_path))
        self.pname = f"[ {os.getpid()}:double match for {self.config.input_ports[0]}&{self.config.input_ports[1]} ]"
        self.cam_id = 0
        self.stream_width = 0
        self.stream_height = 0
        self.item_pool: ObjectPool = ObjectPool(20, DoubleMatchItem)
        self.det_pool: ObjectPool = ObjectPool(40, DetectionRecord)
        self.item_dict: Dict[int, DoubleMatchItem] = {}
        self.tracker: BytetrackHelper = BytetrackHelper(self.config.stream_mot_config)  # 追踪器
        self.main_records: List[DetectionRecord] = []  # 主体目标检测结果
        self.sub_records: List[DetectionRecord] = []  # 次体目标检测结果
        self.http_helper = SimpleHttpHelper(self.config.stream_http_config)  # http帮助类
        self.is_clear_main_records = False  # 用于延迟清理主体检测结果，当mot有值是会在匹配后清理

    def on_start(self):
        super().on_start()
        self.cam_id = self.read_dict[0][StreamKey.STREAM_CAM_ID.name]
        self.stream_width = self.read_dict[0][StreamKey.STREAM_WIDTH.name]
        self.stream_height = self.read_dict[0][StreamKey.STREAM_HEIGHT.name]

    def on_update(self):
        self.release_unused()  # 清理无用资源（一定要在最前面调用）
        super().on_update()
        if self.is_clear_main_records:
            self.is_clear_main_records = False
            self.main_records.clear()

    def release_unused(self):
        # 清空长期未更新点
        clear_keys = []
        for key, item in self.item_dict.items():
            if self.frame_id_cache[1] - item.last_update_id > self.config.dm_lost_frame:
                clear_keys.append(key)
        clear_keys.reverse()
        for key in clear_keys:
            self.item_pool.push(self.item_dict[key])
            self.item_dict.pop(key)  # 从字典中移除item

    def on_get_stream(self, read_idx):
        frame, _ = super().on_get_stream(read_idx)  # 解析视频帧id+视频帧
        if frame is None:  # 没有有效帧
            return frame, None
        # 解析额外数据
        stream_package = self.read_dict[read_idx][self.config.input_ports[read_idx]]
        input_det = stream_package[DetectionKey.DET_PACKAGE_RESULT.name]  # 目标检测结果
        return frame, input_det

    def on_handle_stream(self, idx, frame, input_det):
        """
        处理视频流
        :param idx: 从input_ports[idx]取package
        :param frame: 帧
        :param input_det: 目标检测结果
        :return:
        """
        if input_det is None:
            return None

        if idx == 0:  # 安全帽
            for i in range(len(self.main_records)):
                self.det_pool.push(self.main_records[i])  # 归还到对象池
            self.main_records.clear()  # 有新的检测结果，清空旧安全帽检测记录
            for i, item in enumerate(input_det):
                ltrb = (item[0], item[1], item[2], item[3])
                score = item[4]
                cls = item[5]
                record = self.det_pool.pop()
                record.init(ltrb, score, cls)
                self.main_records.append(record)
            return None
        else:  # 人（此时安全帽记录已经填充）
            input_det = input_det[input_det[:, 5] == 0]
            mot_result = self.tracker.inference(input_det)  # 返回对齐输出后的mot结果
            if mot_result is not None:
                self.is_clear_main_records = True
            # 根据mot结果与主体检测结果匹配
            self.double_match_core(frame, mot_result, self.frame_id_cache[1])
            return mot_result

    def double_match_core(self, frame, input_mot, current_frame_id):
        # mot output: [n, 7]
        # n: n个对象
        # [0,1,2,3]: ltrb bboxes (基于视频流分辨率)
        #   [0]: x1
        #   [1]: y1
        #   [2]: x2
        #   [3]: y2
        # [4]: 置信度
        # [5]: 类别 (下标从0开始)
        # [6]: id
        if input_mot is None:
            return
        # 时间换精度: 根据t排序（y轴排序），配合包围盒匹配，可以使精度更高
        if self.config.dm_y_sort:
            sort_indices = np.argsort(input_mot[:, 1])
            input_mot = input_mot[sort_indices]
            self.main_records.sort(key=lambda x: x.ltrb[1])
        # 构造次体(人) records
        self.sub_records.clear()
        for i, obj in enumerate(input_mot):
            ltrb = obj[:4]
            score = obj[4]
            cls = obj[5]
            obj_id = int(obj[6])
            # 只有次体在合法区域内才匹配
            if not self._is_in_zone(ltrb, self.config.dm_zone):
                continue
            # 更新帧id，避免被回收
            if self.item_dict.__contains__(obj_id):
                self.item_dict[obj_id].common_update(current_frame_id, ltrb)
            else:
                item = self.item_pool.pop()
                item.init(obj_id, current_frame_id, ltrb)
                self.item_dict[obj_id] = item
            record: DetectionRecord = self.det_pool.pop()
            record.init(ltrb, score, cls, obj_id)
            self.sub_records.append(record)

        # 使用包围盒里外匹配，先遍历安全帽，再遍历人
        if self.config.dm_match_method:
            main_results, sub_results = MatchKit.match_bboxes(self.main_records, self.sub_records,
                                                              self.config.dm_match_tolerance)
        else:
            main_results, sub_results = MatchKit.match_bboxes(self.main_records, self.sub_records,
                                                              self.config.dm_match_tolerance)

        # 遍历结果集 (sub和main一一对应)
        for i in range(len(sub_results)):
            # 拿到结果记录索引
            sub_idx = sub_results[i]
            main_idx = main_results[i]
            # 拿到结果记录
            sub_record = self.sub_records[sub_idx]
            main_record = self.main_records[main_idx]
            # 更新结果
            item = self.item_dict[sub_record.obj_id]
            item.main_update(main_record.cls, main_record.ltrb, main_record.score)  # 安全帽类别更新
            # 结果收集
            self.process_result(frame, item)

        # 收尾，清理工作
        self.sub_records.clear()
        # self.main_records.clear()  # 延迟清理，需要可视化

    def process_result(self, frame, item: DoubleMatchItem):
        # 没有报过警且异常状态保持一段时间才发送
        if not item.has_warn and item.valid_count > self.config.dm_valid_count:
            logger.info(f"{self.pname} {DoubleMatchComponent.Warn_Desc[self.config.dm_warn_type]}: "
                        f"obj_id:{item.sub_obj_id} score:{item.main_score}")
            item.has_warn = True  # 一旦视为异常，则一直为异常，避免一个人重复报警
            shot_img = ImgKit.draw_img_box(frame, item.main_ltrb)
            # 发送报警信息给后端
            WarnProxy.send(self.http_helper, self.pname, self.output_dir[0], self.cam_id, self.config.dm_warn_type,
                           item.sub_per_id, shot_img, item.main_score, self.config.stream_export_img_enable,
                           self.config.stream_web_enable)

    def _is_in_zone(self, sub_ltrb, zone_ltrb):
        if len(zone_ltrb) == 0:
            return True
        base_x = (sub_ltrb[0] + sub_ltrb[2]) / 2. / self.stream_width
        base_y = (sub_ltrb[1] + sub_ltrb[3]) / 2. / self.stream_height
        if zone_ltrb[0] < base_x < zone_ltrb[2] and zone_ltrb[1] < base_y < zone_ltrb[3]:
            return True
        else:
            return False

    def on_draw_vis(self, idx, frame, input_mot):
        if input_mot is None:  # 检测安全帽的端口，不显示任何内容
            return None
        text_scale = 1
        text_thickness = 1
        line_thickness = 2
        # 标题线
        num = 0 if input_mot is None else input_mot.shape[0]
        cv2.putText(frame, 'inference_fps:%.2f num:%d' %
                    (1. / max(1e-5, self.update_timer.average_time),
                     num), (0, int(15 * text_scale)),
                    cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=text_thickness)
        # 合法检测区域
        if len(self.config.dm_zone) > 0:
            helmet_zone = self.config.dm_zone
            cv2.rectangle(frame, pt1=(int(helmet_zone[0] * self.stream_width), int(helmet_zone[1] * self.stream_height)),
                          pt2=(int(helmet_zone[2] * self.stream_width), int(helmet_zone[3] * self.stream_height)),
                          color=(0, 255, 0), thickness=line_thickness)
        # 对象基准点、包围盒
        if len(self.config.detection_labels) == 0:
            logger.warning(f"{self.pname} detection_labels的长度为0，请在配置文件中配置detection_labels!")
            return frame
        if input_mot is not None:
            for obj in input_mot:
                ltrb = obj[:4]
                obj_id = int(obj[6])
                # screen_x = int((ltrb[0] + ltrb[2]) / 2)
                # screen_y = int((ltrb[1] + ltrb[3]) / 2)
                cv2.circle(frame, (int(ltrb[0]), int(ltrb[1])), 4, (118, 154, 242), line_thickness)
                cv2.rectangle(frame, pt1=(int(ltrb[0]), int(ltrb[1])), pt2=(int(ltrb[2]), int(ltrb[3])),
                              color=self._get_color(obj_id), thickness=line_thickness)
                if self.item_dict.__contains__(obj_id):
                    cls = int(self.item_dict[obj_id].main_cls)
                    is_warn = self.item_dict[obj_id].has_warn
                    cv2.putText(frame, f"{obj_id}:{self.config.detection_labels[cls]} warn:{is_warn}",
                                (int(ltrb[0]), int(ltrb[1])),
                                cv2.FONT_HERSHEY_PLAIN, text_scale, self._get_color(obj_id), thickness=text_thickness)
                else:
                    cv2.putText(frame, f"{obj_id}",
                                (int(ltrb[0]), int(ltrb[1])),
                                cv2.FONT_HERSHEY_PLAIN, text_scale, self._get_color(obj_id), thickness=text_thickness)
        # 安全帽
        for i, item in enumerate(self.main_records):
            ltrb = item.ltrb
            cls = int(item.cls)
            score = item.score
            if cls in self.config.dm_draw_cls:  # 只绘制异常类型
                cv2.rectangle(frame, pt1=(int(ltrb[0]), int(ltrb[1])), pt2=(int(ltrb[2]), int(ltrb[3])),
                              color=(0, 0, 255), thickness=line_thickness)
            # id_text = f"cls:{self.config.detection_labels[cls]}({score:.2f})"
            # cv2.putText(frame, id_text, (int(ltrb[0]), int(ltrb[1])), cv2.FONT_HERSHEY_PLAIN,
            #             text_scale, (0, 0, 255), thickness=text_thickness)
        # 可视化并返回
        return frame

    def _get_color(self, idx):
        idx = (1 + idx) * 3
        color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
        return color