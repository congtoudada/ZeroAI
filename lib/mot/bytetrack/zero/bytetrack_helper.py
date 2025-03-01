import os
from typing import List
import cv2
import numpy as np

from bytetrack.zero.bytetrack_info import BytetrackInfo
from bytetrack.zero.byte_tracker import BYTETracker, STrack
from reid_core.helper.reid_helper import ReidHelper
from utility.config_kit import ConfigKit
from utility.img_kit import ImgKit


class BytetrackHelper:
    def __init__(self, config_path: str):
        self.config: BytetrackInfo = BytetrackInfo(ConfigKit.load(config_path))
        self.pid = os.getpid()
        self.pname = f"[ {self.pid}:bytetrack ]"
        # byetrack 模型
        self.tracker = BYTETracker(self.config, frame_rate=self.config.bytetrack_args_fps)
        # reid存图缓存
        # key: obj_id
        # value: {count: 0, last_time: 0, last_save_time: 0}  存图次数,上次更新时间,上次存图时间
        self.reid_cache = {}

    def inference(self, input_det, now=0, frame=None, cam_id=1):
        if input_det is None:
            return None
        else:
            result = self.alignment_result(self.tracker.update(input_det))
            # reid存图
            self.save_reid_extension(now, result, frame, cam_id)
            return result

    def save_reid_extension(self, now, mot_result, frame, cam_id=1):
        if not self.config.bytetrack_reid_enable:
            return
        if mot_result is None:
            return
        if frame is None:
            return
        # 清除无用缓存
        clear_list = []
        for k, v in self.reid_cache.items():
            if now - v['last_time'] > self.config.bytetrack_reid_lost_frames:
                clear_list.append(k)
        for k in clear_list:
            self.reid_cache.pop(k)
        # 遍历-存图
        for obj in mot_result:
            cls = int(obj[5])  # 提取当前目标的类别，转换为整数类型
            # 不是人不存(第0类)
            if cls != 0:
                continue
            obj_id = int(obj[6])  # 提取当前目标的唯一标识符，转换为整数类型
            cache_item = None
            if not self.reid_cache.__contains__(obj_id):
                cache_item = {'count': 0, 'last_time': now, 'last_save_time': 0}
                self.reid_cache[obj_id] = cache_item
            else:
                # 发送间隔不满足的跳过
                cache_item = self.reid_cache[obj_id]
                cache_item['last_time'] = now  # 保持对象活性
                valid_interval = min(self.config.bytetrack_reid_max_interval,
                                     self.config.bytetrack_reid_min_interval * cache_item['count'])
                if now - cache_item['last_save_time'] < valid_interval:
                    continue
            conf = obj[4]  # 提取当前目标的置信度
            # 置信度低的不存
            if conf < self.config.bytetrack_reid_conf:
                continue
            ltrb = obj[:4]  # 提取当前目标的边界框坐标，即左上角和右下角的坐标
            # 包围盒太小的不存或比例奇怪的不存
            w = ltrb[2] - ltrb[0]
            h = ltrb[3] - ltrb[1]
            if w < 40 or h < 80:
                continue
            shot_img = ImgKit.crop_img(frame, ltrb)
            if shot_img is not None:
                ReidHelper.send_save_timing(cam_id, self.pid, obj_id, shot_img)
                cache_item['count'] += 1
                cache_item['last_save_time'] = now

    def alignment_result(self, online_targets: List[STrack]):
        """
        # output shape: [n, 7]
        # n: n个对象
        # [0,1,2,3]: ltrb bboxes (基于视频流分辨率)
        #   [0]: x1
        #   [1]: y1
        #   [2]: x2
        #   [3]: y2
        # [4]: 置信度
        # [5]: 类别 (下标从0开始)
        # [6]: id
        :param online_targets:
        :return:
        """
        online_tlwhs = []
        online_ltrbs = []
        online_ids = []
        online_scores = []
        online_classes = []
        for target in online_targets:
            tlwh = target.tlwh
            ltrb = target.tlbr  # 输出本质是ltrb
            vertical = tlwh[2] / tlwh[3] > self.config.bytetrack_args_aspect_ratio_thresh
            if tlwh[2] * tlwh[3] > self.config.bytetrack_args_min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ltrbs.append(ltrb)
                online_ids.append(target.track_id)
                online_scores.append(target.score)
                online_classes.append(target.cls)

        ltrbs = np.array(online_ltrbs).reshape(-1, 4)
        ids = np.array(online_ids).reshape(-1, 1)
        scores = np.array(online_scores).reshape(-1, 1)
        classes = np.array(online_classes).reshape(-1, 1)
        return np.concatenate((ltrbs, scores, classes, ids), axis=1)

    def draw(self, frame, result):
        online_targets: List[STrack] = result
        if online_targets is not None:
            text_scale = 1
            text_thickness = 1
            line_thickness = 2

            for i, obj in enumerate(result):
                x1, y1, w, h = obj[0], obj[1], obj[2], obj[3]
                intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
                obj_id = obj[6]
                cls = int(obj[5])
                score = obj[4]
                if cls < len(self.config.detection_labels):
                    id_text = '{}:{:.2f}({})'.format(int(obj_id), score,
                                                     self.config.detection_labels[cls])
                    color = self.get_color(obj_id)
                    cv2.rectangle(frame, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
                    cv2.putText(frame, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                                thickness=text_thickness)
        return frame

    def get_color(self, idx):
        idx = idx * 3
        color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
        return color