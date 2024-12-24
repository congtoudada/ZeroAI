import os
from typing import List
import cv2
import numpy as np

from bytetrack.zero.info.bytetrack_info import BytetrackInfo
from bytetrack.zero.tracker.byte_tracker import BYTETracker, STrack
from utility.config_kit import ConfigKit


class BytetrackHelper:
    def __init__(self, config_path: str):
        self.config: BytetrackInfo = BytetrackInfo(ConfigKit.load(config_path))
        self.pname = f"[ {os.getpid()}:bytetrack ]"
        # byetrack 模型
        self.tracker = BYTETracker(self.config, frame_rate=self.config.bytetrack_args_fps)

    def inference(self, input_det):
        if input_det is None:
            return None
        else:
            return self.alignment_result(self.tracker.update(input_det))

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