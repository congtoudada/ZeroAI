import os
from typing import List
import cv2
import numpy as np
from loguru import logger

from bytetrack.zero.info.bytetrack_info import BytetrackInfo
from bytetrack.zero.tracker.byte_tracker import BYTETracker, STrack
from zero.core.component.base.base_mot_comp import BaseMOTComponent
from zero.core.key.shared_key import SharedKey
from zero.utility.config_kit import ConfigKit
from zero.utility.timer_kit import TimerKit


class BytetrackComponent(BaseMOTComponent):
    def __init__(self, shared_data, config_path: str):
        super().__init__(shared_data)
        self.config: BytetrackInfo = BytetrackInfo(ConfigKit.load(config_path))
        self.pname = f"[ {os.getpid()}:bytetrack from {self.config.input_port[0]}]"
        # 自身定义
        self.tracker = None
        self.timer = TimerKit()
        # self.width = 640
        # self.height = 640
        # self.test_size = 640  # 无用
        self.online_tlwhs = []
        self.online_ltrbs = []
        self.online_ids = []
        self.online_scores = []
        self.online_classes = []

    def on_start(self):
        super().on_start()
        self.tracker = BYTETracker(self.config, frame_rate=self.config.bytetrack_args_fps)
        # self.width = self.shared_data[self.config.STREAM_ORIGINAL_WIDTH]
        # self.height = self.shared_data[self.config.STREAM_ORIGINAL_HEIGHT]
        # self.test_size = self.shared_data[self.config.DETECTION_TEST_SIZE]

    def on_update(self) -> bool:
        if super().on_update() and self.input_det is not None:
            self.timer.tic()
            self.inference_outputs = self.tracker.update(self.input_det)
            self.resolve_output(self.inference_outputs)  # 解析推理结果(基于目标检测，故一定存在目标)
            self.timer.toc()
            if self.config.bytetrack_vis:
                self._draw_vis()
        return False

    def on_analysis(self):
        logger.info(f"{self.pname} video fps: {1. / max(1e-5, self.update_timer.average_time):.2f}"
                    f" inference fps: {1. / max(1e-5, self.timer.average_time):.2f}")

    def on_resolve_output(self, online_targets: List[STrack]):
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
        self.online_tlwhs.clear()
        self.online_ltrbs.clear()
        self.online_ids.clear()
        self.online_scores.clear()
        self.online_classes.clear()
        for target in online_targets:
            tlwh = target.tlwh
            ltrb = target.tlbr  # 输出本质是ltrb
            vertical = tlwh[2] / tlwh[3] > self.config.bytetrack_args_aspect_ratio_thresh
            if tlwh[2] * tlwh[3] > self.config.bytetrack_args_min_box_area and not vertical:
                self.online_tlwhs.append(tlwh)
                self.online_ltrbs.append(ltrb)
                self.online_ids.append(target.track_id)
                self.online_scores.append(target.score)
                self.online_classes.append(target.cls)

        ltrbs = np.array(self.online_ltrbs).reshape(-1, 4)
        ids = np.array(self.online_ids).reshape(-1, 1)
        scores = np.array(self.online_scores).reshape(-1, 1)
        classes = np.array(self.online_classes).reshape(-1, 1)
        return np.concatenate((ltrbs, scores, classes, ids), axis=1)

    def _draw_vis(self):
        online_targets: List[STrack] = self.inference_outputs
        if online_targets is None:
            cv2.imshow("bytetrack window", self.frame)
        else:
            im = np.ascontiguousarray(np.copy(self.frame))
            # im_h, im_w = im.shape[:2]
            # scale = min(self.config.yolox_args_tsize / im_h, self.config.yolox_args_tsize / im_w)
            text_scale = 1
            text_thickness = 1
            line_thickness = 2

            cv2.putText(im, 'frame:%d video_fps:%.2f bytetrack_fps:%.2f num:%d' %
                        (self.current_frame_id,
                         1. / max(1e-5, self.update_timer.average_time),
                         1. / max(1e-5, self.timer.average_time),
                         len(self.inference_outputs)), (0, int(15 * text_scale)),
                        cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=text_thickness)
            for i, tlwh in enumerate(self.online_tlwhs):
                x1, y1, w, h = tlwh
                intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
                obj_id = int(self.online_ids[i])
                if self.online_scores is not None:
                    id_text = '{}:{:.2f}({})'.format(int(obj_id), self.online_scores[i],
                                                     self.config.detection_labels[int(self.online_classes[i])])
                else:
                    id_text = '{}'.format(int(obj_id))

                color = self.get_color(abs(obj_id))
                cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
                cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                            thickness=text_thickness)
            cv2.imshow("bytetrack window", im)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.config.bytetrack_vis = False
            self.shared_data[SharedKey.EVENT_ESC].set()  # 退出程序


def create_bytetrack_process(shared_data, config_path: str):
    bytetrackComp: BytetrackComponent = BytetrackComponent(shared_data, config_path)  # 创建组件
    bytetrackComp.start()  # 初始化
    bytetrackComp.update()  # 算法逻辑循环
