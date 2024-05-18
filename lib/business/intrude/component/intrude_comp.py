import os
from typing import Dict

import cv2
import numpy as np
from loguru import logger

from common.warn_kit import WarnKit
from intrude.component.intrude_item import IntrudeItem
from intrude.info.intrude_info import IntrudeInfo
from zero.core.component.based.based_mot_comp import BasedMOTComponent
from zero.utility.config_kit import ConfigKit
from zero.utility.img_kit import ImgKit
from zero.utility.object_pool import ObjectPool
from zero.utility.timer_kit import TimerKit


class IntrudeComponent(BasedMOTComponent):
    def __init__(self, shared_data, config_path: str):
        super().__init__(shared_data)
        self.config: IntrudeInfo = IntrudeInfo(ConfigKit.load(config_path))
        self.pname = f"[ {os.getpid()}:intrude for {self.config.input_port[0]}]"
        self.pool: ObjectPool = ObjectPool(20, IntrudeItem)
        self.data_dict: Dict[int, IntrudeItem] = {}
        self.zone_points = []
        self.zone_vec = []
        self.timer = TimerKit()

    def on_start(self):
        super().on_start()
        for point_str in self.config.intrude_zone:
            per_x = float(point_str.split(',')[0])
            per_y = float(point_str.split(',')[1])
            self.zone_points.append((per_x, per_y))
        for i in range(len(self.zone_points)):  # 最后一个点除外
            if i == 0:
                continue
            vec = (self.zone_points[i][0] - self.zone_points[i - 1][0],
                   self.zone_points[i][1] - self.zone_points[i - 1][1],
                   0)
            self.zone_vec.append(vec / np.linalg.norm(vec))

    def on_update(self) -> bool:
        """
        # mot output shape: [n, 7]
        # n: n个对象
        # [0,1,2,3]: tlbr bboxes (基于视频流分辨率)
        #   [0]: x1
        #   [1]: y1
        #   [2]: x2
        #   [3]: y2
        # [4]: 置信度
        # [5]: 类别 (下标从0开始)
        # [6]: id
        """
        if super().on_update() and self.input_mot is not None:
            self.preprocess()
            self.timer.tic()
            for obj in self.input_mot:
                ltrb = obj[:4]
                conf = obj[4]
                cls = int(obj[5])
                obj_id = int(obj[6])
                if not self.data_dict.__contains__(obj_id):  # 没有被记录过
                    item = self.pool.pop()
                    item.init(obj_id, self.current_frame_id)
                    self.data_dict[obj_id] = item
                else:  # 已经记录过
                    in_warn = self.is_in_warn(ltrb)  # 判断是否处于警戒区
                    self.data_dict[obj_id].update(self.current_frame_id, in_warn)
                self.postprocess_item(self.data_dict[obj_id], ltrb)
            self.timer.toc()
            return True
        return False

    def postprocess_item(self, intrude_item: IntrudeItem, ltrb):
        if not intrude_item.has_warn and intrude_item.get_valid_count() >= self.config.intrude_valid_count:
            logger.info("入侵异常")
            shot_img = ImgKit.crop_img(self.frame, ltrb)
            WarnKit.send_warn_result(self.pname, self.output_dir, self.stream_cam_id, 4, 1,
                                     shot_img, self.config.stream_export_img_enable, self.config.stream_web_enable)
            intrude_item.has_warn = True

    def preprocess(self):
        """
        清空长期未更新点
        :return:
        """
        clear_keys = []
        for key, item in self.data_dict.items():
            if self.current_frame_id - item.last_update_id > self.config.intrude_lost_frame:
                clear_keys.append(key)
        for key in clear_keys:
            self.pool.push(self.data_dict[key])
            self.data_dict.pop(key)  # 从字典中移除item

    def is_in_warn(self, ltrb) -> bool:
        base_x, base_y = self.cal_center(ltrb)
        epsilon = 1e-3
        tmp = -1
        for i in range(len(self.zone_points) - 1):  # 最后一个点不计算
            p2o = (base_x - self.zone_points[i][0], base_y - self.zone_points[i][1], 0)  # 区域点->当前点的向量
            p2o_len = np.linalg.norm(p2o)
            if (abs(p2o_len)) > epsilon:  # 避免出现零向量导致叉乘无意义
                cross_z = np.cross(self.zone_vec[i], p2o)[2]  # (n, 1) n为red_vec
                if i == 0:
                    tmp = cross_z
                else:
                    if tmp * cross_z < 0:  # 出现异号说明不在区域内
                        return False
        return True

    def cal_center(self, ltrb):
        """
        计算中心点视口坐标作为2D参考坐标
        :param ltrb:
        :return:
        """
        center_x = (ltrb[0] + ltrb[2]) * 0.5 / self.stream_width
        center_y = (ltrb[1] + ltrb[3]) * 0.5 / self.stream_height
        return center_x, center_y

    def on_draw_vis(self, frame, vis=False, window_name="", is_copy=True):
        if is_copy:
            im = np.ascontiguousarray(np.copy(frame))
        else:
            im = frame
        text_scale = 1
        text_thickness = 1
        line_thickness = 2
        # 标题线
        cv2.putText(im, 'frame:%d video_fps:%.2f inference_fps:%.2f num:%d' %
                    (self.current_frame_id,
                     1. / max(1e-5, self.update_timer.average_time),
                     1. / max(1e-5, self.timer.average_time),
                     self.input_mot.shape[0]), (0, int(15 * text_scale)),
                    cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=text_thickness)
        # 警戒线
        for i, point in enumerate(self.zone_points):
            if i == 0:
                continue
            cv2.line(im, (
            int(self.zone_points[i][0] * self.stream_width), int(self.zone_points[i][1] * self.stream_height)),
                     (int(self.zone_points[i - 1][0] * self.stream_width),
                      int(self.zone_points[i - 1][1] * self.stream_height)),
                     (0, 0, 255), line_thickness)  # 绘制线条

        # 对象基准点、包围盒
        for obj in self.input_mot:
            ltrb = obj[:4]
            obj_id = int(obj[6])
            screen_x = int((ltrb[0] + ltrb[2]) * 0.5)
            screen_y = int((ltrb[1] + ltrb[3]) * 0.5)
            cv2.circle(im, (screen_x, screen_y), 4, (118, 154, 242), line_thickness)
            cv2.rectangle(im, pt1=(int(ltrb[0]), int(ltrb[1])), pt2=(int(ltrb[2]), int(ltrb[3])),
                          color=(0, 0, 255), thickness=1)
            cv2.putText(im, f"{obj_id}",
                        (int(ltrb[0]), int(ltrb[1])),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), thickness=1)
        # 可视化并返回
        return super().on_draw_vis(im, vis, window_name)


def create_process(shared_data, config_path: str):
    intrudeComp: IntrudeComponent = IntrudeComponent(shared_data, config_path)  # 创建组件
    intrudeComp.start()  # 初始化
    intrudeComp.update()  # 算法逻辑循环
