import os
from typing import Dict

import cv2
import numpy as np

from common.warn_kit import WarnKit
from helmet.component.HelmetItem import HelmetItem
from helmet.info.HelmetInfo import HelmetInfo
from zero.core.component.based.based_mot_comp import BasedMOTComponent
from zero.utility.config_kit import ConfigKit
from loguru import logger

from zero.utility.img_kit import ImgKit,ImgKit_img_box
from zero.utility.object_pool import ObjectPool
from zero.utility.timer_kit import TimerKit


class HelmetComponent(BasedMOTComponent):
    def __init__(self, shared_data, config_path: str):
        super().__init__(shared_data)
        self.config: HelmetInfo = HelmetInfo(ConfigKit.load(config_path))
        self.pname = f"[ {os.getpid()}:helmet for {self.config.input_port[0]}]"
        # key: obj_id value: cls
        self.pool: ObjectPool = ObjectPool(20, HelmetItem)
        self.data_dict: Dict[int, HelmetItem] = {}
        self.timer = TimerKit()

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
                    item.init(obj_id, cls, self.current_frame_id)
                    self.data_dict[obj_id] = item
                else:  # 已经记录过
                    self.data_dict[obj_id].update(self.current_frame_id, cls)
                self.postprocess_item(self.data_dict[obj_id], ltrb)
            self.timer.toc()
            return True
        return False

    def postprocess_item(self, helmet_item: HelmetItem, ltrb):
        if not helmet_item.has_warn and helmet_item.get_valid_count() >= self.config.helmet_valid_count:
            if helmet_item.cls == 0 or helmet_item.cls == 2:
                logger.info(f"安全帽佩戴异常: obj_id{helmet_item.obj_id} cls:{helmet_item.cls}")
                helmet_item.has_warn = True
                # shot_img = ImgKit.crop_img(self.frame, ltrb)
                # WarnKit.send_warn_result(self.pname, self.output_dir, self.stream_cam_id, 2, 1,
                #                          shot_img, self.config.stream_export_img_enable, self.config.stream_web_enable)

                shot_img = ImgKit_img_box.draw_img_box(self.frame, ltrb)
                WarnKit.send_warn_result(self.pname, self.output_dir, self.stream_cam_id, 2, 1,
                                         shot_img, self.config.stream_export_img_enable, self.config.stream_web_enable)

    def preprocess(self):
        # 清空长期未更新点
        clear_keys = []
        for key, item in self.data_dict.items():
            if self.current_frame_id - item.last_update_id > self.config.helmet_lost_frame:
                clear_keys.append(key)
        for key in clear_keys:
            self.pool.push(self.data_dict[key])
            self.data_dict.pop(key)  # 从字典中移除item

    def on_draw_vis(self, frame, vis=False, window_name="", is_copy=True):
        if is_copy:
            im = np.ascontiguousarray(np.copy(frame))
        else:
            im = frame
        text_scale = 1
        text_thickness = 1
        line_thickness = 2
        # 标题线
        num = 0 if self.input_mot is None else self.input_mot.shape[0]
        cv2.putText(im, 'frame:%d video_fps:%.2f inference_fps:%.2f num:%d' %
                    (self.current_frame_id,
                     1. / max(1e-5, self.update_timer.average_time),
                     1. / max(1e-5, self.timer.average_time),
                     num), (0, int(15 * text_scale)),
                    cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=text_thickness)

        # 对象基准点、包围盒
        if self.input_mot is not None:
            for obj in self.input_mot:
                cls = obj[5]
                if cls <= 2:
                    ltrb = obj[:4]
                    obj_id = int(obj[6])
                    screen_x = int((ltrb[0] + ltrb[2]) * 0.5)
                    screen_y = int((ltrb[1] + ltrb[3]) * 0.5)
                    cv2.circle(im, (screen_x, screen_y), 4, (118, 154, 242), line_thickness)
                    cv2.rectangle(im, pt1=(int(ltrb[0]), int(ltrb[1])), pt2=(int(ltrb[2]), int(ltrb[3])),
                                  color=(0, 0, 255), thickness=1)
                    cv2.putText(im, f"{obj_id}({cls})",
                                (int(ltrb[0]), int(ltrb[1])),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), thickness=1)
        # 可视化并返回
        return super().on_draw_vis(im, vis, window_name)


def create_process(shared_data, config_path: str):
    helmetComp: HelmetComponent = HelmetComponent(shared_data, config_path)  # 创建组件
    helmetComp.start()  # 初始化
    helmetComp.update()  # 算法逻辑循环
