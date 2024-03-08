import os
from typing import Dict

import cv2
import numpy as np
from loguru import logger

from count.component.count_comp import CountComponent
from zero.core.component.helper.face_helper_comp import FaceHelperComponent
from zero.core.key.shared_key import SharedKey


class CountFaceComponent(CountComponent):
    def __init__(self, shared_data, config_path: str):
        super().__init__(shared_data, config_path)
        self.pname = f"[ {os.getpid()}:face helper ]"
        self.helper = FaceHelperComponent(shared_data,
                                          self.config,
                                          self.stream_cam_id,
                                          self.face_callback)

    def on_start(self):
        super().on_start()
        self.helper.start()

    def on_update(self) -> bool:
        if super().on_update():
            for key, value in self.item_dict.items():
                if self.cen_send(key, value):
                    # img = cv2.imread('res/images/face/database/48-0001.jpg')
                    # self.helper.send(key, img)
                    self.helper.send(key, self._crop_img(self.frame, value.ltrb))
                    break  # 每次最多发送一个请求

            self.helper.update()  # helper特殊update
        return False

    def cen_send(self, obj_id, item):
        diff = self.current_frame_id - item.__getattribute__("last_face_req")
        if self.helper.can_send(obj_id, diff, item.base_y):
            # w = item.ltrb[2] - item.ltrb[0]
            # h = item.ltrb[3] - item.ltrb[1]
            # if w * h < 15 * 15:  # 包围盒太小，不发送
            #     return False
            self.item_dict[obj_id].__setattr__("last_face_req", self.current_frame_id)
            return True
        else:
            return False

    def on_destroy_obj(self, obj_id):
        self.helper.destroy_obj(obj_id)

    def on_create_obj(self, obj):
        obj.__setattr__("last_face_req", self.current_frame_id)

    def on_draw_vis(self, im):
        super().on_draw_vis(im)
        face_dict = self.helper.get_face_dict()
        # 参考线
        point1 = (0, int(self.helper.config.face_cull_y * self.stream_height))
        point2 = (self.stream_width, int(self.helper.config.face_cull_y * self.stream_height))
        point3 = (0, int((1 - self.helper.config.face_cull_y) * self.stream_height))
        point4 = (self.stream_width, int((1 - self.helper.config.face_cull_y) * self.stream_height))
        cv2.line(im, point1, point2, (127, 127, 127), 1)  # 绘制线条
        cv2.line(im, point3, point4, (127, 127, 127), 1)  # 绘制线条
        # 人脸识别结果
        for key, value in face_dict.items():
            ltrb = self.item_dict[key].ltrb
            cv2.putText(im, f"{face_dict[key]['per_id']}",
                        (int((ltrb[0] + ltrb[2]) / 2), int(self.item_dict[key].ltrb[1])),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), thickness=1)

    def face_callback(self, obj_id, per_id, score):
        pass

    def _crop_img(self, im, tlbr):
        x1, y1, x2, y2 = tlbr[0], tlbr[1], tlbr[2], tlbr[3]
        return np.ascontiguousarray(np.copy(im[int(y1): int(y2), int(x1): int(x2)]))
        # return im[int(y1): int(y2), int(x1): int(x2)]

    def _crop_img_border(self, im, tlbr, border=0):
        x1, y1, w, h = tlbr[0], tlbr[1], tlbr[2] - tlbr[0], tlbr[3] - tlbr[1]
        x2 = x1 + w + border
        x1 = x1 - border
        y2 = y1 + h + border
        y1 = y1 - border
        x1 = 0 if x1 < 0 else x1
        y1 = 0 if y1 < 0 else y1
        x2 = im.shape[1] if x2 > im.shape[1] else x2
        y2 = im.shape[0] if y2 > im.shape[0] else y2
        return np.ascontiguousarray(np.copy(im[int(y1): int(y2), int(x1): int(x2)]))

    def on_destroy(self):
        self.helper.destroy()
        super().on_destroy()


def create_count_face_process(shared_data, config_path: str):
    countFaceComp: CountFaceComponent = CountFaceComponent(shared_data, config_path)  # 创建组件
    countFaceComp.start()  # 初始化
    countFaceComp.update()  # 算法逻辑循环
