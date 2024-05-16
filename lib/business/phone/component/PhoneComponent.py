import os
from typing import List
import cv2
import numpy as np
from loguru import logger
import requests
from math import sqrt

from bytetrack.zero.tracker.byte_tracker import BYTETracker, STrack
from phone.info.PhoneInfo import PhoneInfo
from zero.core.component.based.based_mot_comp import BasedMOTComponent
from zero.core.component.based.based_multi_det_comp import BasedMultiDetComponent
from zero.core.key.shared_key import SharedKey
from zero.utility.config_kit import ConfigKit
from zero.utility.timer_kit import TimerKit


class PhoneComponent(BasedMOTComponent):
    def __init__(self, shared_data, config_path: str):
        super().__init__(shared_data)
        self.config: PhoneInfo = PhoneInfo(ConfigKit.load(config_path))
        self.pname = f"[ {os.getpid()}:phone ]"
        self.output_boxes = []

    def on_update(self) -> bool:
        """
        同步状态
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
        :return:
        """
        person_bboxes = []
        phone_bboxes = []
        self.input_mot = None  # MOT输出结果 cls[0]:人 cls[1]:手机
        self.current_frame_id = 0  # 当前帧id
        self.stream_cam_id = 0  # 摄像头id
        self.frame = None  # 当前帧图像

        if super().on_update() and self.input_det is not None:
            for i in range(len(self.input_det)):  # 遍历内一个检测模型的输出
                if self.input_det[i] is not None:
                    for j in range(len(self.input_det[i])):  # 遍历每一个对象
                        # logger.info(f"来自{self.config.input_port[i]}端口，检测类别: {self.input_det[i][j][5]} 包围盒第1个索引: {self.input_det[i][j][1]}")
                        if self.config.input_port[i] == "camera1-yolox1":
                            if self.input_det[i][j][5] == 0.:
                                person_bboxes.append(self.input_det[i][j][:4])
                        elif self.config.input_port[i] == "camera1-yolox2":
                            phone_bboxes.append(self.input_det[i][j][:4])
                        if phone_bboxes:  # 检测到手机
                            self.output_boxes = self.on_calculate(person_bboxes, phone_bboxes)
                            logger.info(f"检测到玩手机的人 {self.output_boxes}")
                            print(f"检测到玩手机的人 {self.output_boxes}")
                            # url = ""
                            # get_response = requests.get(url)
                            # if get_response == 200:
                            #     post_response = request.post(url, data=self.output_boxes)
                            #     if post_response.ok:
                            #         print("POST successfully")
                            #     else:
                            #         print("POST failed")
                            # else:
                            #     print("REQUEST failed")
        return False

    def on_analysis(self):
        logger.info(f"{self.pname} video fps: {1. / max(1e-5, self.update_timer.average_time):.2f}"
                    f" inference fps: {1. / max(1e-5, self.timer.average_time):.2f}")

    def on_calculate(self, person_bboxes, phone_bboxes):
        res = []
        # 中心点计算
        person_centres = []
        for person_bbox in person_bboxes:
            person_centres.append([(person_bbox[0]+person_bbox[2])/2, (person_bbox[1]+person_bbox[3])/2])
        for phone_bbox in phone_bboxes:
            phone_centre = [(phone_bbox[0]+phone_bbox[2])/2, (phone_bbox[1]+phone_bbox[3])/2]
            min_index = -1  # 距离手机最近的人的下标
            min_dist = -1  # 最小距离
            for index, person_centre in enumerate(person_centres):
                if min_dist == -1:
                    min_index = index
                    min_dist = sqrt((phone_centre[0]-person_centre[0])**2 + (phone_centre[1]-person_centre[1])**2)
                else:
                    dist = sqrt((phone_centre[0]-person_centre[0])**2 + (phone_centre[1]-person_centre[1])**2)
                    if dist < min_dist:
                        min_index = index
                        min_dist = dist
            res.append(person_bboxes[min_index])
        return res

    def on_draw_vis(self, frame, vis=False, window_name="window", is_copy=True):
        if vis and frame is not None:
            if self.output_boxes:
                for box in self.output_boxes:
                    # 假设 self.box 是一个包含边界框坐标的列表或元组，形如 [x1, y1, x2, y2]
                    x1, y1, x2, y2 = box
                    # 定义线宽
                    line_thickness = 2
                    # 绘制矩形
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 0, 255), thickness=line_thickness)
            # 显示图像
            cv2.imshow(window_name, frame)
            # 如果按下 'q' 键，则退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.shared_data[SharedKey.EVENT_ESC].set()  # 退出程序
        # 返回处理后的帧
        return super().on_draw_vis(frame, vis, window_name)


def create_process(shared_data, config_path: str):
    phoneComp: PhoneComponent = PhoneComponent(shared_data, config_path)  # 创建组件
    phoneComp.start()  # 初始化
    phoneComp.update()  # 算法逻辑循环
