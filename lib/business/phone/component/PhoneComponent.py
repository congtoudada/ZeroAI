import os
import time

import cv2
from loguru import logger
from math import sqrt
from datetime import datetime, timedelta

from common.warn_kit import WarnKit
from phone.info.PhoneInfo import PhoneInfo
from zero.core.component.based.based_multi_mot_comp import BasedMultiMOTComponent
from zero.utility.config_kit import ConfigKit


class PhoneComponent(BasedMultiMOTComponent):
    def __init__(self, shared_data, config_path: str):
        super().__init__(shared_data)
        self.config: PhoneInfo = PhoneInfo(ConfigKit.load(config_path))
        self.pname = f"[ {os.getpid()}:phone ]"
        self.counter = {}  # 记录连续识别同一对象次数
        self.det_record = {}  # key: person_id  value: last_time
        self.timing_record = None  # 记录上一次定时保存的时间
        self.state = []  # 存储已经报警的id，避免重复报警

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
        person_bboxes = []
        phone_bboxes = []
        person_id = []
        object_bboxes = {}  # 检测目标的bbox

        if super().on_update() and self.input_mot is not None:
            for i in range(len(self.input_mot)):  # 遍历内一个检测模型的输出
                if self.input_mot[i] is not None:
                    for j in range(len(self.input_mot[i])):  # 遍历每一个对象
                        cls = int(self.input_mot[i][j][5])
                        if cls != 0:
                            continue  # 人和手机都是0
                        # logger.info(f"来自{self.config.input_port[i]}端口，检测类别: {self.input_mot[i][j][5]} obj_id: {self.input_mot[i][j][6]}")
                        # print(self.input_mot[i][j][5])
                        if self.config.input_port[i] == "camera3-bytetrack_person1":
                            if int(self.input_mot[i][j][5]) == 0:
                                person_bboxes.append(self.input_mot[i][j][:4])
                                person_id.append(self.input_mot[i][j][6])
                        elif self.config.input_port[i] == "camera3-bytetrack_phone1":
                            phone_bboxes.append(self.input_mot[i][j][:4])
                    if phone_bboxes:  # 检测到手机
                        output_bboxes, output_idx = self.on_calculate(person_bboxes, phone_bboxes)
                        for idx, bbox in zip(output_idx, output_bboxes):
                            object_bboxes[person_id[idx]] = bbox
                        # logger.info(f"检测到玩手机的人 {object_bboxes}")
                        # print(f"检测到玩手机的人 {object_bboxes}")
                        self.on_execute(object_bboxes)
        return False

    def on_analysis(self):
        logger.info(f"{self.pname} video fps: {1. / max(1e-5, self.update_timer.average_time):.2f}"
                    f" inference fps: {1. / max(1e-5, self.timer.average_time):.2f}")

    def on_calculate(self, person_bboxes, phone_bboxes):
        res_bbox = []
        res_idx = []
        # 中心点计算
        person_centres = []
        for person_bbox in person_bboxes:
            person_centres.append([(person_bbox[0] + person_bbox[2]) / 2, (person_bbox[1] + person_bbox[3]) / 2])
        for phone_bbox in phone_bboxes:
            phone_centre = [(phone_bbox[0] + phone_bbox[2]) / 2, (phone_bbox[1] + phone_bbox[3]) / 2]
            min_index = -1  # 距离手机最近的人的下标
            min_dist = -1  # 最小距离
            for index, person_centre in enumerate(person_centres):
                if min_dist == -1:
                    min_index = index
                    min_dist = sqrt(
                        (phone_centre[0] - person_centre[0]) ** 2 + (phone_centre[1] - person_centre[1]) ** 2)
                else:
                    dist = sqrt((phone_centre[0] - person_centre[0]) ** 2 + (phone_centre[1] - person_centre[1]) ** 2)
                    if dist < min_dist:
                        min_index = index
                        min_dist = dist
            if 0 <= min_index < len(person_bboxes):
                res_bbox.append(person_bboxes[min_index])
                res_idx.append(min_index)
        return res_bbox, res_idx

    def on_draw_vis(self, frame, vis=False, window_name="", is_copy=True):
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
        :param frame:
        :param vis:
        :param window_name:
        :param is_copy:
        :return:
        """
        if self.input_mot is not None:
            for i in range(len(self.input_mot)):  # 遍历内一个检测模型的输出
                if self.input_mot[i] is not None:
                    for obj in self.input_mot[i]:  # 遍历每一个对象
                        cls = int(obj[5])
                        if cls == 0:
                            ltrb = obj[:4]
                            obj_id = int(obj[6])
                            cv2.rectangle(frame, pt1=(int(ltrb[0]), int(ltrb[1])), pt2=(int(ltrb[2]), int(ltrb[3])),
                                          color=(0, 0, 255), thickness=1)
                            label = "person" if i == 0 else "phone"
                            cv2.putText(frame, f"{obj_id}({label})",
                                        (int(ltrb[0]), int(ltrb[1])),
                                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), thickness=1)
        # 可视化并返回
        return super().on_draw_vis(frame, vis, window_name)

    def on_execute(self, bboxes):
        # 定时存图
        delta = timedelta(seconds=2)
        now = datetime.now()
        last = self.timing_record
        if last is None or (now - last) >= delta:
            for id, bbox in bboxes.items():
                if self.config.timing_enable:
                    self.on_save_img(bbox, id, self.config.timing_path)
            # print(f"timely save {now}")
            self.timing_record = now

        # 报警
        max_gap = timedelta(seconds=0.5)
        for id in bboxes.keys():
            if id not in self.state:  # 没报过警
                if id not in self.det_record or now - self.det_record[id] > max_gap:
                    self.counter[id] = 1
                    self.det_record[id] = now
                elif now - self.det_record[id] <= max_gap:  # 连续检测到一个人
                    self.counter[id] += 1
                    self.det_record[id] = now
                    if self.counter[id] > 30:  # 连续检测到了k次进行报警
                        # url = ""
                        # get_response = requests.get(url)
                        # if get_response == 200:
                        #     post_response = request.post(url, data=)
                        #     if post_response.ok:
                        #         print("POST successfully")
                        #     else:
                        #         print("POST failed")
                        # else:
                        #     print("REQUEST failed")
                        logger.info(f"{self.pname} warning! id = {id}")
                        path, img = self.on_save_img(bboxes[id], id, self.config.warning_path)
                        WarnKit.send_warn_result(self.pname, self.output_dir, self.stream_cam_id, 1, 1,
                                                 img, self.config.stream_export_img_enable,
                                                 self.config.stream_web_enable)
                        self.state.append(id)
                        self.counter[id] = 0

    def on_save_img(self, bbox, id, path):
        img = self.frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        time_str = time.strftime('%Y%m%d%H%M%S', time.localtime())
        path = path + f"0_{self.stream_cam_id}_{time_str}.jpg"
        cv2.imwrite(path, img)
        return path, img


def create_process(shared_data, config_path: str):
    phoneComp: PhoneComponent = PhoneComponent(shared_data, config_path)  # 创建组件
    phoneComp.start()  # 初始化
    phoneComp.update()  # 算法逻辑循环


if __name__ == "__main__":
    time_str = time.strftime('%Y%m%d', time.localtime())
    print(time_str)
    id_str = f"{2:02d}"
    print(id_str)
