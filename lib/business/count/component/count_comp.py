import os
from typing import Dict

import cv2
import numpy
import numpy as np
from loguru import logger

from count.component.count_item import CountItem
from count.component.count_pool import CountPool
from count.info.count_info import CountInfo
from yolox.tracking_utils.timer import Timer
from zero.core.component.base.based_mot_comp import BasedMOTComponent
from zero.core.key.shared_key import SharedKey
from zero.utility.config_kit import ConfigKit


class CountComponent(BasedMOTComponent):
    def __init__(self, shared_data, config_path: str):
        super().__init__(shared_data)
        self.config: CountInfo = CountInfo(ConfigKit.load(config_path))
        self.pname = f"[ {os.getpid()}:count for {self.config.count_input_port}]"
        self.input_port = self.config.count_input_port
        # 自身定义
        self.in_count = 0  # 进入人数
        self.out_count = 0  # 离开人数
        self.pool: CountPool = CountPool(20, CountItem)
        self.item_dict: Dict[int, CountItem] = {}
        self.red_points = []
        self.red_vecs = []
        self.green_points = []
        self.green_vecs = []
        self.temp_red_result = []
        self.temp_green_result = []
        self.timer = Timer()

    def on_start(self):
        super().on_start()
        self.red_vecs.clear()
        self.green_vecs.clear()
        for i, point_str in enumerate(self.config.count_red):
            self.red_points.append((float(point_str.split(',')[0]), float(point_str.split(',')[1])))
            if i != 0:
                red = np.array([self.red_points[i][0] - self.red_points[i-1][0],
                                self.red_points[i][1] - self.red_points[i-1][1],
                                0])
                self.red_vecs.append(red / np.linalg.norm(red))
        for i, point_str in enumerate(self.config.count_green):
            self.green_points.append((float(point_str.split(',')[0]), float(point_str.split(',')[1])))
            if i != 0:
                green = np.array([self.green_points[i][0] - self.green_points[i-1][0],
                                 self.green_points[i][1] - self.green_points[i-1][1],
                                 0])
                self.green_vecs.append(green / np.linalg.norm(green))

    def on_update(self) -> bool:
        if super().on_update() and self.input_mot is not None:
            self.timer.tic()
            self.preprocess()
            self.process_update()
            self.process_result()
            self.timer.toc()
            if self.config.count_vis:
                self._draw_vis()
            self.postprocess()
        return False

    def preprocess(self):
        # 清空长期未更新点
        clear_keys = []
        for key, item in self.item_dict.items():
            if self.current_frame_id - item.last_update_id > self.config.count_lost_frames:
                clear_keys.append(key)
        for key in clear_keys:
            self.pool.push(self.item_dict[key])  # 放回对象池
            self.item_dict.pop(key)  # 从字典中移除

    def process_update(self):
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
        for obj in self.input_mot:
            ltrb = obj[:4]
            conf = obj[4]
            cls = int(obj[5])
            obj_id = int(obj[6])
            # 1.如果没有则添加
            if not self.item_dict.__contains__(obj_id):
                item: CountItem = self.pool.pop()
                item.init(obj_id, self.config.count_valid_frames)  # 初始化对象
                self.item_dict[obj_id] = item
            # 2.更新状态
            x, y = self._get_base(self.config.count_base, ltrb)
            self.item_dict[obj_id].update(self.current_frame_id, x / self.width, y / self.height)

    def process_result(self):
        """
        处理结果
        :return:
        """
        red_result = []
        green_result = []
        for item in self.item_dict.values():
            if not item.enable:  # 不是有效点，则跳过
                continue
            epsilon = 1e-9
            self.temp_red_result.clear()
            self.temp_green_result.clear()
            # 收集红绿信号（线上为0，线下为1，无效-1）
            for i in range(self.red_points.__len__() - 1):  # 最后一个点不计算
                vec3d = (item.base_x - self.red_points[i][0], item.base_y - self.red_points[i][1], 0)
                vec3d_length = np.linalg.norm(vec3d)
                if abs(vec3d_length) > epsilon:
                    vec3d = vec3d / vec3d_length
                    dot_ret = np.dot(self.red_vecs[i], vec3d)
                    cross_ret = np.cross(self.red_vecs[i], vec3d)[dot_ret > 0, 2]  # (n, 1) n为red_vec
                    self.temp_red_result.append(cross_ret)
            for i in range(self.green_points.__len__() - 1):  # 最后一个点不计算
                vec3d = (item.base_x - self.green_points[i][0], item.base_y - self.green_points[i][1], 0)
                vec3d_length = np.linalg.norm(vec3d)
                if abs(vec3d_length) > epsilon:
                    vec3d = vec3d / vec3d_length
                    dot_ret = np.dot(self.green_vecs[i], vec3d)
                    cross_ret = np.cross(self.green_vecs[i], vec3d)[dot_ret > 0, 2]
                    self.temp_green_result.append(cross_ret)
            item.update_red(self._process_cross(self.temp_red_result))
            item.update_green(self._process_cross(self.temp_green_result))
            # 处理红绿信号
            self._resolve_result(item)

    def postprocess(self):
        for item in self.item_dict.values():
            item.reset_update()  # 所有点无效，直到下次更新

    def _resolve_result(self, item: CountItem):
        """
        红绿信号（线上为0，线下为1，无效-1）
        解析规则：
            1.结果序列为空直接添加
            2.不为空，与数组最后一个元素结果不同才添加
            3.红绿序列长度为2且序列相同，计数有效。根据红序列第一个元素判断方向
            4.计数结果更新后将序列第一个元素删除
        :param item:
        :return:
        """
        if item.red_cur == -1 or item.green_cur == -1:
            return
        if item.red_seq.__len__() == 0:
            item.red_seq.append(item.red_cur)
            item.green_seq.append(item.green_cur)
        else:
            if item.red_seq[-1] != item.red_cur:
                item.red_seq.append(item.red_cur)
            if item.green_seq[-1] != item.green_cur:
                item.green_seq.append(item.green_cur)
            # 计数结果
            if len(item.red_seq) == self.config.count_req_len and item.red_seq == item.green_seq:
                ret = self._get_dir(item.red_seq[0] == 0, self.config.count_reverse)
                if ret:
                    self.in_count += 1
                else:
                    self.out_count += 1
                # 重置计数器
                item.red_seq.pop(0)
                item.green_seq.pop(0)

    def _draw_vis(self):
        im = np.ascontiguousarray(np.copy(self.frame))
        text_scale = 1
        text_thickness = 1
        line_thickness = 2
        # 标题线
        cv2.putText(im,
                    'frame: %d fps: %.2f num: %d in: %d out: %d' % (self.current_frame_id, 1. / max(1e-5, self.timer.average_time),
                                                     self.input_mot.shape[0], self.in_count, self.out_count), (0, int(15 * text_scale)),
                    cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=text_thickness)
        # 红线
        for i, red_point in enumerate(self.red_points):
            if i == 0:
                continue
            cv2.line(im, (int(self.red_points[i][0] * self.width), int(self.red_points[i][1] * self.height)),
                     (int(self.red_points[i-1][0] * self.width), int(self.red_points[i-1][1] * self.height)),
                     (0, 0, 255), line_thickness)  # 绘制线条
        # 绿线
        for i, green_point in enumerate(self.green_points):
            if i == 0:
                continue
            cv2.line(im, (int(self.green_points[i][0] * self.width), int(self.green_points[i][1] * self.height)),
                     (int(self.green_points[i-1][0] * self.width), int(self.green_points[i-1][1] * self.height)),
                     (255, 0, 0), line_thickness)  # 绘制线条
        # 对象基准点、红绿信息
        for item in self.item_dict.values():
            screen_x = int(item.base_x * self.width)
            screen_y = int(item.base_y * self.height)
            cv2.circle(im, (screen_x, screen_y), 4, (118, 154, 242), line_thickness)
            cv2.putText(im, str(item.red_cur), (screen_x, screen_y), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                        thickness=text_thickness)
            cv2.putText(im, str(item.green_cur), (screen_x + 10, screen_y), cv2.FONT_HERSHEY_PLAIN, text_scale, (255, 0, 0),
                        thickness=text_thickness)
        # 对象包围盒
        for obj in self.input_mot:
            ltrb = obj[:4]
            cv2.rectangle(im, pt1=(int(ltrb[0]), int(ltrb[1])), pt2=(int(ltrb[2]), int(ltrb[3])),
                          color=(0, 0, 255), thickness=1)
        cv2.imshow("count window", im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.config.bytetrack_vis = False
            self.shared_data[SharedKey.EVENT_ESC].set()  # 退出程序

    def _get_base(self, base, ltrb):
        """
        检测基准 0:包围盒中心点 1:包围盒左上角
        :param base:
        :return:
        """
        if base == 0:
            return (ltrb[0] + ltrb[2]) / 2, (ltrb[1] + ltrb[3]) / 2
        else:
            return ltrb[0], ltrb[1]

    def _get_dir(self, dir, reverse):
        # 无反向: dir = True, 在线之上, 进入, return True; dir = False, 在线之下，离开, return False
        if not reverse:
            return dir
        else:
            return not dir

    def _process_cross(self, results):
        # 全部 < 0, 在线之上, return 0; 存在 > 0, return 1; 接近0, 在线附近无效, return -1
        epsilon = 1e-9
        for re in results:
            if abs(re) < epsilon:
                return -1
            if re > 0:
                return 1
        return 0


def create_count_process(shared_data, config_path: str):
    countComp: CountComponent = CountComponent(shared_data, config_path)  # 创建组件
    countComp.start()  # 初始化
    countComp.update()  # 算法逻辑循环


if __name__ == '__main__':
    arr1 = [0, 1]
    arr2 = [0]
    arr3 = np.array(arr1)
    arr4 = np.array(arr2)
    print(arr1 == arr2)
    print(np.array_equal(arr3, arr4))


    # [0,0.5] [0.75,0.5] [1, 0.6]
    # [0.25,0.25]
    ref_vec = np.array([0.75, 0, 0, -0.25, 0.1, 0]).reshape(2, 3)
    input_vec = np.array([0.25, -0.25, 0])
    print(np.dot(ref_vec, input_vec)[np.dot(ref_vec, input_vec) < 0])
    # print(np.cross(ref_vec, input_vec))

    # ref_vec = np.array([0.75, 0, 0])
    # ref_vec = np.array([0.25, 0.1, 0])
    # print(np.cross(ref_vec, input_vec))
