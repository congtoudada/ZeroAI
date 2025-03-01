import os
import time
import traceback
from typing import Dict
import cv2
import numpy as np
from loguru import logger

from src.intrude.intrude_info import IntrudeInfo
from src.intrude.intrude_item import IntrudeItem
from bytetrack.zero.bytetrack_helper import BytetrackHelper
from insight.zero.component.face_helper import FaceHelper
from simple_http.simple_http_helper import SimpleHttpHelper
from src.utility.warn_proxy import WarnProxy
from zero.core.based_stream_comp import BasedStreamComponent
from zero.key.detection_key import DetectionKey
from zero.key.global_key import GlobalKey
from zero.key.stream_key import StreamKey
from utility.config_kit import ConfigKit
from utility.img_kit import ImgKit
from utility.object_pool import ObjectPool


class IntrudeComponent(BasedStreamComponent):
    """
    特定区域入侵检测
    """

    def __init__(self, shared_memory, config_path: str):
        super().__init__(shared_memory)
        self.config: IntrudeInfo = IntrudeInfo(ConfigKit.load(config_path))
        self.pname = f"[ {os.getpid()}:intrude for {self.config.input_ports[0]}]"
        self.pool: ObjectPool = ObjectPool(20, IntrudeItem)
        self.cam_id = 0
        self.stream_width = 0
        self.stream_height = 0
        self.data_dict: Dict[int, IntrudeItem] = {}
        self.zone_points = []
        self.zone_vec = []
        self.tracker: BytetrackHelper = BytetrackHelper(self.config.stream_mot_config)  # 追踪器
        self.http_helper = SimpleHttpHelper(self.config.stream_http_config)  # http帮助类
        self.face_helper: FaceHelper = None
        self.intrude_zone = []  # 检测区域像素 ltrb

    def on_start(self):
        super().on_start()
        if self.config.intrude_face_enable:
            self.face_helper = FaceHelper(self.config.intrude_face_config, self._face_callback)
        self.cam_id = self.read_dict[0][StreamKey.STREAM_CAM_ID.name]
        self.stream_width = int(self.read_dict[0][StreamKey.STREAM_WIDTH.name])
        self.stream_height = int(self.read_dict[0][StreamKey.STREAM_HEIGHT.name])
        # 预计算
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

    def on_update(self):
        self.release_unused()  # 清理无用资源（一定要在最前面调用）
        super().on_update()
        if self.config.intrude_face_enable:
            # 人脸识别请求
            current_id = self.frame_id_cache[0]
            for key, value in self.data_dict.items():
                # 没有进入报警区域，就直接返回
                if value.valid_count == 0:
                    continue
                self.face_helper.try_send(current_id, self.frames[0], value.ltrb, key, value.base_x,
                                          value.base_y, self.cam_id)
            # 人脸识别帮助tick，用于接受响应
            self.face_helper.tick(current_id)

    def _face_callback(self, obj_id, per_id, score):
        if self.data_dict.__contains__(obj_id):
            self.data_dict[obj_id].per_id = per_id
            self.data_dict[obj_id].score = score

    def on_get_stream(self, read_idx):
        frame, _ = super().on_get_stream(read_idx)  # 解析视频帧id+视频帧
        if frame is None:  # 没有有效帧
            return frame, None
        # 解析额外数据
        stream_package = self.read_dict[read_idx][self.config.input_ports[read_idx]]
        input_det = stream_package[DetectionKey.DET_PACKAGE_RESULT.name]  # 目标检测结果
        return frame, input_det

    def on_handle_stream(self, idx, frame, input_det):
        if input_det is None:
            return None

        input_det = input_det[input_det[:, 5] == 0]
        frame_id = self.frame_id_cache[0]
        mot_result = self.tracker.inference(input_det, frame_id, frame, self.cam_id)  # 返回对齐输出后的mot结果
        # 根据mot结果进行计数
        self._intrude_core(frame, mot_result, frame_id, frame.shape[1], frame.shape[0])
        return mot_result

    def _intrude_core(self, frame, input_mot, current_frame_id, width, height) -> bool:
        """
        # mot output shape: [n, 7]
        # n: n个对象
        # [0,1,2,3]: ltrb bboxes (基于视频流分辨率)
        #   [0]: x1
        #   [1]: y1
        #   [2]: x2
        #   [3]: y2
        # [4]: 置信度
        # [5]: 类别 (下标从0开始)
        # [6]: id
        """
        if input_mot is None:
            return
        # 清空前一帧状态
        for item in self.data_dict.values():
            item.reset_update()

        for obj in input_mot:
            ltrb = obj[:4]
            conf = obj[4]
            cls = int(obj[5])
            obj_id = int(obj[6])
            if cls == 0:  # 人
                # 更新Item状态
                if not self.data_dict.__contains__(obj_id):  # 没有被记录过
                    item = self.pool.pop()
                    item.init(obj_id, current_frame_id)
                    self.data_dict[obj_id] = item
                else:  # 已经记录过（更新状态）
                    in_warn = self._is_in_warn(ltrb)  # 判断是否处于警戒区
                    x, y = self._get_base(0, ltrb)  # 基于包围盒中心点计算百分比x,y
                    self.data_dict[obj_id].update(current_frame_id, in_warn, x / width, y / height, ltrb)

                # 处理Item结果
                item = self.data_dict[obj_id]
                # 如果开启人脸检测，小于重试次数的陌生人不报警
                if self.face_helper.face_dict.__contains__(obj_id):
                    retry = self.face_helper.face_dict[obj_id]["retry"]
                else:
                    retry = 0
                if self.config.intrude_face_enable:
                    if item.per_id == 1 and retry < self.face_helper.config.face_max_retry:
                        continue
                # 如果Item没有报过警且报警帧数超过有效帧，判定为入侵异常
                if not item.has_warn and item.get_valid_count() > self.config.intrude_valid_count:
                    logger.info(f"{self.pname} obj_id: {obj_id} 入侵异常")
                    # 全图带bbox
                    shot_img = ImgKit.draw_img_box(frame, ltrb)
                    screen_x = int((ltrb[0] + ltrb[2]) * 0.5)
                    screen_y = int((ltrb[1] + ltrb[3]) * 0.5)
                    cv2.circle(shot_img, (screen_x, screen_y), 4, (118, 154, 242), 2)
                    # 画警戒线
                    for i, point in enumerate(self.zone_points):
                        if i == 0:
                            continue
                        cv2.line(shot_img, (
                            int(self.zone_points[i][0] * self.stream_width),
                            int(self.zone_points[i][1] * self.stream_height)),
                                 (int(self.zone_points[i - 1][0] * self.stream_width),
                                  int(self.zone_points[i - 1][1] * self.stream_height)),
                                 (0, 255, 255), 2)  # 绘制线条
                    item.has_warn = True
                    # 发送报警信息给后端
                    WarnProxy.send(self.http_helper, self.pname, self.output_dir[0], self.cam_id, 4, item.per_id,
                                   shot_img, 1, self.config.stream_export_img_enable, self.config.stream_web_enable)

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

    def release_unused(self):
        """
        清空长期未更新点
        :return:
        """
        clear_keys = []
        for key, item in self.data_dict.items():
            if self.frame_id_cache[0] - item.last_update_id > self.config.intrude_lost_frame:
                clear_keys.append(key)
        clear_keys.reverse()  # 从尾巴往前删除，确保索引正确性
        for key in clear_keys:
            self.pool.push(self.data_dict[key])
            if self.face_helper is not None:
                self.face_helper.destroy_obj(key)
            self.data_dict.pop(key)  # 从字典中移除item

    def _is_in_warn(self, ltrb) -> bool:
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

    def on_draw_vis(self, idx, frame, input_mot):
        text_scale = 2
        text_thickness = 2
        line_thickness = 3
        # 标题线
        num = 0 if input_mot is None else input_mot.shape[0]
        cv2.putText(frame, 'inference_fps:%.2f num:%d' %
                    (1. / max(1e-5, self.update_timer.average_time),
                     num), (0, int(15 * text_scale)),
                    cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=text_thickness)
        # 警戒线
        for i, point in enumerate(self.zone_points):
            if i == 0:
                continue
            cv2.line(frame, (
            int(self.zone_points[i][0] * self.stream_width), int(self.zone_points[i][1] * self.stream_height)),
                     (int(self.zone_points[i - 1][0] * self.stream_width),
                      int(self.zone_points[i - 1][1] * self.stream_height)),
                     (0, 0, 255), line_thickness)  # 绘制线条

        # 对象基准点、包围盒
        if input_mot is not None:
            for obj in input_mot:
                cls = obj[5]
                if cls == 0:
                    ltrb = obj[:4]
                    obj_id = int(obj[6])
                    obj_color = self._get_color(obj_id)
                    screen_x = int((ltrb[0] + ltrb[2]) * 0.5)
                    screen_y = int((ltrb[1] + ltrb[3]) * 0.5)
                    cv2.circle(frame, (screen_x, screen_y), 4, (118, 154, 242), line_thickness)
                    cv2.rectangle(frame, pt1=(int(ltrb[0]), int(ltrb[1])), pt2=(int(ltrb[2]), int(ltrb[3])),
                                  color=obj_color, thickness=line_thickness)
                    cv2.putText(frame, f"{obj_id}",
                                (int(ltrb[0]), int(ltrb[1])),
                                cv2.FONT_HERSHEY_PLAIN, text_scale, obj_color, thickness=text_thickness)
                    if self.data_dict.__contains__(obj_id):
                        if self.data_dict[obj_id].has_warn:
                            cv2.putText(frame, "error",
                                        (int(ltrb[0] + 50), int(ltrb[1])),
                                        cv2.FONT_HERSHEY_PLAIN, text_scale, obj_color, thickness=text_thickness)
                        else:
                            cv2.putText(frame, "normal",
                                        (int(ltrb[0] + 50), int(ltrb[1])),
                                        cv2.FONT_HERSHEY_PLAIN, text_scale, obj_color, thickness=text_thickness)

        if self.config.intrude_face_enable:
            face_dict = self.face_helper.face_dict
            # 人脸参考线
            # y
            # point1 = (0, int(self.face_helper.config.face_cull_up_y * self.stream_height))
            point2 = (self.stream_width, int(self.face_helper.config.face_cull_up_y * self.stream_height))
            # point3 = (0, int((1 - self.face_helper.config.face_cull_down_y) * self.stream_height))
            point4 = (self.stream_width, int((1 - self.face_helper.config.face_cull_down_y) * self.stream_height))
            # x
            point5 = (int(self.face_helper.config.face_cull_left_x * self.stream_width), 0)
            # point6 = (int(self.face_helper.config.face_cull_left_x * self.stream_width), self.stream_height)
            point7 = (int((1 - self.face_helper.config.face_cull_right_x) * self.stream_width), 0)
            # point8 = (int((1 - self.face_helper.config.face_cull_right_x) * self.stream_width), self.stream_height)
            # 交线
            line_up1 = (point5[0], point2[1])
            line_up2 = (point7[0], point2[1])
            line_down1 = (point5[0], point4[1])
            line_down2 = (point7[0], point4[1])
            line_left1 = (point5[0], point2[1])
            line_left2 = (point5[0], point4[1])
            line_right1 = (point7[0], point2[1])
            line_right2 = (point7[0], point4[1])
            cv2.line(frame, line_up1, line_up2, (255, 255, 0), line_thickness)  # 绘制线条
            cv2.line(frame, line_down1, line_down2, (255, 255, 0), line_thickness)  # 绘制线条
            cv2.line(frame, line_left1, line_left2, (255, 255, 0), line_thickness)  # 绘制线条
            cv2.line(frame, line_right1, line_right2, (255, 255, 0), line_thickness)  # 绘制线条
            # 人脸识别结果
            for key, value in face_dict.items():
                if self.data_dict.__contains__(key):
                    ltrb = self.data_dict[key].ltrb
                    obj_id = self.data_dict[key].obj_id
                    obj_color = self._get_color(obj_id)
                    cv2.putText(frame, f"per_id:{face_dict[key]['per_id']}",
                                (int((ltrb[0] + ltrb[2]) / 2), int(self.data_dict[key].ltrb[1] + 20)),
                                cv2.FONT_HERSHEY_PLAIN, text_scale, obj_color, thickness=text_thickness)
        # 可视化并返回
        return frame

    def _get_color(self, idx):
        idx = (1 + idx) * 3
        color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
        return color


def create_process(shared_memory, config_path: str):
    comp: IntrudeComponent = IntrudeComponent(shared_memory, config_path)  # 创建组件
    try:
        comp.start()  # 初始化
        # 初始化结束通知
        shared_memory[GlobalKey.LAUNCH_COUNTER.name] += 1
        while not shared_memory[GlobalKey.ALL_READY.name]:
            time.sleep(0.1)
        comp.update()  # 算法逻辑循环
    except KeyboardInterrupt:
        comp.on_destroy()
    except Exception as e:
        # 使用 traceback 打印堆栈信息
        logger.error(f"IntrudeComponent: {e}")
        logger.error(traceback.format_exc())  # 打印完整的堆栈信息
        comp.on_destroy()
