import os
import time
import traceback
import cv2
from typing import Dict
import numpy as np
from loguru import logger

from src.card.card_info import CardInfo
from src.card.card_item import CardItem
from bytetrack.zero.bytetrack_helper import BytetrackHelper
from simple_http.simple_http_helper import SimpleHttpHelper
from src.utility.warn_proxy import WarnProxy
from zero.core.based_stream_comp import BasedStreamComponent
from zero.key.detection_key import DetectionKey
from zero.key.global_key import GlobalKey
from zero.key.stream_key import StreamKey
from utility.config_kit import ConfigKit
from utility.object_pool import ObjectPool
from utility.timer_kit import TimerKit


class CardComponent(BasedStreamComponent):
    def __init__(self, shared_memory, config_path: str):
        super().__init__(shared_memory)
        self.config: CardInfo = CardInfo(ConfigKit.load(config_path))
        self.pname = f"[ {os.getpid()}:card for {self.config.input_ports[0]} ]"
        self.cam_id = 0
        self.stream_width = 0
        self.stream_height = 0
        # 定义变量
        self.pool: ObjectPool = ObjectPool(20, CardItem)
        self.item_dict: Dict[int, CardItem] = {}  # 空字典，用于存储对象的 ID 和对应的 CardItem
        self.card_dict: Dict[int, int] = {}  # 空字典，用于存储进入刷卡区的对象的 ID 和对应的 CardItem
        self.gate_dict: Dict[int, int] = {}  # 空字典，用于存储进入刷卡区的对象的 ID 和对应的 CardItem value为0啥也不做，-1代刷卡，1正常
        self.red_points = []
        self.red_vecs = []  # 初始化用于存储红色信号点和向量的列表
        self.green_points = []
        self.green_vecs = []  # 初始化用于存储绿色信号点和向量的列
        self.temp_red_result = []
        self.temp_green_result = []  # 初始化临时存储红色和绿色结果的列表
        self.timer = TimerKit()  # 用于计时
        self.valid = False  # False表示没有检测到代刷卡行为
        self.valid_count = self.config.card_warning_frame
        self.tracker: BytetrackHelper = BytetrackHelper(self.config.stream_mot_config)  # 追踪器
        self.http_helper = SimpleHttpHelper(self.config.stream_http_config)  # http帮助类

    def on_start(self):
        super().on_start()
        self.cam_id = self.read_dict[0][StreamKey.STREAM_CAM_ID.name]
        self.stream_width = int(self.read_dict[0][StreamKey.STREAM_WIDTH.name])
        self.stream_height = int(self.read_dict[0][StreamKey.STREAM_HEIGHT.name])
        # 预计算
        self.red_vecs.clear()
        self.green_vecs.clear()  # 清空存储红色和绿色向量的列表
        # 遍历配置中的红色信号点
        for i, point_str in enumerate(self.config.card_red):
            self.red_points.append(
                (float(point_str.split(',')[0]), float(point_str.split(',')[1])))  # 将红色信号点的坐标添加到 self.red_points 列表中
            if i != 0:  # 如果不是第一个点
                # 计算当前点与前一个点之间的向量
                red = np.array([self.red_points[i][0] - self.red_points[i - 1][0],
                                self.red_points[i][1] - self.red_points[i - 1][1],
                                0])
                self.red_vecs.append(red / np.linalg.norm(red))  # 将归一化后的向量添加到 self.red_vecs 列表中
        # 遍历配置中的绿色信号点
        for i, point_str in enumerate(self.config.card_green):
            self.green_points.append((float(point_str.split(',')[0]), float(point_str.split(',')[1])))
            if i != 0:
                green = np.array([self.green_points[i][0] - self.green_points[i - 1][0],
                                  self.green_points[i][1] - self.green_points[i - 1][1],
                                  0])
                self.green_vecs.append(green / np.linalg.norm(green))

    def on_update(self):
        self.release_unused()  # 清理无用资源（一定要在最前面调用）
        super().on_update()

    def release_unused(self):
        # 清空长期未更新点
        item_clear_keys = []
        card_clear_keys = []
        for key, item in self.item_dict.items():
            #  检查当前帧序号与上次更新帧序号之间的差值是否大于配置中指定的丢失帧数阈值
            # card_lost_frames丢失帧数阈值
            if self.frame_id_cache[0] - item.last_update_id > self.config.card_item_lost_frames:
                item_clear_keys.append(key)
        item_clear_keys.reverse()  # 从尾巴往前删除，确保索引正确性
        # 遍历需要清除的键列表
        for key in item_clear_keys:
            self.pool.push(self.item_dict[key])  # 放回对象池
            self.item_dict.pop(key)  # 从字典中移除
            self.on_destroy_obj(key)
        # 清除card区长期未更新的值
        for key, value in self.card_dict.items():
            #  检查当前帧序号与上次更新帧序号之间的差值是否大于配置中指定的丢失帧数阈值
            if self.frame_id_cache[0] - value > self.config.card_lost_frames:  # card_lost_frames丢失帧数阈值
                card_clear_keys.append(key)
        card_clear_keys.reverse()  # 从尾巴往前删除，确保索引正确性
        # 遍历需要清除的键列表
        for key in card_clear_keys:
            self.card_dict.pop(key)  # 从字典中移除

    def on_get_stream(self, read_idx):
        frame, _ = super().on_get_stream(read_idx)  # 解析视频帧id+视频帧
        if frame is None:  # 没有有效帧
            return frame, None
        # 解析额外数据
        stream_package = self.read_dict[read_idx][self.config.input_ports[read_idx]]
        input_det = stream_package[DetectionKey.DET_PACKAGE_RESULT.name]  # 目标检测结果
        return frame, input_det

    def on_handle_stream(self, idx, frame, input_det):
        """
        处理视频流
        :param idx: 固定为0（只从input_ports[0]取数据）
        :param frame: 帧
        :param input_det: 目标检测结果
        :return:
        """
        if input_det is None:
            return None

        input_det = input_det[input_det[:, 5] == 0]
        frame_id = self.frame_id_cache[0]
        mot_result = self.tracker.inference(input_det, frame_id, frame, self.cam_id)  # 返回对齐输出后的mot结果
        # 根据mot结果进行计数
        self._card_core(frame, mot_result, frame_id, frame.shape[1], frame.shape[0])
        return mot_result

    def _card_core(self, frame, input_mot, current_frame_id, width, height):
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
        if input_mot is None:
            return
        for item in self.item_dict.values():
            item.reset_update()
        for obj in input_mot:
            cls = int(obj[5])  # 提取当前目标的类别，转换为整数类型
            if cls == 0:  # 人
                ltrb = obj[:4]  # 提取当前目标的边界框坐标，即左上角和右下角的坐标
                conf = obj[4]  # 提取当前目标的置信度
                obj_id = int(obj[6])  # 提取当前目标的唯一标识符，转换为整数类型
                # 1.如果没有则添加
                if not self.item_dict.__contains__(obj_id):  # 检查当前目标是否在 item_dict 中
                    self.gate_dict[obj_id] = 0  # 加入item_dict时为初始状态0
                    item: CardItem = self.pool.pop()  # 如果当前目标不在 item_dict 中，则从对象池中取出一个对象，这里假设为 CountItem 类型
                    item.init(obj_id,
                              self.config.card_item_valid_frames)  # 初始化对象 ，传入目标的唯一标识符和有效帧数,card_valid_frames对象稳定出现多少帧，才开始计算
                    self.item_dict[obj_id] = item  # 将初始化后的对象添加到 item_dict 字典中，以目标的唯一标识符为键
                    self.on_create_obj(item)  # 调用 on_create_obj 方法，处理新创建的对象
                # 2.更新状态
                x, y = self._get_base(self.config.card_item_base, ltrb)  # 调用 _get_base 方法获取目标的基准点坐标
                # 调用对象的 update 方法，更新目标的状态信息，包括当前帧序号、归一化后的坐标和边界框坐标
                self.item_dict[obj_id].update(current_frame_id, x / self.stream_width, y / self.stream_height, ltrb)
        # 收集结果
        self._process_result(frame)

    def _process_result(self, frame):
        """
        处理结果
        :return:
        """
        for key, item in self.item_dict.items():  # 遍历目标字典中的每个目标
            if not item.enable:  # 不是有效点，则跳过
                continue
            self.temp_red_result.clear()
            self.temp_green_result.clear()  # 清空临时红色和绿色结果列表
            x = item.base_x
            y = item.base_y
            base = [x, y]
            last_base = [item.last_x, item.last_y]
            if self.is_within_card_area(base):
                self.card_dict[item.obj_id] = item.last_update_id
            if self.is_through_gate_line(key, base, last_base):
                self.judge_valid(item.obj_id, frame, base)

    # 判断是否在刷卡区
    def is_within_card_area(self, base):
        # 在这里实现判断是否在刷卡点矩形区域内的逻辑
        # base 是目标的基点坐标，可以根据该坐标判断目标位置
        card_area = self.green_points
        return card_area[0][0] <= base[0] <= card_area[2][0] and card_area[0][1] <= base[1] <= card_area[2][1]

    # 判断是否通过了门
    def is_through_gate_line(self, key, base, last_base):
        # 不在合法运动区域内就剔除
        if not (self.config.card_valid_zone[0] < base[0] < self.config.card_valid_zone[2] and \
                self.config.card_valid_zone[1] < base[1] < self.config.card_valid_zone[3]):
            return False

        # 在这里实现判断是否通过大门线的逻辑
        if self.gate_dict[key] != 0:
            return False
        # base 是目标的基点坐标，可以根据该坐标判断目标位置
        gate_line = self.red_points
        # 这里使用简化的方法，假设大门线为水平线，目标的中心点y坐标大于等于大门线的y坐标即为通过了大门线
        if last_base[1] > gate_line[0][1] >= base[1]:
            return True
        else:
            return False

    # 对每个通过门的目标进行代刷卡行为的判断
    def judge_valid(self, key, frame, base):
        valid = key in self.card_dict
        if not valid:
            self.gate_dict[key] = 1
            self.valid = True  # 检测到代刷卡行为
            self.valid_count = self.config.card_warning_frame
            logger.info(f"{self.pname} {key} 代刷卡行为")  # 控制台打印
            screen_x = int(base[0] * self.stream_width)
            screen_y = int(base[1] * self.stream_height)
            cv2.circle(frame, (screen_x, screen_y), 12, (0, 0, 255), 3)
            # 发送报警信息给后端
            WarnProxy.send(self.http_helper, self.pname, self.output_dir[0], self.cam_id, 3, 1,
                           frame, 1, self.config.stream_export_img_enable, self.config.stream_web_enable)
        else:
            self.gate_dict[key] = 2
            # print(str(key)+"正常通过")   #控制台打印

    def on_destroy_obj(self, obj_id):
        pass

    def on_create_obj(self, obj):
        pass

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

    #  用于在图像上进行可视化操
    def on_draw_vis(self, idx, frame, input_mot):
        text_scale = 2
        text_thickness = 2
        line_thickness = 3
        # 标题线
        num = 0 if input_mot is None else input_mot.shape[0]
        # 在图像上添加文本信息，包括帧序号、视频帧率、推理帧率、目标数量以及进入和离开人数等信息
        cv2.putText(frame, 'inference_fps:%.2f num:%d' %
                    (1. / max(1e-5, self.timer.average_time),
                     num), (0, int(15 * text_scale)),
                    cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=text_thickness)
        # 红线
        for i, red_point in enumerate(self.red_points):
            if i == 0:
                continue
            cv2.line(frame,
                     (int(self.red_points[i][0] * self.stream_width), int(self.red_points[i][1] * self.stream_height)),
                     (int(self.red_points[i - 1][0] * self.stream_width),
                      int(self.red_points[i - 1][1] * self.stream_height)),
                     (0, 0, 255), line_thickness)  # 绘制线条
        # 绿线
        for i, green_point in enumerate(self.green_points):
            if i == 0:
                continue
            cv2.line(frame, (
            int(self.green_points[i][0] * self.stream_width), int(self.green_points[i][1] * self.stream_height)),
                     (int(self.green_points[i - 1][0] * self.stream_width),
                      int(self.green_points[i - 1][1] * self.stream_height)),
                     (255, 0, 0), line_thickness)  # 绘制线条
        # 对象基准点、红绿信息
        for item in self.item_dict.values():
            screen_x = int(item.base_x * self.stream_width)
            screen_y = int(item.base_y * self.stream_height)
            cv2.circle(frame, (screen_x, screen_y), 4, (118, 154, 242), line_thickness)
            # cv2.putText(frame, str(item.red_cur), (screen_x, screen_y), cv2.FONT_HERSHEY_PLAIN,
            #             text_scale, (0, 0, 255), thickness=text_thickness)
            # cv2.putText(frame, str(item.green_cur), (screen_x + 10, screen_y), cv2.FONT_HERSHEY_PLAIN,
            #             text_scale, (255, 0, 0), thickness=text_thickness)
        # 包围盒有效区域
        bbox_min = (self.config.card_valid_zone[0] * self.stream_width,
                    self.config.card_valid_zone[1] * self.stream_height)
        bbox_max = (self.config.card_valid_zone[2] * self.stream_width,
                    self.config.card_valid_zone[3] * self.stream_height)
        cv2.rectangle(frame, pt1=(int(bbox_min[0]), int(bbox_min[1])), pt2=(int(bbox_max[0]), int(bbox_max[1])),
                      color=(0, 255, 0), thickness=text_thickness)
        # 对象包围盒
        if input_mot is not None:
            for obj in input_mot:
                cls = int(obj[5])
                if cls == 0:
                    ltrb = obj[:4]
                    obj_id = int(obj[6])
                    text = ""
                    if self.gate_dict[obj_id] == 2:
                        text = "normal"
                    elif self.gate_dict[obj_id] == 1:
                        text = "error"
                    cv2.rectangle(frame, pt1=(int(ltrb[0]), int(ltrb[1])), pt2=(int(ltrb[2]), int(ltrb[3])),
                                  color=(0, 0, 255), thickness=text_thickness)
                    cv2.putText(frame, f"{obj_id}" + text,
                                (int(ltrb[0]), int(ltrb[1])),
                                cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=text_thickness)

        self.draw_warning(frame)
        # 可视化并返回
        return frame

    # 绘制警报信息
    def draw_warning(self, frame):
        if self.valid:
            # 设置字体和大小
            self.valid_count = self.valid_count - 1
            if self.valid_count == 0:
                self.valid = False
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.25
            font_color = (0, 0, 255)  # 红色
            line_type = 3

            # 计算文本位置
            text = 'WARNING: Anomaly Detected!'
            text_size = cv2.getTextSize(text, font, font_scale, line_type)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = (frame.shape[0] + text_size[1]) // 2

            # 在帧上绘制文本
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, line_type)


def create_process(shared_memory, config_path: str):
    comp: CardComponent = CardComponent(shared_memory, config_path)  # 创建组件
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
        logger.error(f"CardComponent: {e}")
        logger.error(traceback.format_exc())  # 打印完整的堆栈信息
        comp.on_destroy()
