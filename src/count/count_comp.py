import os
import time
import traceback
from typing import Dict
import cv2
import numpy as np
from loguru import logger

from src.count.count_item import CountItem
from src.count.count_info import CountInfo
from bytetrack.zero.bytetrack_helper import BytetrackHelper
from simple_http.simple_http_helper import SimpleHttpHelper
from zero.core.based_stream_comp import BasedStreamComponent
from zero.key.detection_key import DetectionKey
from zero.key.global_key import GlobalKey
from zero.key.stream_key import StreamKey
from utility.config_kit import ConfigKit
from utility.img_kit import ImgKit
from utility.object_pool import ObjectPool


class CountComponent(BasedStreamComponent):
    """
    计数组件
    """
    def __init__(self, shared_memory, config_path: str):
        super().__init__(shared_memory)
        self.config: CountInfo = CountInfo(ConfigKit.load(config_path))
        self.pname = f"[ {os.getpid()}:count for {self.config.input_ports[0]}]"
        self.cam_id = 0
        self.stream_width = 0
        self.stream_height = 0
        # 自身定义
        self.in_count = 0  # 进入人数
        self.out_count = 0  # 离开人数
        self.pool: ObjectPool = ObjectPool(20, CountItem)  # 对象池
        self.item_dict: Dict[int, CountItem] = {}  # 当前检测对象字典
        self.red_points = []  # 预计算红色点位置集合
        self.red_vecs = []  # 预计算红色向量集合
        self.green_points = []  # 预计算绿色点位置集合
        self.green_vecs = []  # 预计算绿色向量集合
        self.tracker: BytetrackHelper = BytetrackHelper(self.config.stream_mot_config)  # 追踪器
        self.http_helper = SimpleHttpHelper(self.config.stream_http_config)  # http帮助类

    def on_start(self):
        super().on_start()
        self.cam_id = self.read_dict[0][StreamKey.STREAM_CAM_ID.name]
        self.stream_width = int(self.read_dict[0][StreamKey.STREAM_WIDTH.name])
        self.stream_height = int(self.read_dict[0][StreamKey.STREAM_HEIGHT.name])
        # 预计算
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

    def on_update(self):
        self.release_unused()  # 清理无用资源（一定要在最前面调用）
        super().on_update()

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
        mot_result = self.tracker.inference(input_det)  # 返回对齐输出后的mot结果
        # 根据mot结果进行计数
        self._count_core(frame, mot_result, self.frame_id_cache[0], frame.shape[1], frame.shape[0])
        return mot_result

    def _count_core(self, frame, input_mot, current_frame_id, width, height):
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
        # 清空前一帧状态
        for item in self.item_dict.values():
            item.reset_update()
        # 更新状态
        for obj in input_mot:
            cls = int(obj[5])
            if cls == 0:
                ltrb = obj[:4]
                conf = obj[4]
                obj_id = int(obj[6])
                # 1.如果没有则添加
                if not self.item_dict.__contains__(obj_id):
                    item: CountItem = self.pool.pop()
                    item.init(obj_id, self.config.count_valid_frames)  # 初始化对象
                    self.item_dict[obj_id] = item
                    self.on_create_obj(item)
                # 2.更新状态
                x, y = self._get_base(self.config.count_base, ltrb)
                self.item_dict[obj_id].update(current_frame_id, x / width, y / height, ltrb)
        # 收集结果
        self._process_result(frame)

    def _process_result(self, frame):
        """
        处理结果
        :return:
        """
        for item in self.item_dict.values():
            if not item.enable:  # 不是有效点，则跳过
                continue
            epsilon = 1e-3
            temp_red_result = []
            temp_green_result = []
            # 收集红绿信号（线上为0，线下为1，无效-1）
            for i in range(len(self.red_points) - 1):  # 最后一个点不计算
                vec3d = (item.base_x - self.red_points[i][0], item.base_y - self.red_points[i][1], 0)
                vec3d_length = np.linalg.norm(vec3d)
                if abs(vec3d_length) > epsilon:
                    # vec3d = vec3d / vec3d_length
                    dot_ret = np.dot(self.red_vecs[i], vec3d)
                    if dot_ret > 0:
                        cross_ret = np.cross(self.red_vecs[i], vec3d)[2]  # (n, 1) n为red_vec
                        temp_red_result.append(cross_ret)
            for i in range(len(self.green_points) - 1):  # 最后一个点不计算
                vec3d = (item.base_x - self.green_points[i][0], item.base_y - self.green_points[i][1], 0)
                vec3d_length = np.linalg.norm(vec3d)
                if abs(vec3d_length) > epsilon:
                    # vec3d = vec3d / vec3d_length
                    dot_ret = np.dot(self.green_vecs[i], vec3d)
                    if dot_ret > 0:
                        cross_ret = np.cross(self.green_vecs[i], vec3d)[2]
                        temp_green_result.append(cross_ret)
            item.update_red(self._process_cross(temp_red_result))
            item.update_green(self._process_cross(temp_green_result))
            # 处理红绿信号
            self._resolve_per_result(frame, item)

    def _resolve_per_result(self, frame, item: CountItem):
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
        # logger.info(f"{self.pname} {item.obj_id} flags: {item.red_cur} - {item.green_cur}")
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
            if len(item.red_seq) == 2 and item.red_seq == item.green_seq:
                ret = self._get_dir(item.red_seq[0] == 0, self.config.count_reverse)
                if ret:
                    self.in_count += 1
                    self.send_result(frame, 1, item)
                else:
                    self.out_count += 1
                    self.send_result(frame, 2, item)
                # 重置计数器
                item.red_seq.pop(0)
                item.green_seq.pop(0)

    def send_result(self, frame, status: int, item: CountItem):
        """
        结果通知
        :param frame:
        :param status: 1进2出
        :param item:
        :return:
        """
        # 导出图
        time_str = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        status_str = "In" if status == 1 else "Out"
        img_path = os.path.join(self.output_dir[0], f"{time_str}_{status_str}.jpg")
        img_shot = ImgKit.crop_img(frame, item.ltrb)

        if img_shot.size == 0:
            logger.error(f"{self.pname} 警告: 裁剪的图像为空。")
            return

        if self.config.stream_export_img_enable:
            cv2.imwrite(img_path, img_shot)
            logger.info(f"{self.pname} 存图成功，路径: {img_path}")

        if self.config.stream_web_enable:
            # 通知后端
            data = {
                "recordTime": time_str,
                "camId": self.cam_id,
                "status": status,
                "shotImg": os.path.abspath(img_path)
            }
            # WebKit.post(f"{WebKit.Prefix_url}/count", data)
            self.http_helper.post("/algorithm/count", data)

    def on_destroy_obj(self, obj_id):
        pass

    def on_create_obj(self, obj):
        pass

    def release_unused(self):
        # 清空长期未更新点
        clear_keys = []
        for key, item in self.item_dict.items():
            if self.frame_id_cache[0] - item.last_update_id > self.config.count_lost_frames:
                clear_keys.append(key)
        clear_keys.reverse()  # 从尾巴往前删除，确保索引正确性
        for key in clear_keys:
            self.pool.push(self.item_dict[key])  # 放回对象池
            self.item_dict.pop(key)  # 从字典中移除
            self.on_destroy_obj(key)

    def on_draw_vis(self, idx, frame, input_mot):
        # 1920x1080
        # text_scale = 1
        # text_thickness = 1
        # line_thickness = 2
        text_scale = 2
        text_thickness = 2
        line_thickness = 3
        # 标题线
        num = 0 if input_mot is None else input_mot.shape[0]
        cv2.putText(frame, 'inference_fps:%.2f num:%d in:%d out:%d' %
                    (1. / max(1e-5, self.update_timer.average_time),
                     num, self.in_count, self.out_count), (0, int(15 * text_scale)),
                    cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=text_thickness)
        # 红线
        for i, red_point in enumerate(self.red_points):
            if i == 0:
                continue
            cv2.line(frame,
                     (int(self.red_points[i][0] * self.stream_width),
                      int(self.red_points[i][1] * self.stream_height)),
                     (int(self.red_points[i-1][0] * self.stream_width),
                      int(self.red_points[i-1][1] * self.stream_height)),
                     (0, 0, 255), line_thickness)  # 绘制线条
        # 绿线
        for i, green_point in enumerate(self.green_points):
            if i == 0:
                continue
            cv2.line(frame, (int(self.green_points[i][0] * self.stream_width),
                             int(self.green_points[i][1] * self.stream_height)),
                     (int(self.green_points[i-1][0] * self.stream_width),
                      int(self.green_points[i-1][1] * self.stream_height)),
                     (255, 0, 0), line_thickness)  # 绘制线条
        # 对象基准点、红绿信息
        for item in self.item_dict.values():
            screen_x = int(item.base_x * self.stream_width)
            screen_y = int(item.base_y * self.stream_height)
            cv2.circle(frame, (screen_x, screen_y), 4, (118, 154, 242), line_thickness)
            cv2.putText(frame, str(item.red_cur), (screen_x, screen_y), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                        thickness=text_thickness)
            cv2.putText(frame, str(item.green_cur), (screen_x + 10, screen_y), cv2.FONT_HERSHEY_PLAIN, text_scale, (255, 0, 0),
                        thickness=text_thickness)
        # 对象包围盒
        if input_mot is not None:
            for obj in input_mot:
                cls = obj[5]
                if cls == 0:
                    ltrb = obj[:4]
                    obj_id = int(obj[6])
                    cv2.rectangle(frame, pt1=(int(ltrb[0]), int(ltrb[1])), pt2=(int(ltrb[2]), int(ltrb[3])),
                                  color=self._get_color(obj_id), thickness=line_thickness)
                    cv2.putText(frame, f"{obj_id}",
                                (int(ltrb[0]), int(ltrb[1])),
                                cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=text_thickness)
        # 可视化并返回
        return frame

    def _get_color(self, idx):
        idx = (1 + idx) * 3
        color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
        return color

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
        epsilon = 1e-3
        for re in results:
            if abs(re) < epsilon:
                return -1
            if re > 0:
                return 1
        return 0


def create_process(shared_memory, config_path: str):
    comp: CountComponent = CountComponent(shared_memory, config_path)  # 创建组件
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
        logger.error(f"CountComponent: {e}")
        logger.error(traceback.format_exc())  # 打印完整的堆栈信息
        comp.on_destroy()


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
