import os
import time
import traceback
from typing import Dict, List
import cv2
import numpy as np
from loguru import logger

from src.utility.match_kit import DetectionRecord, MatchKit
from src.helmet.helmet_info import HelmetInfo
from src.helmet.helmet_item import HelmetItem
from bytetrack.zero.bytetrack_helper import BytetrackHelper
from simple_http.simple_http_helper import SimpleHttpHelper
from src.utility.warn_proxy import WarnProxy
from zero.core.based_stream_comp import BasedStreamComponent
from zero.key.detection_key import DetectionKey
from zero.key.global_key import GlobalKey
from zero.key.stream_key import StreamKey
from utility.config_kit import ConfigKit
from utility.img_kit import ImgKit
from utility.object_pool import ObjectPool


class HelmetComponent(BasedStreamComponent):
    """
    规范佩戴安全帽检测组件
    """

    def __init__(self, shared_memory, config_path: str):
        super().__init__(shared_memory)
        self.config: HelmetInfo = HelmetInfo(ConfigKit.load(config_path))
        self.pname = f"[ {os.getpid()}:helmet for {self.config.input_ports[0]}&{self.config.input_ports[1]} ]"
        self.cam_id = 0
        self.stream_width = 0
        self.stream_height = 0
        # key: obj_id value: cls
        self.pool: ObjectPool = ObjectPool(20, HelmetItem)
        self.record_pool: ObjectPool = ObjectPool(40, DetectionRecord)
        self.data_dict: Dict[int, HelmetItem] = {}
        self.tracker: BytetrackHelper = BytetrackHelper(self.config.stream_mot_config)  # 追踪器
        self.helmet_records: List[DetectionRecord] = []  # 安全帽目标检测结果（每帧更新）
        self.current_mot = None  # 当前帧人的追踪结果，如果非None，则最后要消耗掉检测结果
        self.http_helper = SimpleHttpHelper(self.config.stream_http_config)  # http帮助类

    def on_start(self):
        super().on_start()
        self.cam_id = self.read_dict[0][StreamKey.STREAM_CAM_ID.name]
        self.stream_width = self.read_dict[0][StreamKey.STREAM_WIDTH.name]
        self.stream_height = self.read_dict[0][StreamKey.STREAM_HEIGHT.name]

    def on_update(self) -> bool:
        self.release_unused()  # 清理无用资源（一定要在最前面调用）
        super().on_update()
        # 后处理，如果存在人，则清空当前帧的检测结果
        if self.current_mot is not None:
            self.helmet_records.clear()
            self.current_mot = None
        return True

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
        :param idx: 从input_ports[idx]取package
        :param frame: 帧
        :param input_det: 目标检测结果
        :return:
        """
        if input_det is None:
            return None

        if idx == 0:  # 安全帽
            for i in range(len(self.helmet_records)):
                self.record_pool.push(self.helmet_records[i])  # 归还到对象池
            self.helmet_records.clear()  # 有新的安全帽检测结果，清空旧安全帽检测记录
            for i, item in enumerate(input_det):
                ltrb = (item[0], item[1], item[2], item[3])
                score = item[4]
                cls = item[5]
                record = self.record_pool.pop()
                record.init(ltrb, score, cls)
                self.helmet_records.append(record)
            return None
        else:  # 人（此时安全帽记录已经填充）
            input_det = input_det[input_det[:, 5] == 0]
            mot_result = self.tracker.inference(input_det)  # 返回对齐输出后的mot结果
            self.current_mot = mot_result
            # 根据mot结果进行安全帽检测
            self._helmet_core(frame, mot_result, self.frame_id_cache[0], frame.shape[1], frame.shape[0])
            return mot_result

    def _helmet_core(self, frame, input_mot, current_frame_id, width, height) -> bool:
        # mot output: [n, 7]
        # n: n个对象
        # [0,1,2,3]: ltrb bboxes (基于视频流分辨率)
        #   [0]: x1
        #   [1]: y1
        #   [2]: x2
        #   [3]: y2
        # [4]: 置信度
        # [5]: 类别 (下标从0开始)
        # [6]: id
        if input_mot is None:
            return
        # 时间换精度: 根据t排序（y轴排序），配合包围盒匹配，可以使精度更高
        if self.config.helmet_y_sort:
            sort_indices = np.argsort(input_mot[:, 1])
            input_mot = input_mot[sort_indices]
            self.helmet_records.sort(key=lambda x: x.ltrb[1])
        # 构造person records
        person_records = []
        for i, obj in enumerate(input_mot):
            ltrb = obj[:4]
            score = obj[4]
            cls = obj[5]
            obj_id = int(obj[6])
            # 只有在检测区域内才匹配
            if not self._is_in_zone(ltrb, self.config.helmet_zone):
                continue
            # 更新帧id，避免被回收
            if self.data_dict.__contains__(obj_id):
                self.data_dict[obj_id].last_update_id = current_frame_id
            else:
                item = self.pool.pop()
                item.init(obj_id, 1, current_frame_id)  # 1:正常
                self.data_dict[obj_id] = item
            record: DetectionRecord = self.record_pool.pop()
            record.init(ltrb, score, cls, obj_id)
            person_records.append(record)

        # 使用包围盒里外匹配，先遍历安全帽，再遍历人
        helmet_results, person_results = MatchKit.match_bboxes(self.helmet_records, person_records)

        # 遍历结果集 (person和helmet一一对应)
        for i in range(len(person_results)):
            # 拿到结果记录索引
            per_idx = person_results[i]
            helmet_idx = helmet_results[i]
            # 拿到结果记录
            person = person_records[per_idx]
            helmet = self.helmet_records[helmet_idx]
            # 更新结果
            item = self.data_dict[person.obj_id]
            item.cls_update(helmet.cls)  # 安全帽类别更新
            # 结果收集
            self.process_result(frame, item, person.ltrb)
        # 结束！
        person_records.clear()

    def process_result(self, frame, helmet_item: HelmetItem, ltrb):
        # 没有报过警且异常状态保持一段时间才发送
        if not helmet_item.has_warn and helmet_item.get_valid_count() > self.config.helmet_valid_count:
            logger.info(f"安全帽佩戴异常: obj_id:{helmet_item.obj_id} cls:{helmet_item.cls}")
            helmet_item.has_warn = True  # 一旦视为异常，则一直为异常，避免一个人重复报警
            shot_img = ImgKit.draw_img_box(frame, ltrb)
            # 发送报警信息给后端
            WarnProxy.send(self.http_helper, self.pname, self.output_dir[0], self.cam_id, 2, 1,
                           shot_img, 1, self.config.stream_export_img_enable, self.config.stream_web_enable)

    def release_unused(self):
        # 清空长期未更新点
        clear_keys = []
        for key, item in self.data_dict.items():
            if self.frame_id_cache[0] - item.last_update_id > self.config.helmet_lost_frame:
                clear_keys.append(key)
        clear_keys.reverse()
        for key in clear_keys:
            self.pool.push(self.data_dict[key])
            self.data_dict.pop(key)  # 从字典中移除item

    def on_draw_vis(self, idx, frame, input_mot):
        if input_mot is None:  # 检测安全帽的端口，不显示任何内容
            return None
        text_scale = 1
        text_thickness = 1
        line_thickness = 2
        # 标题线
        num = 0 if input_mot is None else input_mot.shape[0]
        cv2.putText(frame, 'inference_fps:%.2f num:%d' %
                    (1. / max(1e-5, self.update_timer.average_time),
                     num), (0, int(15 * text_scale)),
                    cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=text_thickness)
        # 安全帽区域
        if len(self.config.helmet_zone) > 0:
            helmet_zone = self.config.helmet_zone
            cv2.rectangle(frame, pt1=(int(helmet_zone[0] * self.stream_width), int(helmet_zone[1] * self.stream_height)),
                          pt2=(int(helmet_zone[2] * self.stream_width), int(helmet_zone[3] * self.stream_height)),
                          color=(0, 255, 0), thickness=line_thickness)
        # 对象基准点、包围盒
        if len(self.config.detection_labels) == 0:
            logger.warning(f"{self.pname} detection_labels的长度为0，请在配置文件中配置detection_labels!")
            return frame
        if input_mot is not None:
            for obj in input_mot:
                ltrb = obj[:4]
                obj_id = int(obj[6])
                # screen_x = int((ltrb[0] + ltrb[2]) / 2)
                # screen_y = int((ltrb[1] + ltrb[3]) / 2)
                cv2.circle(frame, (int(ltrb[0]), int(ltrb[1])), 4, (118, 154, 242), line_thickness)
                cv2.rectangle(frame, pt1=(int(ltrb[0]), int(ltrb[1])), pt2=(int(ltrb[2]), int(ltrb[3])),
                              color=self._get_color(obj_id), thickness=line_thickness)
                if self.data_dict.__contains__(obj_id):
                    cls = int(self.data_dict[obj_id].cls)
                    is_warn = self.data_dict[obj_id].has_warn
                    cv2.putText(frame, f"{obj_id}:{self.config.detection_labels[cls]} warn:{is_warn}",
                                (int(ltrb[0]), int(ltrb[1])),
                                cv2.FONT_HERSHEY_PLAIN, text_scale, self._get_color(obj_id), thickness=text_thickness)
                else:
                    cv2.putText(frame, f"{obj_id}",
                                (int(ltrb[0]), int(ltrb[1])),
                                cv2.FONT_HERSHEY_PLAIN, text_scale, self._get_color(obj_id), thickness=text_thickness)
        # 安全帽
        for i, item in enumerate(self.helmet_records):
            ltrb = item.ltrb
            cls = int(item.cls)
            score = item.score
            if cls == 2:  # 只绘制未佩戴的
                cv2.rectangle(frame, pt1=(int(ltrb[0]), int(ltrb[1])), pt2=(int(ltrb[2]), int(ltrb[3])),
                              color=(0, 0, 255), thickness=line_thickness)
            # id_text = f"cls:{self.config.detection_labels[cls]}({score:.2f})"
            # cv2.putText(frame, id_text, (int(ltrb[0]), int(ltrb[1])), cv2.FONT_HERSHEY_PLAIN,
            #             text_scale, (0, 0, 255), thickness=text_thickness)
        # 可视化并返回
        return frame

    def _get_color(self, idx):
        idx = (1 + idx) * 3
        color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
        return color

    def _is_in_zone(self, person_ltrb, helmet_ltrb):
        if len(helmet_ltrb) == 0:
            return True
        # base_x = ((person_ltrb[0] + person_ltrb[2]) / 2) / self.stream_width
        # base_y = ((person_ltrb[1] + person_ltrb[3]) / 2) / self.stream_height
        base_x = person_ltrb[0] / self.stream_width
        base_y = person_ltrb[1] / self.stream_height
        if helmet_ltrb[0] < base_x < helmet_ltrb[2] and helmet_ltrb[1] < base_y < helmet_ltrb[3]:
            return True
        else:
            return False


def create_process(shared_memory, config_path: str):
    comp: HelmetComponent = HelmetComponent(shared_memory, config_path)  # 创建组件
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
        logger.error(f"HelmetComponent: {e}")
        logger.error(traceback.format_exc())  # 打印完整的堆栈信息
        comp.on_destroy()
