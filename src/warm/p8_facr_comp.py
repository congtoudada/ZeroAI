
import time
import traceback
from typing import Dict

import cv2
from loguru import logger

from bytetrack.zero.bytetrack_helper import BytetrackHelper
from insight.zero.component.face_helper import FaceHelper
from src.warm.p8_facr_info import P8FacrInfo
from utility.config_kit import ConfigKit
from zero.core.based_stream_comp import BasedStreamComponent
from zero.info.based_stream_info import BasedStreamInfo
from zero.key.detection_key import DetectionKey
from zero.key.global_key import GlobalKey
from zero.key.stream_key import StreamKey


class P8FacrComponent(BasedStreamComponent):

    def __init__(self, shared_memory, config_path: str):
        super().__init__(shared_memory)
        self.config: P8FacrInfo = P8FacrInfo(ConfigKit.load(config_path))
        self.tracker: BytetrackHelper = BytetrackHelper(self.config.stream_mot_config)  # 追踪器
        self.face_helper: FaceHelper = None
        self.item_dict: Dict[int, dict] = {}  # key: obj_id {"per_id": per_id "score": score, "ltrb": ltrb}
        self.stream_width = 1920
        self.stream_height = 1080

    def on_start(self):
        super().on_start()
        # cam_id = self.read_dict[0][StreamKey.STREAM_CAM_ID.name]
        self.stream_width = int(self.read_dict[0][StreamKey.STREAM_WIDTH.name])
        self.stream_height = int(self.read_dict[0][StreamKey.STREAM_HEIGHT.name])
        self.face_helper = FaceHelper(self.config.facr_config, self._face_callback)

    def _face_callback(self, obj_id, per_id, score):
        if self.item_dict.__contains__(obj_id):
            if per_id != 1:
                self.item_dict[obj_id]["per_id"] = per_id
                self.item_dict[obj_id]["score"] = score
                print(f"识别结果: {per_id} | {score}")

    def on_get_stream(self, read_idx):
        frame, _ = super().on_get_stream(read_idx)  # 解析视频帧id+视频帧
        if frame is None:  # 没有有效帧
            return frame, None
        # 解析额外数据
        stream_package = self.read_dict[read_idx][self.config.input_ports[read_idx]]
        input_det = stream_package[DetectionKey.DET_PACKAGE_RESULT.name]  # 目标检测结果
        return frame, input_det

    def on_handle_stream(self, idx, frame, input_det) -> object:
        frame_id = self.frame_id_cache[idx]
        logger.info(f"frame_id: {frame_id}")
        # if input_det is not None:
        #     for item in input_det:
        #         logger.info(f"ltrb:{item[:4]} cls:{item[5]}")
        mot_result = None
        if input_det is not None:
            input_det = input_det[input_det[:, 5] == 0]  # 保留人
            mot_result = self.tracker.inference(input_det)  # (n,7)
            if mot_result is None:
                return
            for item in mot_result:
                # logger.info(f"obj_id: {item[6]} ltrb:{item[:4]} cls:{item[5]}")
                obj_id = item[6]
                ltrb = item[:4]
                if not self.item_dict.__contains__(obj_id):
                    self.item_dict[obj_id] = {}
                    self.item_dict[obj_id]["per_id"] = 1
                    self.item_dict[obj_id]["score"] = 0
                self.item_dict[obj_id]["ltrb"] = ltrb
        return mot_result

    def on_update(self):
        super().on_update()
        # 人脸识别请求
        current_id = self.frame_id_cache[0]
        for obj_id, item in self.item_dict.items():
            ltrb = item["ltrb"]
            # ltrb[3] = ltrb[3] - (ltrb[3] - ltrb[1]) * 0.5
            base_y = ((ltrb[1] + ltrb[3]) / 2) / self.stream_height
            self.face_helper.try_send(current_id, self.frames[0], ltrb, obj_id, base_y)
        # 人脸识别帮助tick，用于接受响应
        self.face_helper.tick(current_id)

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

    def on_draw_vis(self, idx, frame, input_mot):
        text_scale = 1
        text_thickness = 1
        line_thickness = 2
        # 参考线
        point1 = (0, int(self.face_helper.config.face_cull_up_y * self.stream_height))
        point2 = (self.stream_width, int(self.face_helper.config.face_cull_up_y * self.stream_height))
        point3 = (0, int((1 - self.face_helper.config.face_cull_down_y) * self.stream_height))
        point4 = (self.stream_width, int((1 - self.face_helper.config.face_cull_down_y) * self.stream_height))
        cv2.line(frame, point1, point2, (127, 127, 127), 1)  # 绘制线条
        cv2.line(frame, point3, point4, (127, 127, 127), 1)  # 绘制线条
        # 人
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
                                cv2.FONT_HERSHEY_PLAIN, 1, obj_color, thickness=text_thickness)
        # 人脸识别结果
        for key, value in self.item_dict.items():
            ltrb = value["ltrb"]
            per_id = value["per_id"]
            cv2.putText(frame, f"{per_id}",
                        (int((ltrb[0] + ltrb[2]) / 2), int(ltrb[1])),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), thickness=1)
        # 可视化并返回
        return frame

    def _get_color(self, idx):
        idx = (1 + idx) * 3
        color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
        return color

def create_process(shared_memory, config_path: str):
    comp = P8FacrComponent(shared_memory, config_path)  # 创建组件
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
        logger.error(f"P8FacrComponent: {e}")
        logger.error(traceback.format_exc())  # 打印完整的堆栈信息
        comp.on_destroy()
