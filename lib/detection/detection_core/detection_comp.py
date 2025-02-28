import os
import sys
from abc import ABC
import cv2

from detection_core.detection_info import DetectionInfo
from detection_core.i_detection_wrapper import IDetectionWrapper
from reid_core.helper.reid_helper import ReidHelper
from utility.img_kit import ImgKit
from zero.core.based_stream_comp import BasedStreamComponent
from zero.key.detection_key import DetectionKey
from zero.key.stream_key import StreamKey


class DetectionComponent(BasedStreamComponent, ABC):
    def __init__(self, shared_memory, config_path: str):
        super().__init__(shared_memory)
        self.pid = os.getpid()
        self.pname = f"[ {self.pid}:detection ]"
        self.cam_ids = []  # 输入端口对应的cam_id
        self.last_reid_time = []  # 每个输入端口last_frame_id
        self.save_id = 0
        # 由子类on_start加载
        self.config: DetectionInfo = None  # config
        self.predictor: IDetectionWrapper = None  # 推理模型
        self.conf = 0.5

    def on_start(self):
        """
        初始化时调用一次
        :return:
        """
        super().on_start()
        self.conf = self.config.detection_reid_conf  # reid存图置信度阈值
        for i, output_port in enumerate(self.config.output_ports):
            self.write_dict[i][output_port] = None  # yolox package
        if self.config.detection_reid_enable:
            for i, input_port in enumerate(self.config.input_ports):
                cam_id = self.read_dict[i][StreamKey.STREAM_CAM_ID.name]
                self.cam_ids.append(cam_id)
                self.last_reid_time.append(0)

    def on_handle_stream(self, idx, frame, user_data):
        """
        # output shape: [n, 6]
        # n: n个对象
        # [0,1,2,3]: ltrb bboxes (基于视频流分辨率)
        #   [0]: x1
        #   [1]: y1
        #   [2]: x2
        #   [3]: y2
        # [4]: 置信度
        # [5]: 类别 (下标从0开始)
        :param user_data:
        :param idx:
        :param frame:
        :return:
        """
        # 不存在新帧，直接返回
        if frame is None:
            return None
        # 推理获得 ndarray (n, 6)
        result = self.predictor.inference(frame)  # tensor(n,6)
        # 填充输出
        frame_id = self.frame_id_cache[idx]
        package = {StreamKey.STREAM_PACKAGE_FRAME_ID.name: frame_id,
                   StreamKey.STREAM_PACKAGE_FRAME.name: frame,
                   DetectionKey.DET_PACKAGE_RESULT.name: result}
        self.write_dict[idx][self.config.output_ports[idx]] = package  # 填充输出(result为None代表无目标)
        # reid存图
        if self.config.detection_reid_enable and result is not None:
            if frame_id - self.last_reid_time[idx] > self.config.detection_reid_interval:
                for obj in result:
                    cls = obj[5]
                    # 不是类别0不存 (0一般是person)
                    if cls != 0:
                        continue
                    score = obj[4]
                    # 置信度低的不存
                    if score < self.conf:
                        continue
                    ltrb = obj[:4]
                    # 包围盒太小的不存或比例奇怪的不存
                    w = ltrb[2] - ltrb[0]
                    h = ltrb[3] - ltrb[1]
                    # if w < 25 or h < 50 or (h*1.0 / w*1.0) < 1.0:
                    if w < 50 or h < 100:
                        continue
                    shot_img = ImgKit.crop_img(frame, ltrb)
                    if shot_img is not None:
                        self.save_id = (self.save_id + 1) % sys.maxsize
                        ReidHelper.send_save_timing(self.cam_ids[idx], self.pid, self.save_id, shot_img)
                self.last_reid_time[idx] = frame_id
        return result

    def on_draw_vis(self, idx, frame, process_data):
        """
        可视化函数
        :param idx:
        :param frame:
        :param data: tensor(n,6)
        :return:
        """
        if process_data is None:
            return frame
        text_scale = 2
        text_thickness = 2
        line_thickness = 3
        cv2.putText(frame, 'inference_fps:%.2f num:%d' %
                    (1. / max(1e-5, self.update_timer.average_time),
                     process_data.shape[0]), (0, int(15 * text_scale)),
                    cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=text_thickness)
        # scale = min(self.config.yolox_args_tsize / frame.shape[0],
        #             self.config.yolox_args_tsize / frame.shape[1])
        for i in range(process_data.shape[0]):
            # tlbr = data[i, :4] / scale
            tlbr = process_data[i, :4]
            x1, y1, w, h = tlbr[0], tlbr[1], tlbr[2] - tlbr[0], tlbr[3] - tlbr[1]
            score = process_data[i, 4]
            cls = int(process_data[i, 5])
            if cls < len(self.config.detection_labels):
                id_text = f"{self.config.detection_labels[cls]}({score:.2f})"
            else:
                id_text = f"{score:.2f}"
            intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
            cv2.rectangle(frame, intbox[0:2], intbox[2:4],
                          color=self._get_color(cls),
                          thickness=line_thickness)
            cv2.putText(frame, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                        thickness=text_thickness)
        return frame

    def _get_color(self, idx):
        idx = (1 + idx) * 3
        color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
        return color

# def create_process(shared_memory, config_path: str):
#     comp = YoloxComponent(shared_memory, config_path)  # 创建组件
#     try:
#         comp.start()  # 初始化
#         # 初始化结束通知
#         shared_memory[GlobalKey.LAUNCH_COUNTER.name] += 1
#         while not shared_memory[GlobalKey.ALL_READY.name]:
#             time.sleep(0.1)
#         comp.update()  # 算法逻辑循环
#     except KeyboardInterrupt:
#         comp.on_destroy()
#     except Exception as e:
#         logger.error(f"YoloxComponent: {e}")
#         logger.error(traceback.format_exc())  # 打印完整的堆栈信息
#         comp.on_destroy()
