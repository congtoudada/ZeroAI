import os
import cv2
import numpy as np
from loguru import logger
from conf.algorithm.detection.yolox.exps.yolox_s_head import Exp
from yolox.exp import get_exp
from yolox.zero.component.predictor import create_zero_predictor
from yolox.zero.info.yolox_info import YoloxInfo
from zero.core.component.base.base_det_comp import BaseDetComponent
from zero.core.component.feature.launcher_comp import LauncherComponent
from zero.core.key.shared_key import SharedKey
from zero.utility.config_kit import ConfigKit
from zero.utility.timer_kit import TimerKit


class YoloxComponent(BaseDetComponent):
    def __init__(self, shared_data, config_path: str):
        super().__init__(shared_data)
        self.config: YoloxInfo = YoloxInfo(ConfigKit.load(config_path))
        self.pname = f"[ {os.getpid()}:yolox for {self.config.yolox_args_expn}]"
        # 自身定义
        self.output_dir = ""  # 输出目录
        self.predictor = None  # 推理模型
        self.timer = TimerKit()  # 计时器
        self.scale = []  # 结果缩放尺寸

    def on_start(self):
        """
        初始化时调用一次
        :return:
        """
        super().on_start()
        exp: Exp = get_exp(self.config.yolox_args_exp_file, self.config.yolox_args_name)
        # 设置输出文件夹
        # folder = os.path.splitext(os.path.basename(self.shared_data[SharedKey.STREAM_URL]))[0]
        folder = "output/yolox"
        self.output_dir = os.path.join(exp.output_dir, self.config.yolox_args_expn, folder)
        os.makedirs(self.output_dir, exist_ok=True)
        # 创建zero框架版的yolox目标检测器
        self.predictor = create_zero_predictor(self.config, exp, self.pname)
        self.output_detect_info = {
            SharedKey.DETECTION_ID: 0,
            SharedKey.DETECTION_FRAME: None,
            SharedKey.DETECTION_OUTPUT: []
        }
        for i in range(len(self.config.STREAM_ORIGINAL_WIDTH)):
            height_key = self.config.STREAM_ORIGINAL_HEIGHT[i]
            width_key = self.config.STREAM_ORIGINAL_WIDTH[i]
            self.scale.append(min(self.config.yolox_args_tsize / float(self.shared_data[height_key]),
                                  self.config.yolox_args_tsize / float(self.shared_data[width_key])))
            self.shared_data[self.config.DETECTION_TEST_SIZE[i]] = exp.test_size

    def on_update(self) -> bool:
        """
        每帧调用一次
        :return:
        """
        if super().on_update():
            # 目标检测
            self.timer.tic()
            # 推理获得结果和图片
            # outputs: List[tensor(n, 7)]
            outputs, img_info = self.predictor.inference(self.frame, None)
            self.inference_outputs = outputs[0]  # List[tensor(n, 7)] -> tensor(n, 7)
            if self.inference_outputs is not None:
                self.resolve_output(self.inference_outputs)  # 解析推理结果
            self.timer.toc()  # 计算推理开始到输出结果的耗时
            if self.config.yolox_vis:  # opencv可视化
                self._draw_vis()
        return False

    def on_analysis(self):
        logger.info(f"{self.pname} video fps: {1. / max(1e-5, self.update_timer.average_time):.2f}"
                    f" yolox inference fps: {1. / max(1e-5, self.timer.average_time):.2f}")

    def on_resolve_output(self, inference_outputs):
        """
        # yolox inference shape: [n,7]
        # [0,1,2,3]: ltrb bboxes (tsize分辨率下)
        #   [0]: x1
        #   [1]: y1
        #   [2]: x2
        #   [3]: y2
        # [4] * [5]: 置信度 (eg. 0.8630*0.7807)
        # [6]: 类别 (下标从0开始 eg. 0为人)
        # output shape: [n, 6]
        # n: n个对象
        # [0,1,2,3]: ltrb bboxes (基于视频流分辨率)
        #   [0]: x1
        #   [1]: y1
        #   [2]: x2
        #   [3]: y2
        # [4]: 置信度
        # [5]: 类别 (下标从0开始)
        :param inference_outputs:
        :return:
        """
        outputs_cpu = inference_outputs.cpu().numpy()
        bboxes = outputs_cpu[:, :4] / self.scale[self.cur_stream_idx]
        scores = outputs_cpu[:, 4] * outputs_cpu[:, 5]
        classes = outputs_cpu[:, 6]
        if scores.ndim == 1:
            scores = np.expand_dims(scores, axis=1)
            classes = np.expand_dims(classes, axis=1)
        return np.concatenate((bboxes, scores, classes), axis=1)

    def _draw_vis(self):
        if self.inference_outputs is None:
            cv2.imshow(f"yolox window {self.cur_stream_idx}", self.frame)
        else:
            im = np.ascontiguousarray(np.copy(self.frame))
            # im_h, im_w = im.shape[:2]
            # scale = min(self.config.yolox_args_tsize / im_h, self.config.yolox_args_tsize / im_w)
            text_scale = 1
            text_thickness = 1
            line_thickness = 2

            cv2.putText(im, 'frame:%d video_fps:%.2f inference_fps:%.2f num:%d' %
                        (self.current_frame_id[self.cur_stream_idx],
                         1. / max(1e-5, self.update_timer.average_time),
                         1. / max(1e-5, self.timer.average_time),
                         self.inference_outputs.shape[0]), (0, int(15 * text_scale)),
                        cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=text_thickness)
            for i in range(self.inference_outputs.shape[0]):
                tlbr = self.inference_outputs[i, :4] / self.scale[self.cur_stream_idx]
                x1, y1, w, h = tlbr[0], tlbr[1], tlbr[2] - tlbr[0], tlbr[3] - tlbr[1]
                score = self.inference_outputs[i, 4] * self.inference_outputs[i, 5]
                id_text = f"{score:.2f}"
                intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
                cv2.rectangle(im, intbox[0:2], intbox[2:4],
                              color=(0, 0, 255),  # bgr
                              thickness=line_thickness)
                cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                            thickness=text_thickness)
            cv2.imshow(f"yolox window {self.cur_stream_idx}", im)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.config.yolox_vis = False
            self.shared_data[SharedKey.EVENT_ESC].set()  # 退出程序


def create_process(shared_data, config_path: str):
    yoloxComp: YoloxComponent = YoloxComponent(shared_data, config_path)  # 创建组件
    yoloxComp.start()  # 初始化
    yoloxComp.update()  # 算法逻辑循环


if __name__ == '__main__':
    launcher = LauncherComponent("conf/application-dev.yaml")
    launcher.start()
