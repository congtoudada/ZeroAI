import os
import sys
import time
import traceback
import cv2
from loguru import logger

from zero.core.component.helper.feature.algorithm_group_comp import AlgorithmGroupComponent
from zero.core.info.feature.stream_info import StreamInfo
from zero.core.component.base.component import Component
from zero.core.key.shared_key import SharedKey
from zero.utility.config_kit import ConfigKit


class StreamComponent(Component):
    """
    StreamComponent: 摄像头取流组件
    """
    def __init__(self, shared_data, config_path: str):
        super().__init__(shared_data)
        self.config: StreamInfo = StreamInfo(ConfigKit.load(config_path))
        self.algo_group_comp = AlgorithmGroupComponent(shared_data, self.config)
        self.pname = f"[ {os.getpid()}:camera{self.config.stream_cam_id} ]"
        self.cap = None
        self.frame_fps = 24
        # 自身用数据
        self.success_frame = 0  # 当前已读帧数（不包含丢帧）
        self.drop_flag = 0  # 丢帧标记
        self.last_time = 0  # 上次读取视频流的时间
        self.stream_frame_info = {}

    def on_start(self):
        """
        初始化时调用一次
        :return:
        """
        super().on_start()
        self.cap = cv2.VideoCapture(self.config.stream_url)
        self.frame_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.stream_frame_info = {
            SharedKey.STREAM_FRAME_ID: 0,
            SharedKey.STREAM_FRAME: None
        }
        self.shared_data[self.config.STREAM_FRAME_INFO] = self.stream_frame_info
        self.shared_data[self.config.STREAM_ORIGINAL_WIDTH] = int(self.config.stream_width)
        self.shared_data[self.config.STREAM_ORIGINAL_HEIGHT] = int(self.config.stream_height)
        self.shared_data[self.config.STREAM_ORIGINAL_CHANNEL] = int(self.config.stream_channel)
        self.shared_data[self.config.STREAM_ORIGINAL_FPS] = self.frame_fps
        self.shared_data[self.config.STREAM_URL] = self.config.stream_url
        self.shared_data[self.config.STREAM_CAMERA_ID] = self.config.stream_cam_id
        self.shared_data[self.config.STREAM_UPDATE_FPS] = self.config.stream_update_fps
        self.shared_data[SharedKey.STREAM_WAIT_COUNTER_MAX] += len(self.config.stream_algorithm)
        # self.face_helper.start()
        # img = cv2.imread('res/images/face/database/48-0001.jpg')
        # self.face_helper.send(1, img)

    def second_start(self):
        # 多进程启动算法
        self.algo_group_comp.start()

        # 等待所有算法初始化完成
        while self.shared_data[SharedKey.STREAM_WAIT_COUNTER] < self.shared_data[SharedKey.STREAM_WAIT_COUNTER_MAX]:
            time.sleep(0.2)

        logger.info(f"{self.pname} 所有算法成功初始化！开始取流 URL: {self.config.stream_url} fps: {self.frame_fps}")

    def on_update(self) -> bool:
        """
        每帧调用一次
        :return:
        """
        try:
            if self.cap.isOpened() and self._can_read():
                # self.face_helper.update()
                status, frame = self.cap.read()
                if status:
                    self.success_frame = (self.success_frame + 1) % sys.maxsize
                    frame = cv2.resize(frame, (self.config.stream_width, self.config.stream_height))
                    self.stream_frame_info[SharedKey.STREAM_FRAME_ID] = self.success_frame
                    # self.stream_frame_info[SharedKey.STREAM_FRAME_TIME] = time.time()
                    # self.stream_frame_info[SharedKey.STREAM_FRAME] = frame
                    self.stream_frame_info[SharedKey.STREAM_FRAME] = frame.flatten()
                    # self.stream_frame_info[SharedKey.STREAM_FRAME] = cv2.imencode(".jpg", frame)[1].tobytes()
                    # 填充输出
                    self.shared_data[self.config.STREAM_FRAME_INFO] = self.stream_frame_info
                    if self.config.stream_delay:
                        time.sleep(1.0 / (self.frame_fps * self.config.stream_delay_speed))
                    self.analysis(self.success_frame)  # 分析报告
                else:
                    # logger.error(f"{self.pname} 取流结束或异常！")
                    time.sleep(1)

        except Exception as e:
            logger.error(f"{self.pname} {traceback.format_exc()}")
            time.sleep(1)  # 等待1s
            self.cap = cv2.VideoCapture(self.config.stream_url)  # 尝试重新取流
        return False

    def on_analysis(self):
        logger.info(f"{self.pname} success_frame: {self.success_frame}")

    def _can_read(self) -> bool:
        self.drop_flag += 1
        if self.drop_flag >= self.config.stream_drop_interval:
            self.drop_flag = 0
            return False
        return True


def create_process(shared_data, config_path: str):
    # 创建视频流组件
    streamComp: StreamComponent = StreamComponent(shared_data, config_path)
    streamComp.start()  # 一阶段：初始化自身
    # 在初始化结束通知给流进程
    shared_data[SharedKey.STREAM_WAIT_COUNTER] += 1
    streamComp.second_start()  # 二阶段：初始化算法
    streamComp.update()  # 算法逻辑循环


if __name__ == '__main__':
    pass
    # # -------------------------------- 1.初始化 --------------------------------
    # ret = ConfigKit.load("conf/application-dev.yaml")
    # arr = []
    # for cam in ret['cam_list']:
    #     arr.append(ConfigKit.load(cam))
    # # instance = StreamInfo().set_attrs(arr[0])
    # instance = StreamInfo(arr[0])
    #
    # # 配置文件代理
    # config_proxy = multiprocessing.Manager().dict(instance.to_dict())
    #
    # # 进程间代理
    # shared_proxy = multiprocessing.Manager().dict()
    # esc_event = multiprocessing.Manager().Event()
    #
    # shared_proxy[SharedKey.STREAM_WAIT_COUNTER] = 0
    # shared_proxy[SharedKey.STREAM_WAIT_MAX] = 1
    # shared_proxy[SharedKey.EVENT_ESC] = esc_event
    #
    # # 临时代理（传递一些一次性的单向数据）
    # # cache_proxy = multiprocessing.Manager().dict()
    # # cache_proxy[CacheKey.STREAM_WAIT_MAX] = 1
    #
    # # -------------------------------- 2.工作循环 --------------------------------
    # Process(target=create_stream_process,
    #         args=(config_proxy, shared_proxy),
    #         daemon=True).start()
    #
    # time.sleep(2)
    # shared_proxy[SharedKey.STREAM_WAIT_COUNTER] = 1
    # # time.sleep(99999)
    # width = int(shared_proxy[SharedKey.STREAM_ORIGINAL_WIDTH])
    # height = int(shared_proxy[SharedKey.STREAM_ORIGINAL_HEIGHT])
    # channel = int(shared_proxy[SharedKey.STREAM_ORIGINAL_CHANNEL])
    # while True:
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         esc_event.set()
    #         break
    #     if shared_proxy[SharedKey.STREAM_ORIGINAL_FRAME] is not None:
    #         # 通过np.ascontiguousarray和np.copy的组合，你可以确保得到一个连续存储的数组副本，这在进行进一步的图像处理或分析时可能会更加高效。
    #         image = np.ascontiguousarray(np.copy(shared_proxy[SharedKey.STREAM_ORIGINAL_FRAME]))
    #         image = np.reshape(image, (height, width, channel))
    #         cv2.imshow("debug window", image)
    #
    # logger.info("程序将在3s后退出！")
    # for i in [3, 2, 1]:
    #     logger.info(f"倒计时: {i}")
    #     time.sleep(1)
