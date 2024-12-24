import os
import sys
import time
import cv2
from UltraDict import UltraDict
from loguru import logger

from zero.core.component import Component
from zero.info.stream_info import StreamInfo
from zero.key.global_key import GlobalKey
from zero.key.stream_key import StreamKey
from utility.core.config_kit import ConfigKit


class StreamComponent(Component):
    """
    StreamComponent: 摄像头取流组件
        输出端口: STREAM_OUTPUT_PORT + cam_id
    """
    def __init__(self, shared_memory, config_path: str):
        super().__init__(shared_memory)
        self.config: StreamInfo = StreamInfo(ConfigKit.load(config_path))
        self.stream_shared_memory = UltraDict(name=self.config.output_port,
                                              shared_lock=self.config.lock_mode)
        self.pname = f"[ {os.getpid()}:camera{self.config.stream_cam_id} ]"
        self.cap = None
        self.frame_fps = 24
        # 自身用数据
        self.read_frame_id = 0  # 当前已读帧数（不包含丢帧）
        # self.real_frame_id = 0  # 真实视频帧数（包含丢帧）
        self.read_flag = 0  # 读帧标记

    def on_start(self):
        """
        初始化时调用一次
        :return:
        """
        super().on_start()

        self.cap = cv2.VideoCapture(self.config.stream_url)
        self.frame_fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        stream_package = {
            StreamKey.STREAM_PACKAGE_FRAME_ID.name: 0,
            StreamKey.STREAM_PACKAGE_FRAME.name: None
        }
        self.stream_shared_memory[self.config.output_port] = stream_package
        self.stream_shared_memory[StreamKey.STREAM_CAM_ID.name] = self.config.stream_cam_id
        self.stream_shared_memory[StreamKey.STREAM_WIDTH.name] = width / self.config.stream_reduce_scale
        self.stream_shared_memory[StreamKey.STREAM_HEIGHT.name] = height / self.config.stream_reduce_scale
        self.stream_shared_memory[StreamKey.STREAM_FPS.name] = self.frame_fps
        # 在初始化结束通知给流进程
        self.shared_memory[GlobalKey.LAUNCH_COUNTER.name] += 1

        # ------------------------ 多进程启动算法 ------------------------
        # 等待所有算法初始化完成
        while not self.shared_memory[GlobalKey.ALL_READY.name]:
            if self.config.stream_runtime_enable:
                self.cap.grab()
            else:
                time.sleep(0.2)

        # 初始化后先丢N帧
        if self.config.stream_runtime_enable:
            for i in range(self.config.stream_runtime_drop_count):
                self.cap.grab()
        logger.info(f"{self.pname} 所有算法成功初始化！开始取流 URL: {self.config.stream_url}")

    def on_update(self) -> bool:
        """
        每帧调用一次
        :return:
        """
        self.read_flag += 1
        # self.real_frame_id += 1
        if self.read_flag >= self.config.stream_read_frequency:
            self.read_flag = 0
            # 读取帧并解码
            status, frame = self.cap.read()
            if status:
                self.process_frame(frame)
        else:
            # 丢帧
            self.cap.grab()
        if not self.config.stream_runtime_enable:  # 非实时视频流，额外引入延迟
            time.sleep(1.0 / self.frame_fps)
        return False

    def process_frame(self, frame):
        """
        处理帧
        :param frame:
        :return:
        """
        self.read_frame_id = (self.read_frame_id + 1) % sys.maxsize
        frame = frame[::self.config.stream_reduce_scale, ::self.config.stream_reduce_scale, :]  # 缩放图片
        stream_package = {StreamKey.STREAM_PACKAGE_FRAME_ID.name: self.read_frame_id,
                          StreamKey.STREAM_PACKAGE_FRAME.name: frame}
        # 填充输出
        self.stream_shared_memory[self.config.output_port] = stream_package

    def on_destroy(self):
        time.sleep(1)  # 下游业务先释放内存（这里可以设置0~3的值）
        self.stream_shared_memory.unlink()
        super().on_destroy()


def create_process(shared_memory, config_path: str):
    # 创建视频流组件
    comp: StreamComponent = StreamComponent(shared_memory, config_path)
    try:
        comp.start()  # 初始化
        # shared_memory[GlobalKey.LAUNCH_COUNTER.name] += 1  # 视频流在内部递增
        comp.update()  # 算法逻辑循环
    except KeyboardInterrupt:
        comp.on_destroy()
    except Exception as e:
        logger.error(f"StreamComponent: {e}")
        comp.on_destroy()

