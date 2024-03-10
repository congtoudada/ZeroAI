import os
import time

import cv2
import numpy as np
from loguru import logger

from samples.lib.business.print_stream.info.print_stream_info import PrintStreamInfo
from zero.core.component.base.component import Component
from zero.core.component.helper.feature.save_video_helper_comp import SaveVideoHelperComponent
from zero.core.key.shared_key import SharedKey
from zero.utility.config_kit import ConfigKit
from zero.utility.timer_kit import TimerKit


class PrintStreamComponent(Component):
    def __init__(self, shared_data, config_path: str):
        super().__init__(shared_data)
        self.config: PrintStreamInfo = PrintStreamInfo(ConfigKit.load(config_path))  # 配置文件
        self.pname = f"[ {os.getpid()}:print_stream for {self.config.input_port}]"  # 组件标识，用于log
        self.frame = None
        self.update_timer = TimerKit()  # 计算帧率
        self._tic = False  # 计算帧率用
        self.current_frame_id = 0  # 当前帧id
        self.frame = None  # 当前帧图像
        self.video_writer: SaveVideoHelperComponent = None  # 存视频帮助类
        # ---- 从流组件获取的参数 ---
        self.stream_width = 0
        self.stream_height = 0
        self.stream_channel = 0
        self.stream_fps = 0
        self.stream_url = ""
        self.stream_cam_id = 0
        self.update_fps = 0

    def on_start(self):
        """
        初始化时调用一次
        :return:
        """
        super().on_start()
        # 获取摄像头视频流的信息
        self.stream_width = self.shared_data[self.config.STREAM_ORIGINAL_WIDTH]
        self.stream_height = self.shared_data[self.config.STREAM_ORIGINAL_HEIGHT]
        self.stream_channel = self.shared_data[self.config.STREAM_ORIGINAL_CHANNEL]
        self.stream_fps = self.shared_data[self.config.STREAM_ORIGINAL_FPS]
        self.stream_url = self.shared_data[self.config.STREAM_URL]
        self.stream_cam_id = self.shared_data[self.config.STREAM_CAMERA_ID]
        self.update_fps = self.shared_data[self.config.STREAM_UPDATE_FPS]
        # 初始化视频导出帮助类
        if self.config.stream_save_video_enable:
            # 设置输出文件夹: (output_dir, 取流路径的文件名）
            # 设置输出文件：取流路径的文件名.mp4
            # folder = os.path.splitext(os.path.basename(self.shared_data[SharedKey.STREAM_URL]))[0]
            filename = os.path.basename(self.stream_url).split('.')[0]
            output_dir = os.path.join(self.config.stream_output_dir, filename)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{filename}.mp4")
            self.video_writer = SaveVideoHelperComponent(output_path,
                                                         self.config.stream_save_video_width,
                                                         self.config.stream_save_video_height,
                                                         self.config.stream_save_video_fps)
            logger.info(f"{self.pname} 输出视频路径: {output_path}")

    def on_resolve_stream(self) -> bool:
        """
        自定义生命周期函数，用于解析从流组件获取的信息
        :return: 返回解析成功or失败
        """
        # 只有不同帧才有必要计算
        frame_info = self.shared_data[self.config.STREAM_FRAME_INFO]  # 从共享缓存获取指定摄像头的输出
        # 只有当前帧序号与获取的帧序号不同时，说明有新的画面传来，才有必要处理
        if frame_info is not None and self.current_frame_id != int(frame_info[SharedKey.STREAM_FRAME_ID]):
            # 更新帧序号
            self.current_frame_id = int(frame_info[SharedKey.STREAM_FRAME_ID])
            # 这里拷贝一份流图像，避免处理过程中，流图像被替换
            self.frame = np.reshape(np.ascontiguousarray(np.copy(frame_info[SharedKey.STREAM_FRAME])),
                                    (self.stream_height, self.stream_width, self.stream_channel))
            # 打印性能分析报告
            self.analysis(frame_info[SharedKey.STREAM_FRAME_ID])
            return True
        else:
            return False

    def on_update(self) -> bool:
        """
        每帧调用，主要用于运行算法逻辑
        :return: bool。通常只有父类返回True，才会执行子类更新逻辑。
        """
        if super().on_update():
            if self.update_fps > 0:  # 控制update执行帧率
                time.sleep(1.0 / self.update_fps)
            ret = self.on_resolve_stream()  # 解析流
            if ret:
                if not self._tic:
                    self.update_timer.tic()
                else:
                    self.update_timer.toc()
                self._tic = not self._tic
                return True
        return False

    def on_draw_vis(self, frame, vis=False, window_name="window", is_copy=True):
        """
        可视化时调用，用于可视化内容
        :param frame: 传入图像
        :param vis: 是否可视化
        :param window_name: 窗口名
        :param is_copy: 是否拷贝图像
        :return:
        """
        if vis and frame is not None:
            frame = cv2.resize(frame, (self.config.print_stream_show_width, self.config.print_stream_show_height))
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.shared_data[SharedKey.EVENT_ESC].set()  # 按q退出程序
        return frame

    def on_analysis(self):
        logger.info(f"{self.pname} video fps: {1. / max(1e-5, self.update_timer.average_time):.2f}")

    def on_destroy(self):
        """
        销毁时调用，用于释放资源
        :return:
        """
        if self.video_writer is not None:
            self.video_writer.destroy()
        super().on_destroy()

    def start(self):
        """
        start操作，控制初始化
        :return:
        """
        super().start()
        logger.info(f"{self.pname} 成功初始化！")
        # 在初始化结束通知给流进程
        self.shared_data[SharedKey.STREAM_WAIT_COUNTER] += 1

    def update(self):
        """
        update操作，控制每帧更新。对于算法组件而言就是死循环，直到收到主进程的退出事件才结束
        :return:
        """
        while True:
            if self.enable:
                self.on_update()
                # 每帧最后，需要可视化和录制视频时才需要绘图
                if self.config.stream_draw_vis_enable or self.config.stream_save_video_enable:
                    if self.frame is not None:
                        im = self.on_draw_vis(self.frame, self.config.stream_draw_vis_enable,
                                              self.config.print_stream_show_title)
                        if self.config.stream_save_video_enable:
                            self.save_video(im, self.video_writer)
            if self.esc_event.is_set():
                self.destroy()
                return

    def save_video(self, frame, vid_writer: SaveVideoHelperComponent):
        """
        保存视频
        :param frame:
        :param vid_writer:
        :return:
        """
        if frame is not None:
            vid_writer.write(frame)


def create_process(shared_data, config_path: str):
    countComp: PrintStreamComponent = PrintStreamComponent(shared_data, config_path)  # 创建组件
    countComp.start()  # 初始化
    countComp.update()  # 算法逻辑循环
