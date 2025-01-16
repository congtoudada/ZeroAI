import os
from typing import List
import cv2
from UltraDict import UltraDict
from loguru import logger

from zero.core.component import Component
from zero.core.global_constant import GlobalConstant
from zero.helper.analysis_helper import AnalysisHelper
from zero.helper.save_video_helper import SaveVideoHelper
from zero.info.based_stream_info import BasedStreamInfo
from zero.key.global_key import GlobalKey
from zero.key.stream_key import StreamKey
from utility.rtsp_kit import RtspKit
from abc import ABC, abstractmethod


class BasedStreamComponent(Component, ABC):
    """
    基于视频流的算法组件
    """

    def __init__(self, shared_memory):
        super().__init__(shared_memory)
        self.config: BasedStreamInfo = None
        self.pname = f"[ {os.getpid()}:BasedStreamComp ]"
        self.read_dict = []  # 读视频流共享内存
        self.frame_id_cache = []  # 各视频流帧序号缓存，用于对比是否存在新帧
        self.frames = []  # 各视频流当前帧
        self.window_name = []  # 窗口名
        self.output_dir = []  # 输出目录
        self.video_writers: List[SaveVideoHelper] = []  # 存储视频帮助类
        self.rtsp_writers: List[RtspKit] = []  # rtsp推流工具
        self.write_dict = []  # 输出字典
        self.is_draw = []  # 是否需要绘图

    def on_start(self):
        for i, input_port in enumerate(self.config.input_ports):
            # 初始化读字典
            self.read_dict.append(UltraDict(name=input_port,
                                            shared_lock=GlobalConstant.LOCK_MODE))
            if not self.read_dict[i].__contains__(input_port):  # 如果字典不存在视频流输出Key，报错！
                logger.error(f"{self.pname} 输入端口不存在: {input_port} 请检查拼写错误！")
                break
            cam_id = self.read_dict[i][StreamKey.STREAM_CAM_ID.name]  # 摄像头id
            width = self.read_dict[i][StreamKey.STREAM_WIDTH.name]
            height = self.read_dict[i][StreamKey.STREAM_HEIGHT.name]
            fps = self.read_dict[i][StreamKey.STREAM_FPS.name]
            # 初始化视频流帧序号缓存
            self.frame_id_cache.append(0)
            self.frames.append(None)
            # 初始化窗口名
            self.window_name.append(self.pname + ":" + input_port)
            # 初始化输出目录
            if self.config.stream_output_absolute:
                self.output_dir.append(
                    os.path.abspath(os.path.join(self.config.stream_output_dir, f"camera{cam_id}")))
            else:
                self.output_dir.append(os.path.join(self.config.stream_output_dir, f"camera{cam_id}"))
            os.makedirs(self.output_dir[i], exist_ok=True)
            # 初始化视频存储
            if self.config.stream_save_video_enable:
                # 设置输出文件夹
                filename = f"{self.config.stream_save_video_filename}_camId{cam_id}"
                output_path = os.path.join(self.output_dir[i], f"{filename}.mp4")
                video_writer = SaveVideoHelper(output_path, self.config.stream_save_video_resize,
                                               self.config.stream_save_video_width,
                                               self.config.stream_save_video_height,
                                               self.config.stream_save_video_fps)
                self.video_writers.append(video_writer)
                logger.info(f"{self.pname} 视频输出路径: {output_path}")
            # 初始化rtsp
            if self.config.stream_rtsp_enable:
                self.rtsp_writers.append(RtspKit(self.config.stream_rtsp_url, width, height, fps))
            # 初始化输出端口（可选）
            if self.config.output_ports and len(self.config.output_ports) > 0:
                self.write_dict.append(UltraDict(name=f"{self.config.output_ports[i]}",
                                                 shared_lock=GlobalConstant.LOCK_MODE))
                self.write_dict[i][StreamKey.STREAM_CAM_ID.name] = cam_id
                self.write_dict[i][StreamKey.STREAM_WIDTH.name] = width
                self.write_dict[i][StreamKey.STREAM_HEIGHT.name] = height
                self.write_dict[i][StreamKey.STREAM_FPS.name] = fps
            # 初始化是否绘图
            self.is_draw.append(self.config.stream_draw_vis_enable or self.config.stream_save_video_enable or
                                self.config.stream_rtsp_enable)

    def on_update(self):
        # 记录算法耗时
        if self.config.log_analysis:
            self.update_timer.tic()

        # 处理每一个输入端口
        for i, input_port in enumerate(self.config.input_ports):
            frame, user_data = self.on_get_stream(i)  # 解析流
            if frame is None:  # 跳过无效帧
                continue
            self.frames[i] = frame
            user_data = self.on_handle_stream(i, frame, user_data)  # 处理流

            # 是否需要绘图
            if self.is_draw[i]:
                frame = self.on_draw_vis(i, frame, user_data)  # 在多输入端口时，通常只有一个端口返回frame
            # 是否需要可视化
            if frame is not None:
                if self.config.stream_draw_vis_enable:
                    if self.config.stream_draw_vis_resize:
                        # resize会涉及图像拷贝
                        cv2.imshow(self.window_name[i],
                                   cv2.resize(frame,
                                              (self.config.stream_draw_vis_width, self.config.stream_draw_vis_height)))
                    else:
                        cv2.imshow(self.window_name[i], frame)
                # 是否需要录制
                if self.config.stream_save_video_enable:
                    self.video_writers[i].write(frame)
                # 是否需要推流
                if self.config.stream_rtsp_enable:
                    self.rtsp_writers[i].push(frame)

        # 记录算法耗时
        if self.config.log_analysis:
            self.update_timer.toc()
        self.update_delay = max(self.default_update_delay - self.update_timer.recent_time, 0)  # 根据当前推理耗时反推延迟
        # 记录性能日志
        if self.config.log_analysis:
            AnalysisHelper.refresh(f"{self.pname} max time", self.update_timer.max_time * 1000, 100)
            AnalysisHelper.refresh(f"{self.pname} average time", self.update_timer.average_time * 1000)
        # opencv等待
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.shared_memory[GlobalKey.EVENT_ESC.name].set()  # 退出程序

    def on_get_stream(self, read_idx):
        """
        根据下标解析视频流数据包（默认只解析帧id和视频帧，如果有额外数据需用户额外处理）
        :param read_idx: 端口索引
        :return: 最新帧图像（如果不存在最新帧返回None）, 用户数据
        """
        # 只有不同帧才有必要计算
        current_cache_id = self.frame_id_cache[read_idx]  # 当前帧id
        stream_package = self.read_dict[read_idx][self.config.input_ports[read_idx]]  # 视频流数据包
        if stream_package is None:  # 上一个阶段数据包未填充，返回
            return None, None
        current_stream_id = stream_package[StreamKey.STREAM_PACKAGE_FRAME_ID.name]
        if current_cache_id != current_stream_id:
            self.frame_id_cache[read_idx] = current_stream_id
            return stream_package[StreamKey.STREAM_PACKAGE_FRAME.name].copy(), None  # 拷贝图片返回，减少对共享内存的读写
        else:
            return None, None

    @abstractmethod
    def on_handle_stream(self, idx, frame, user_data) -> object:
        """
        处理每一个视频帧（子类重写）
        :param idx: 端口索引
        :param frame: 最新帧图像（如果不存在最新帧返回None）
        :param user_data: 用户从流中解析的额外数据
        :return: 处理后的数据，会传递给on_draw_vis
        """
        return None

    def on_draw_vis(self, idx, frame, user_data):
        """
        可视化绘图函数
        :param idx: 端口索引
        :param frame: 最新帧图像（如果不存在最新帧返回None）
        :param user_data: 处理后的数据
        :return: 绘制后的图像
        """
        return frame

    def on_destroy(self):
        for i in range(len(self.write_dict)):
            self.write_dict[i].unlink()
        for vid in self.video_writers:
            vid.on_destroy()
        super().on_destroy()