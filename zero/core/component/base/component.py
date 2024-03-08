import os
from abc import ABC

from loguru import logger

from zero.core.info.base.base_info import BaseInfo
from zero.core.key.shared_key import SharedKey
from zero.utility.log_kit import LogKit


class Component(ABC):
    def __init__(self, shared_data: dict):
        self.pname = "component"
        self.enable = True
        self.shared_data: dict = shared_data
        self.config: BaseInfo = None
        self.pname = f"[ {os.getpid()}:component ]"
        self.esc_event = None

    def on_start(self):
        if self.shared_data is not None:
            self.esc_event = self.shared_data[SharedKey.EVENT_ESC]
        if LogKit.load_info(self.config):  # 新进程设置日志
            pass
            # logger.info(f"{self.pname} 成功运行日志模块! 输出路径: {self.config.log_output_path}")
        else:
            logger.info(f"{self.pname} 日志模块被关闭!")

    def on_update(self) -> bool:
        return True

    def on_destroy(self):
        logger.info(f"{self.pname} destroy!")

    def on_analysis(self):
        pass

    def start(self):
        """
        组件初始化
        :return:
        """
        self.on_start()

    def pause(self):
        """
        组件暂停
        :return:
        """
        self.enable = False

    def update(self):
        """
        组件更新（建议逐帧调用）
        :return:
        """
        while True:
            if self.enable:
                self.on_update()
            if self.esc_event.is_set():
                self.destroy()
                return

    def resume(self):
        """
        组件继续运行
        :return:
        """
        self.enable = True

    def destroy(self):
        """
        组件销毁
        :return:
        """
        self.on_destroy()

    def analysis(self, analysis_flag: int):
        """
        组件分析报告
        :param analysis_flag:
        :return:
        """
        if self.config.log_analysis and analysis_flag % self.config.log_analysis_frequency == 0:
            self.on_analysis()

