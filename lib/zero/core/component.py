import os
import time
from abc import ABC
from loguru import logger
import json

from utility.log_kit import LogKit
from utility.timer_kit import TimerKit
from zero.info.base_info import BaseInfo
from zero.key.global_key import GlobalKey


class Component(ABC):
    """
    所有组件的基类
    """
    def __init__(self, shared_memory: dict):
        self.enable = True  # 是否启用
        self.shared_memory: dict = shared_memory  # 全局共享内存
        self.config: BaseInfo = None  # 配置文件
        self.pname = f"[ {os.getpid()}:component ]"
        self.esc_event = None  # 退出事件
        self.update_delay = 1.0 / 60  # 每帧延迟
        self.default_update_delay = 1.0 / 60  # 默认帧延迟（避免反复计算）
        self.update_timer = TimerKit()

    def __on_start(self):
        # 绑定退出事件（根组件绑定即可）
        if self.shared_memory is not None:
            self.esc_event = self.shared_memory[GlobalKey.EVENT_ESC.name]
        # 初始化日志模块，只有root组件才需要配置
        if not LogKit.load_info(self.config):
            logger.info(f"{self.pname} 日志模块被关闭!")
        # 转换为带缩进的JSON字符串并输出
        if self.config is not None:
            json_string = json.dumps(self.config.__dict__, indent=4)
            logger.info(f"{self.pname} {type(self)} 配置文件参数: \n{json_string}")
            self.default_update_delay = 1.0 / self.config.update_fps
        else:
            logger.info(f"{self.pname} {type(self)} 没有配置文件！")
            self.default_update_delay = 1.0 / 30
        self.update_delay = self.default_update_delay
        self.on_start()

    def on_start(self):
        """
        初始化时调用一次
        """
        self.on_start()

    def __on_update(self):
        self.update_delay = self.default_update_delay
        self.on_update()

    def on_update(self):
        """
        每帧执行
        """
        pass

    def __on_destroy(self):
        logger.info(f"{self.pname} destroy!")
        self.on_destroy()

    def on_destroy(self):
        """
        组件销毁时执行
        """
        pass

    def pause(self):
        """
        组件暂停运行（根组件暂停，全部暂停）
        :return:
        """
        self.enable = False

    def resume(self):
        """
        组件继续运行
        :return:
        """
        self.enable = True

    def start(self):
        # 组件初始化
        self.__on_start()

    def update(self):
        # 收到退出信号
        if self.esc_event.is_set():
            self.__on_destroy()
            return
        # 组件更新
        while True:
            if self.enable:
                self.__on_update()  # 先执行父组件的更新
            if self.esc_event.is_set():
                self.__on_destroy()
                return
            if self.update_delay >= 0:
                time.sleep(self.update_delay)
