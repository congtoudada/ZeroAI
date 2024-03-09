import multiprocessing
import os
import signal
import sys
import time
from multiprocessing import Process

import cv2
from loguru import logger

from zero.core.component.base.component import Component
from zero.core.component.feature.stream_comp import create_process
from zero.core.component.helper.feature.service_group_comp import ServiceGroupComponent
from zero.core.info.feature.app_info import AppInfo
from zero.core.key.shared_key import SharedKey
from zero.utility.config_kit import ConfigKit


class LauncherComponent(Component):
    """
    LauncherComponent: 算法启动入口组件
    """
    def __init__(self, application_path: str):
        super().__init__(None)
        self.config: AppInfo = AppInfo(ConfigKit.load(application_path))
        self.service_group_comp = None
        self.pname = f"[ {os.getpid()}:main ]"
        self.global_shared_proxy = None  # 全局代理
        self.esc_event = None

    def on_start(self):
        """
        初始化时调用一次
        :return:
        """
        super().on_start()
        # 设置进程开启方式
        # multiprocessing.set_start_method('spawn')
        if sys.platform.startswith('linux'):  # linux默认fork，但fork不支持cuda
            multiprocessing.set_start_method('spawn')
        self.esc_event = multiprocessing.Manager().Event()
        signal.signal(signal.SIGINT, self.handle_termination)
        signal.signal(signal.SIGTERM, self.handle_termination)

        # -------------------------------- 1.初始化共享内存 --------------------------------
        self.global_shared_proxy: dict = multiprocessing.Manager().dict()
        self.global_shared_proxy[SharedKey.LOCK] = multiprocessing.Manager().Lock()
        self.global_shared_proxy[SharedKey.EVENT_ESC] = self.esc_event
        self.global_shared_proxy[SharedKey.WAIT_COUNTER] = 0

        # -------------------------------- 2.初始化全局服务 --------------------------------
        self.service_group_comp = ServiceGroupComponent(self.global_shared_proxy, self.config)
        self.service_group_comp.start()
        # 等待所有服务初始化完成
        while self.global_shared_proxy[SharedKey.WAIT_COUNTER] < len(self.config.service):
            time.sleep(0.2)
        # -------------------------------- 初始化全局服务End --------------------------------

        # -------------------------------- 3.初始化视频流 -----------------------------------
        self.global_shared_proxy[SharedKey.STREAM_WAIT_COUNTER] = 0
        self.global_shared_proxy[SharedKey.STREAM_WAIT_COUNTER_MAX] = 0
        for i, cam_config_path in enumerate(self.config.cam_list):
            logger.info(f"{self.pname} 初始化摄像头: {cam_config_path}")
            self.global_shared_proxy[SharedKey.STREAM_WAIT_COUNTER_MAX] += 1
            # --- 初始化每个摄像头的算法 ---
            Process(target=create_process,
                    args=(self.global_shared_proxy, cam_config_path),
                    daemon=False).start()
        # -------------------------------- 初始化视频流End --------------------------------------

    def on_update(self) -> bool:
        time.sleep(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.esc_event.set()
            self.destroy()
        return False

    def update(self):
        while True:
            if self.enable:
                self.on_update()
            if self.esc_event.is_set():
                self.destroy()
                return

    def on_destroy(self):
        logger.info("程序将在3s后退出！")
        for i in [3, 2, 1]:
            logger.info(f"倒计时: {i}")
            time.sleep(1)
        logger.info("程序终止！")
        sys.exit(0)

    def handle_termination(self, signal_num, frame):
        print(f'接收到信号 {signal_num}, 开始清理并退出...')
        # 在这里执行清理操作，例如关闭文件、断开网络连接等
        self.esc_event.set()
        self.destroy()


if __name__ == '__main__':
    launcher = LauncherComponent("conf/application-dev.yaml")
    launcher.start()
    launcher.update()


