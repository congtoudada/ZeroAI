import multiprocessing
import os
import pickle
import sys
import time
import requests
from UltraDict import UltraDict
from loguru import logger

from zero.core.base_web_comp import BaseWebComponent
from zero.core.component import Component
from zero.core.global_constant import GlobalConstant
from zero.helper.launch_helper import LaunchHelper
from zero.helper.analysis_helper import AnalysisHelper
from zero.info.launch_info import LaunchInfo
from zero.key.global_key import GlobalKey
from utility.config_kit import ConfigKit
from utility.timer_kit import TimerKit


class LaunchComponent(Component):

    """
    LauncherComponent: 算法启动入口组件
    """
    def __init__(self, application_path: str):
        super().__init__(None)  # LaunchComponent作为主进程，是共享内存的持有者，无需存
        self.config: LaunchInfo = LaunchInfo(ConfigKit.load(application_path))
        self.pname = f"[ {os.getpid()}:main ]"  # 重写父类pname
        self.esc_event = None  # 退出事件
        self.launch_timer = TimerKit(max_flag=0)  # 性能分析器
        self.analysis_flag = 0  # 打印性能分析报告标志

    def on_start(self):
        """
        初始化时调用一次
        :return:
        """
        if os.path.exists(self.config.app_running_file):
            os.remove(self.config.app_running_file)
        # -------------------------------- 1.初始化变量 --------------------------------
        # 设置子进程开启方式
        if sys.platform.startswith('linux'):  # linux默认fork，但fork可能不支持cuda
            multiprocessing.set_start_method('spawn')
        # 初始化退出信号事件
        self.esc_event = multiprocessing.Manager().Event()
        # 注册终止信号 Ctrl+C可以触发
        # signal.signal(signal.SIGINT, self.handle_termination)
        # signal.signal(signal.SIGTERM, self.handle_termination)

        # self.shared_memory: dict = multiprocessing.Manager().dict()
        self.shared_memory = UltraDict(name="global", shared_lock=GlobalConstant.LOCK_MODE)
        self.shared_memory[GlobalKey.EVENT_ESC.name] = self.esc_event
        self.shared_memory[GlobalKey.LAUNCH_COUNTER.name] = 0
        self.shared_memory[GlobalKey.ALL_READY.name] = False
        # ########################## 初始化变量 End ############################

        # -------------------------------- 2.初始化全局服务 --------------------------------
        self.launch_timer.tic()
        launch_helper = LaunchHelper(self.shared_memory)
        launch_helper.execute(self.config.service_list)
        while self.shared_memory[GlobalKey.LAUNCH_COUNTER.name] < len(self.config.service_list):
            time.sleep(0.2)
        logger.info(f"{self.pname} all service init!")
        # ########################## 初始化全局服务 End ############################

        # -------------------------------- 3.初始化视频流 -----------------------------------
        self.shared_memory[GlobalKey.LAUNCH_COUNTER.name] = 0
        launch_helper.execute(self.config.stream_list)
        while self.shared_memory[GlobalKey.LAUNCH_COUNTER.name] < len(self.config.stream_list):
            time.sleep(0.2)
        logger.info(f"{self.pname} all stream init!")
        # ########################## 初始化视频流 End ############################

        # -------------------------------- 3.初始化算法 -----------------------------------------
        self.shared_memory[GlobalKey.LAUNCH_COUNTER.name] = 0
        launch_helper.execute(self.config.algorithm_list)
        while self.shared_memory[GlobalKey.LAUNCH_COUNTER.name] < len(self.config.algorithm_list):
            time.sleep(0.2)
        logger.info(f"{self.pname} all algorithm init!")
        self.shared_memory[GlobalKey.ALL_READY.name] = True  # 所有脚本初始化完毕
        self.launch_timer.toc()
        logger.info(f"{self.pname} 全部脚本启动完毕！启动各脚本耗时: {self.launch_timer.average_time:.6f}s")
        # ########################## 初始化算法 End ############################

        # -------------------------------- 4.主进程相关 --------------------------------------
        # 写文件用于监听，文件删除算法终止
        # 假设 self.config.app_running_file 是一个文件路径
        dir_path = os.path.dirname(self.config.app_running_file)
        # 如果目录不存在，则创建它
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        write_data = {"main": "running"}
        pickle.dump(write_data, open(self.config.app_running_file, 'wb'))
        # ########################## 主进程相关 End ############################

    def on_update(self):
        self.analysis_flag += 1
        if self.config.app_analysis_enable and self.analysis_flag >= self.config.app_analysis_interval:  # 打印分析报告
            self.analysis_flag = 0
            if not self.esc_event.is_set():
                AnalysisHelper.show()
        if not os.path.exists(self.config.app_running_file):  # 当运行文件删除时程序退出
            self.esc_event.set()

    def on_destroy(self):
        self.shared_memory[GlobalKey.ALL_READY.name] = True
        if BaseWebComponent.is_running:
            requests.get(f'http://localhost:{BaseWebComponent.port}/shutdown')
        logger.info("程序将在3s后退出！")
        for i in [3, 2, 1]:
            logger.info(f"倒计时: {i}")
            time.sleep(1)
        if self.config.log_analysis:
            AnalysisHelper.destroy()
        self.shared_memory.unlink()  # 释放共享内存
        if os.path.exists(self.config.app_running_file):
           os.remove(self.config.app_running_file)
        logger.info("程序终止！")
        sys.exit(0)

    # def handle_termination(self, signal_num, frame):
    #     logger.info(f'{self.pname} 接收到信号 {signal_num}, 开始清理并退出...')
    #     self.esc_event.set()


