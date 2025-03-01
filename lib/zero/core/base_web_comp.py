import os
import signal
import sys
from abc import ABC, abstractmethod
from flask import Flask
from loguru import logger

from zero.core.component import Component


class BaseWebComponent(Component, ABC):

    app = Flask(__name__)
    is_running = False
    host = "192.168.1.12"
    port = 5000  # 监听端口

    def __init__(self, shared_memory):
        super().__init__(shared_memory)

    def run_server(self):
        if not BaseWebComponent.is_running:
            @BaseWebComponent.app.route('/shutdown', methods=['GET'])
            def shutdown():
                """强制退出 Flask 应用"""
                logger.info(f"[ {os.getpid()}:reid_search_person ] exit!")
                # os._exit(0)
                # sys.exit(0)
                self.on_destroy()
                os.kill(os.getpid(), signal.SIGTERM)
                logger.info(f"[ {os.getpid()}:reid_search_person ] exit failed!")
                return "Shutdown initiated."

            BaseWebComponent.is_running = True
            BaseWebComponent.app.run(host=BaseWebComponent.host, port=BaseWebComponent.port)
