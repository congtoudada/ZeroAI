import os
import sys
from abc import ABC, abstractmethod
from flask import Flask

from zero.core.component import Component


class BaseWebComponent(Component, ABC):

    app = Flask(__name__)
    is_running = False
    host = "192.168.43.68"
    port = 5000  # 监听端口

    def __init__(self, shared_memory):
        super().__init__(shared_memory)

    def run_server(self):
        if not BaseWebComponent.is_running:
            @BaseWebComponent.app.route('/shutdown')
            def shutdown():
                """强制退出 Flask 应用"""
                # sys.exit(0)
                os._exit(0)

            BaseWebComponent.is_running = True
            BaseWebComponent.app.run(host=BaseWebComponent.host,port=BaseWebComponent.port)
