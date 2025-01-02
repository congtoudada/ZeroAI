import os
import time
import cv2
from UltraDict import UltraDict
from loguru import logger

from simple_http.simple_http_helper_info import SimpleHttpHelperInfo
from simple_http.simple_http_key import SimpleHttpKey
from utility.config_kit import ConfigKit
from zero.core.global_constant import GlobalConstant


class SimpleHttpHelper:
    def __init__(self, config):
        if config is None or config == "":
            return
        self.config: SimpleHttpHelperInfo = SimpleHttpHelperInfo(ConfigKit.load(config))
        self.shared_memory = UltraDict(name=self.config.output_port, shared_lock=GlobalConstant.LOCK_MODE)

    def get(self, uri: str):
        """
        get请求
        """
        req_package = {
            SimpleHttpKey.HTTP_PACKAGE_URI.name: uri,
            SimpleHttpKey.HTTP_PACKAGE_METHOD.name: 1,
            SimpleHttpKey.HTTP_PACKAGE_JSON.name: None
        }
        if not self.shared_memory.__contains__(self.config.output_port):
            logger.error(f"发送Http请求失败, 没有开启SimpleHttpService! port: {self.config.output_port}")
            return
        self.shared_memory[self.config.output_port].put(req_package)

    def post(self, uri: str, data: dict):
        """
        post请求
        """
        req_package = {
            SimpleHttpKey.HTTP_PACKAGE_URI.name: uri,
            SimpleHttpKey.HTTP_PACKAGE_METHOD.name: 2,
            SimpleHttpKey.HTTP_PACKAGE_JSON.name: data
        }
        if not self.shared_memory.__contains__(self.config.output_port):
            logger.error("发送Http请求失败, 没有开启SimpleHttpService!")
            return
        self.shared_memory[self.config.output_port].put(req_package)
