import json
import multiprocessing
import os
import traceback
from typing import List
import requests
from UltraDict import UltraDict
from loguru import logger

from simple_http.simple_http_info import SimpleHttpInfo
from simple_http.simple_http_key import SimpleHttpKey
from simple_http.simple_http_task import SimpleHttpTask
from zero.core.component import Component
from zero.key.global_key import GlobalKey
from utility.config_kit import ConfigKit
from utility.object_pool import ObjectPool


class SimpleHttpComponent(Component):
    """
    简单的Http服务
    """
    def __init__(self, shared_memory, config_path: str):
        super().__init__(shared_memory)
        self.config: SimpleHttpInfo = SimpleHttpInfo(ConfigKit.load(config_path))  # 配置文件内容
        self.http_shared_memory = UltraDict(name=self.config.input_port)
        self.pname = f"[ {os.getpid()}:{self.config.input_port} ]"
        self.req_queue = None  # 后端请求队列
        self.headers = {
            'content-type': 'application/json;charset=utf-8'
        }
        self.delay_queue: List[SimpleHttpTask] = []  # 延迟队列（所有消息必须延迟发送，确保后端能正确copy图片）
        self.task_pool: ObjectPool = ObjectPool(20, SimpleHttpTask)  # 对象池

    def on_start(self):
        super().on_start()
        # 初始化请求缓存
        self.req_queue = multiprocessing.Manager().Queue()
        self.http_shared_memory[self.config.input_port] = self.req_queue

    def on_update(self) -> bool:
        # 处理请求
        while not self.req_queue.empty():
            req_package = self.req_queue.get()
            uri = req_package[SimpleHttpKey.HTTP_PACKAGE_URI.name]
            method = req_package[SimpleHttpKey.HTTP_PACKAGE_METHOD.name]
            content = req_package[SimpleHttpKey.HTTP_PACKAGE_JSON.name]
            full_url = self._get_full_url(uri)
            if self.config.http_delay_enable:
                self.send_request_delay(full_url, method, content)
            else:
                self.send_request(full_url, method, content)
        # 处理延迟发送
        if self.config.http_delay_enable:
            remove_keys = []
            for i in range(len(self.delay_queue) - 1, -1, -1):  # 逆序遍历
                self.delay_queue[i].update()  # 更新每个任务的状态
                if self.delay_queue[i].delay_flag > self.config.http_delay_frame:
                    task = self.delay_queue[i]
                    self.send_request(task.url, task.method, task.content)
                    self.task_pool.push(task)
                    remove_keys.append(i)
            # for i in range(len(remove_keys)):
            for i, key in enumerate(remove_keys):
                self.delay_queue.pop(key)
        return False

    def send_request_delay(self, url, method, content):
        task: SimpleHttpTask = self.task_pool.pop()
        task.init(url=url, method=method, content=content)
        self.delay_queue.append(task)

    def send_request(self, url, method, content):
        response = None
        try:
            if method == 1:  # GET
                logger.info(f"{self.pname} 发送Get请求，路径: {url}")
                response = requests.get(url)
            elif method == 2:
                logger.info(f"{self.pname} 发送Post请求，路径: {url}")
                response = requests.post(url, headers=self.headers, data=json.dumps(content))
        except Exception as e:
            logger.error(f"{self.pname} {e}")
        if response is not None:
            if response.status_code == 200:
                logger.info(f"{self.pname} 成功收到后端响应，路径: {url}")
            else:
                logger.error(f"{self.pname} 请求失败[{response.status_code}]，没有收到后端响应，路径: {url}")

    def _get_full_url(self, uri: str) -> str:
        return f"http://{self.config.http_web_address}{uri}"

    def on_destroy(self):
        self.http_shared_memory.unlink()
        super().on_destroy()


def create_process(shared_memory, config_path: str):
    comp = SimpleHttpComponent(shared_memory, config_path)
    try:
        comp.start()
        shared_memory[GlobalKey.LAUNCH_COUNTER.name] += 1
        comp.update()
    except KeyboardInterrupt:
        comp.on_destroy()
    except Exception as e:
        logger.error(f"SimpleHttpComponent: {e}")
        logger.error(traceback.format_exc())  # 打印完整的堆栈信息
        comp.on_destroy()


if __name__ == '__main__':
    url = "http://localhost:8080/unity/camera_list"
    print("开始Get请求")
    response = requests.get(url)
    print(response)

    url = "http://localhost:8080/unity/record_delete"
    headers = {
        'content-type': 'application/json;charset=utf-8'
    }
    data = {
        'key': 10,
        'pageType': 1
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    print(response)
    # url = "http://210.30.97.233:8080/unity/camera_list"
    # print("开始Post请求")
    # WebKit.post(url, {"ReqPersonWarnADTO": {'camId': 10}})
    # WebKit.post(url, {"DTO": 10})

