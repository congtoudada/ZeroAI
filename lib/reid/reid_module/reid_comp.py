import multiprocessing
import os
import sys
import time
import yaml
import cv2
from loguru import logger
import numpy as np
import yaml
from zero.core.component.service.base_service_comp import BaseServiceComponent
from reid_key import reidKey
from zero.utility.config_kit import ConfigKit
from reid_info import ReidInfo
from zero.utility.timer_kit import TimerKit
from reid_recognizer import Reid




class ReidComponent(BaseServiceComponent):

    def __init__(self, shared_data, config_path: str):
        super().__init__(shared_data)
        print(type(config_path))
        config1 = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
        self.config: ReidInfo = ReidInfo(ConfigKit.load(config_path))
        print(self.config)

        self.pname = f"[ {os.getpid()}:reid ]"
        print(config1['reid'])
        print(config1['database'])
        reid_model: Reid = Reid(config1['reid'][0])
        #
        self.query_img = None
        self.update_count = 0
        self.timer = TimerKit()



    def on_start(self):
        super().on_start()
        # 初始化请求缓存
        self.query_img = [] # 初始化请求 img
        # self.shared_data[reidKey.REQ.name] = self.req_queue
        # 仅仅是 大概 结构 具体如何写 根据 img的定义再修改
    def on_update(self) -> bool:
        if super().on_update():
            # if self.config.reid_update_fps > 0:
            #     time.sleep(1.0 / self.config.reid_update_fps)
            # #
            # self.update_count = (self.update_count + 1) % sys.maxsize
            # # # 处理请求
            # while not self.query_img.empty():
            #     # 处理请求
            #     req = self.query_img.get()
            #     if req[reidKey.Query] == reidKey.ID_ST:  # 具体是不是这样实现 看 req 是如何定义的
            #         target_query_id = req[TARGET]
            # #         # 返回结果
            #         img_list = req[reidKey.IMG]
            #         res = moudle.solve(img_list,target_query_id)  # 通过模型计算结果
            # #         rsp_key = reidKey.RSP.name + self.config.reid_port + str(pid)
            # #         if self.shared_data.__contains__(rsp_key):
            # #             self.shared_data[rsp_key].put({
            # #
            # #                 # 写入结果
            # #             })
            # #         break  # 每次最多处理一个响应
            #     if req[reidKey.Query] == reidKey.SJ_ID:
            #         img_list = req[reidKey.IMG]
            #         # res = moudle.solve(img_list)  # 通过模型计算结果
            # #         rsp_key = reidKey.RSP.name + self.config.reid_port + str(pid)
            # #         if self.shared_data.__contains__(rsp_key):
            # #             self.shared_data[rsp_key].put({
            # #
            # #                 # 写入结果
            # #             })
            # #         break  # 每次最多处理一个响应
            return False




def create_process(shared_data, config_path: str):
    reidComponent: ReidComponent = ReidComponent(shared_data, config_path)  # 创建组件
    reidComponent.start()  # 初始化
    reidComponent.update()  # 算法逻辑循环

if __name__ == '__main__':
    config_path = "conf/algorithm/reid/reid_root.yaml"
    # config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    img = {}
    component = ReidComponent(img,config_path)
    module = Reid(config_path)
    module.run()
    begin = time.time()

