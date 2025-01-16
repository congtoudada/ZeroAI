import json
import multiprocessing
import os
import time
import traceback
from UltraDict import UltraDict
from flask import request
from loguru import logger

from reid_core.helper.reid_helper import ReidHelper
from reid_core.reid_search_person_info import ReidSearchPersonInfo
from utility.config_kit import ConfigKit
from zero.core.base_web_comp import BaseWebComponent
from zero.core.global_constant import GlobalConstant
from zero.key.global_key import GlobalKey


class ReidSearchPersonComp(BaseWebComponent):
    def __init__(self, shared_memory, config_path: str):
        super().__init__(shared_memory)
        self.config: ReidSearchPersonInfo = ReidSearchPersonInfo(ConfigKit.load(config_path))
        self.reid_helper = ReidHelper(self.config.reid_sp_helper_config, None, self.search_person_callback)
        self.pname = f"[ {os.getpid()}:reid_search_person ]"
        self.is_searching = False  # 是否正在执行找人请求
        self.tick_fps = 30
        self.tick_max_cnt = self.tick_fps * 10  # 一个请求最多处理10s
        self.tick_flag = 0
        self.rsp_package = []

    def on_start(self):
        # 处理 POST 请求
        @BaseWebComponent.app.route('/search_person')
        def search_person():
            # 获取 GET 请求中名为 'param' 的参数
            per_id = request.args.get('per_id', type=int)  # 自动转换为 int 类型
            if per_id is not None:
                # return f'Parameter received: {per_id}'
                self.rsp_package.clear()
                self.is_searching = True
                self.tick_flag = 0
                is_valid, img_path = self.reid_helper.try_send_search_person(per_id)
                if not is_valid:  # per_id没有对应的本地特征
                    return []
                # 添加第一张图
                cam_id = img_path.split('_')[-1].split('.')[0]
                self.rsp_package.append({
                    "cam_id": cam_id,
                    "time": time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()),
                    "img_path": img_path,
                    "score": 1.0,
                })
                # 轮询
                while self.is_searching and self.tick_flag < self.tick_max_cnt:
                    self.reid_helper.tick()
                    self.tick_flag += 1
                    time.sleep(1.0 / self.tick_fps)
                # 响应
                return json.dumps(self.rsp_package)
            else:
                return None

    def search_person_callback(self, package: list):
        self.is_searching = False
        # print(f"{self.pname} search success! {package}")
        self.rsp_package = self.rsp_package + package


def create_process(shared_memory, config_path: str):
    comp = ReidSearchPersonComp(shared_memory, config_path)
    try:
        comp.start()
        shared_memory[GlobalKey.LAUNCH_COUNTER.name] += 1
        comp.run_server()  # 启动服务(已经启动则忽略)
        # comp.update()
    except KeyboardInterrupt:
        comp.on_destroy()
    except Exception as e:
        logger.error(f"ReidSearchPersonComponent: {e}")
        logger.error(traceback.format_exc())  # 打印完整的堆栈信息
        comp.on_destroy()


if __name__ == '__main__':
    shared_memory = UltraDict(name="global", shared_lock=GlobalConstant.LOCK_MODE)
    shared_memory[GlobalKey.EVENT_ESC.name] = multiprocessing.Manager().Event()
    reid_comp = ReidSearchPersonComp(shared_memory, config_path="")
    reid_comp.start()
