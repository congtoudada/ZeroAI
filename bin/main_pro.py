import multiprocessing
import sys
import time
from multiprocessing import Process
import cv2
import numpy as np
from loguru import logger

from zero.core.info.feature.stream_info import StreamInfo
from zero.core.deprecated.cache_key import CacheKey
from zero.core.key.shared_key import SharedKey
from zero.core.component.feature.stream_comp import get_stream_process
from zero.utility.config_kit import ConfigKit


def run():
    # -------------------------------- 1.初始化 --------------------------------
    # 设置进程开启方式
    if sys.platform.startswith('linux'):  # linux默认fork，但fork不支持cuda
        multiprocessing.set_start_method('spawn')

    ret = ConfigKit.load("conf/application-dev.yaml")
    arr = []
    for cam in ret['cam_list']:
        arr.append(ConfigKit.load(cam))
    instance = StreamInfo(arr[0])

    # 配置文件代理
    config_proxy = multiprocessing.Manager().dict(instance.to_dict())

    # 进程间代理
    shared_proxy = multiprocessing.Manager().dict()
    esc_event = multiprocessing.Manager().Event()
    lock = multiprocessing.Manager().Lock()

    shared_proxy[SharedKey.STREAM_WAIT_COUNTER] = 0
    shared_proxy[SharedKey.EVENT_ESC] = esc_event
    shared_proxy[SharedKey.STREAM_LOCK] = lock
    shared_proxy[SharedKey.STREAM_ORIGINAL_FRAME] = None

    # 临时代理（传递一些一次性的单向数据）
    cache_proxy = multiprocessing.Manager().dict()
    cache_proxy[CacheKey.STREAM_WAIT_MAX] = 1

    # -------------------------------- 2.工作循环 --------------------------------
    Process(target=get_stream_process,
            args=(config_proxy, shared_proxy, cache_proxy),
            daemon=True).start()

    time.sleep(2)
    shared_proxy[SharedKey.STREAM_WAIT_COUNTER] = 1
    width = int(shared_proxy[SharedKey.STREAM_ORIGINAL_WIDTH])
    height = int(shared_proxy[SharedKey.STREAM_ORIGINAL_HEIGHT])
    channel = int(shared_proxy[SharedKey.STREAM_ORIGINAL_CHANNEL])
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            esc_event.set()
            break
        if shared_proxy[SharedKey.STREAM_ORIGINAL_FRAME] is not None:
            image = np.copy(shared_proxy[SharedKey.STREAM_ORIGINAL_FRAME])
            image = np.reshape(image, (height, width, channel))
            cv2.imshow("debug window", image)

    logger.info("程序将在3s后退出！")
    for i in [3, 2, 1]:
        logger.info(f"倒计时: {i}")
        time.sleep(1)


if __name__ == '__main__':
    run()

