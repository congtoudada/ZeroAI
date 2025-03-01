# 测试退出
import requests

from zero.core.base_web_comp import BaseWebComponent

if __name__ == '__main__':
    try:
        requests.get(f'http://{BaseWebComponent.host}:{BaseWebComponent.port}/shutdown')
    except Exception as e:
        pass