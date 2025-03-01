# 测试退出
import socket

import requests

from zero.core.base_web_comp import BaseWebComponent


def is_port_open(host, port, timeout=1):
    """
    判断指定主机上的某端口是否启用。

    :param host: 主机名或 IP 地址
    :param port: 端口号
    :param timeout: 超时时间(单位：秒)，默认 1 秒
    :return: True/False
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)  # 设置连接超时
    try:
        sock.connect((host, port))
        # 如果能连接成功，则认为端口启用
        return True
    except (socket.timeout, ConnectionRefusedError):
        # 超时或连接被拒绝，端口未启用
        return False
    finally:
        sock.close()


if __name__ == '__main__':
    test_host = '127.0.0.1'
    test_port = 5000

    if is_port_open(test_host, test_port):
        print(f'端口 {test_port} 已启用，正在监听连接。')
    else:
        print(f'端口 {test_port} 未启用或连接被拒绝。')

    # try:
    #     requests.get(f'http://{BaseWebComponent.host}:{BaseWebComponent.port}/shutdown')
    # except Exception as e:
    #     pass


