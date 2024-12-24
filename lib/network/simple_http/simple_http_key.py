from enum import Enum


class SimpleHttpKey(Enum):
    """
    Http请求Key
    使用举例: SimpleHttpKey.HTTP_REQ.name
    """
    # --- 实际请求key REQ ---
    # HTTP_REQ = 0  # 人脸识别请求Key（走配置文件了）
    # 请求地址 eg.http://localhost:8080/algorithm/xxx (xxx为URI实际值)
    # 如果请求方式为GET，请自行把参数写在URI中，如runtime_count?in=1&out=2
    HTTP_PACKAGE_URI = 1
    HTTP_PACKAGE_METHOD = 2  # 请求类型 GET:1 POST:2
    HTTP_PACKAGE_JSON = 3  # 请求内容





