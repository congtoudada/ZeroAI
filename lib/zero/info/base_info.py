import json
from typing import Any, Dict


class BaseInfo:
    def __init__(self, data: dict = None):
        self.log_enable = True  # 日志开关
        self.log_level = 1  # 日志过滤等级
        self.log_clean = True  # 是否清除过期日志
        self.log_output_path = ""  # 日志输出路径
        self.log_analysis = True  # 性能日志
        self.update_fps = 60  # 更新帧率（-1代表无上限）
        self.lock_mode = False  # Ultradict是否加锁
        self.set_attrs(data)

    def set_attrs(self, data: Dict[str, Any], prefix: str = ""):
        if data is None:
            return
        for key, value in data.items():
            if isinstance(value, dict):
                self.set_attrs(value, f"{prefix}{key}_")
            else:
                setattr(self, prefix+key, value)
        return self

    def to_dict(self):
        return self.__dict__

    def to_json(self):
        return json.dumps(self.__dict__)

    def from_json(self, data: str):
        self.set_attrs(json.loads(data))

