import json
from typing import Any, Dict


class BaseInfo:
    def __init__(self, data: dict = None):
        self.log_enable = True
        self.log_level = 1
        self.log_clean = True
        self.log_output_path = ""
        self.log_analysis = True
        self.log_analysis_frequency = 20
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

