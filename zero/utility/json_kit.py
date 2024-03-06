import json


class JsonKit:
    @staticmethod
    def dict_to_json(data: dict):
        return json.dumps(dict)

    # 写入Yaml文件
    @staticmethod
    def json_to_dict(json_str: str):
        return json.loads(str)