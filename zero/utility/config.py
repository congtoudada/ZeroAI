from zero.utility.yaml import YamlKit


class ConfigKit:
    @staticmethod
    def load_by_path(template_path: str, cur_path: str) -> dict:
        """
        加载配置文件，用当前配置重载模板配置
        :param template_path: 模板路径
        :param cur_path: 当前路径
        :return:
        """
        template_file = YamlKit.read_yaml(file=template_path)
        cur_file = YamlKit.read_yaml(file=cur_path)
        return ConfigKit.equip_file(template_file, cur_file)

    @staticmethod
    def load_by_dict(template_dict: dict, cur_dict: dict) -> dict:
        """
        加载配置文件，用当前配置重载模板配置
        :param template_dict: 模板yaml文件
        :param cur_dict: 当前yaml文件
        :return:
        """
        # 更新
        ConfigKit._update(template_dict, cur_dict)
        ConfigKit._addNew(template_dict, cur_dict)
        return template_dict

    @staticmethod
    def _update(template_dict: dict, override_dict: dict):
        # 更新模板参数
        for key in template_dict.keys():
            if override_dict.__contains__(key):
                # 如果value是字典，需要递归
                if type(template_dict[key]) is dict:
                    ConfigKit._update(template_dict[key], override_dict[key])
                else:
                    template_dict[key] = override_dict[key]

    @staticmethod
    def _addNew(template_dict: dict, override_dict: dict):
        # 添加新参数
        for key in override_dict.keys():
            if not template_dict.__contains__(key):
                template_dict[key] = override_dict[key]


