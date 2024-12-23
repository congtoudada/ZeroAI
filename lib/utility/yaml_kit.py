import yaml


class YamlKit:
    @staticmethod
    def read_yaml(file, encoding='utf-8'):
        """
        读取Yaml文件
        :param file:
        :param encoding:
        :return:
        """
        with open(file, encoding=encoding) as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

    # 写入Yaml文件
    @staticmethod
    def write_yaml(file, wtdata, encoding='utf-8'):
        """
        写入yaml文件
        :param file:
        :param wtdata:
        :param encoding:
        :return:
        """
        with open(file, encoding=encoding, mode='w') as f:
            yaml.dump(wtdata, stream=f, allow_unicode=True)