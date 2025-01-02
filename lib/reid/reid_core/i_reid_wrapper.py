from abc import ABC, abstractmethod


class IReidWrapper(ABC):
    @abstractmethod
    def extract_feature(self, img):
        """
        抽取特征
        :param img: 输入图像(ndarray格式)
        """
        pass

