from abc import ABC, abstractmethod


class IReidWrapper(ABC):
    @abstractmethod
    def inference(self, img):
        """
        输入图像，返回特征
        :param img: 输入图像(ndarray格式 h,w,3 RGB)
        """
        pass

