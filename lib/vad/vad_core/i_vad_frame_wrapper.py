from abc import ABC, abstractmethod


class IVadFrameWrapper(ABC):

    @abstractmethod
    def inference_batch(self, frames_batch):
        """
        输入图像batch，返回分数
        :param frames_batch: 输入图像 List[ndarray(n,h,w,3)] BGR n由算法决定 len=batch
        """
        pass

    @abstractmethod
    def inference(self, frames):
        """
        输入图像，返回分数
        :param frames: 输入图像 ndarray(n,h,w,3) BGR n由算法决定
        """
        pass

