import os
import random

import cv2
import numpy as np
import torch.nn.functional as F
import torch
from PIL import Image
from loguru import logger
from torchvision import transforms
from torchvision.ops import roi_align

from jigsaw.models import model
from jigsaw.zero.vad_jigsaw_info import VadJigsawInfo
from utility.config_kit import ConfigKit
from vad_core.i_vad_obj_wrapper import IVadObjWrapper


class VadJigsawHelper(IVadObjWrapper):
    """
    对象级视频异常检测器
    """

    def __init__(self, config_path: str, sample_nums):
        self.config: VadJigsawInfo = VadJigsawInfo(ConfigKit.load(config_path))
        self.pname = f"[ {os.getpid()}:vad_jigsaw ]"
        self.feature_size = tuple(self.config.vad_jigsaw_align_size)
        self.sample_nums = sample_nums
        self.model = model.WideBranchNet(time_length=sample_nums,
                                         num_classes=[sample_nums ** 2, 81])

        self.to_tensor = transforms.ToTensor()
        self.on_start()

    def on_start(self):
        state = torch.load(self.config.vad_jigsaw_ckpt)
        logger.info('vad load: ' + self.config.vad_jigsaw_ckpt)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.config.vad_jigsaw_device)
        self.model.eval()

    def crop_objs(self, frames, objs_info):
        """
        :param frames: ndarray(n=sample_num,h,w,3)
        :param objs_info: List[ndarray(n,7)]
        Returns: List[(3, sample_num, 64, 64)]
        """
        # frames = frames.astype(np.float32)
        # C: channel
        # D: 图片数
        # H: 高
        # W: 宽
        C, D, H, W = frames.shape[3], frames.shape[0], frames.shape[1], frames.shape[2]
        # print("frames shape:", frames.shape)
        torch_imgs = None
        # 图片预处理
        for i in range(D):
            # 手动归一化可能有问题
            # frames[i] = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
            # frames[i] /= 255.0
            # ToTensor归一化
            img = frames[i]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img, 'RGB')
            img = self.to_tensor(img).unsqueeze(dim=0).permute(1, 0, 2, 3)
            if torch_imgs is None:
                torch_imgs = img
            else:
                torch_imgs = torch.cat((torch_imgs, img), dim=1)

        objs = []
        torch_imgs = torch_imgs.reshape(1, -1, H, W)
        for i, item in enumerate(objs_info):
            bbox = torch.from_numpy(item[:4]).float()
            feat = roi_align(torch_imgs, [bbox.unsqueeze(dim=0)],
                             output_size=self.feature_size)
            obj = feat.reshape(-1, D, self.feature_size[0], self.feature_size[1])

            # obj加工
            spatial_perm = np.arange(9)
            obj = self.jigsaw(obj, border=2, patch_size=20, permuation=spatial_perm, dropout=False)
            obj = torch.from_numpy(obj)
            perm = np.arange(self.sample_nums)
            obj = obj[:, perm, :, :]
            obj = torch.clamp(obj, 0., 1.)

            # 添加到结果集
            objs.append(obj)
        return objs

    def jigsaw(self, clip, border=2, patch_size=20, permuation=None, dropout=False):
        patch_list = self.split_image(clip, border, patch_size)
        clip = self.concat(patch_list, border=border, patch_size=patch_size,
                           permuation=permuation, num=3, dropout=dropout)
        return clip

    def split_image(self, clip, border=2, patch_size=20):
        """
        image: (C, T, H, W)
        """
        patch_list = []

        for i in range(3):
            for j in range(3):
                y_offset = border + patch_size * i
                x_offset = border + patch_size * j
                patch_list.append(clip[:, :, y_offset: y_offset + patch_size, x_offset: x_offset + patch_size])

        return patch_list

    def concat(self, patch_list, border=2, patch_size=20, permuation=np.arange(9), num=3, dropout=False):
        """
        batches: [(C, T, h1, w1)]
        """
        clip = np.zeros((3, self.sample_nums, 64, 64), dtype=np.float32)
        drop_ind = random.randint(0, len(permuation) - 1)
        for p_ind, i in enumerate(permuation):
            if drop_ind == p_ind and dropout:
                continue
            y = i // num
            x = i % num
            y_offset = border + patch_size * y
            x_offset = border + patch_size * x
            clip[:, :, y_offset: y_offset + patch_size, x_offset: x_offset + patch_size] = patch_list[p_ind]
        return clip

    def inference(self, frames, objs_info):
        """
        输入图像，返回分数
        :param frames: 输入图像 ndarray(n,h,w,3) BGR n由算法决定
        :param objs_info: 对象信息 List[ndarray(n,7)]
        # [0,1,2,3]: ltrb bboxes (基于视频流分辨率)
        #   [0]: x1
        #   [1]: y1
        #   [2]: x2
        #   [3]: y2
        # [4]: 置信度
        # [5]: 类别 (下标从0开始)
        # [6]: id  （可选）
        """
        if frames is None:
            return [1, 1]
        if objs_info is None or len(objs_info) == 0:
            return [1, 1]
        objs = self.crop_objs(frames, objs_info)
        objs = torch.stack(objs).to(self.config.vad_jigsaw_device)  # list->ndarray
        with torch.no_grad():
            temp_logits, spat_logits = self.model(objs)
            temp_logits = temp_logits.view(-1, self.sample_nums, self.sample_nums)
            spat_logits = spat_logits.view(-1, 9, 9)
        # 空间分数
        spat_probs = F.softmax(spat_logits, -1)
        diag = torch.diagonal(spat_probs, offset=0, dim1=-2, dim2=-1)
        scores = diag.min(-1)[0].cpu().numpy()
        # 时间分数
        temp_probs = F.softmax(temp_logits, -1)
        diag2 = torch.diagonal(temp_probs, offset=0, dim1=-2, dim2=-1)
        scores2 = diag2.min(-1)[0].cpu().numpy()
        # 只返回帧内最异常对象的分数
        final_score = (np.min(scores).item(), np.min(scores2).item())
        # final_score = np.min(scores).item() + np.min(scores2) * 0.5
        # final_scores -= final_scores.min()
        # final_scores /= final_scores.max()
        return final_score

    def inference_batch(self, frames_batch, objs_info_batch):
        """
        输入图像batch，返回分数
        :param frames_batch: 输入图像 List[ndarray(n,h,w,3)] BGR n由算法决定 len=batch
        :param objs_info_batch: 对象信息 List[ndarray(n,7)]
        # [0,1,2,3]: ltrb bboxes (基于视频流分辨率)
        #   [0]: x1
        #   [1]: y1
        #   [2]: x2
        #   [3]: y2
        # [4]: 置信度
        # [5]: 类别 (下标从0开始)
        # [6]: id  （可选）
        """
        pass
