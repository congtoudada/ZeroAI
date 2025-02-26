import os
import cv2
import numpy as np
import torch

from utility.config_kit import ConfigKit
from vad.vad_mae.model.model_factory import mae_cvt_patch16
from vad.vad_mae.zero.vad_mae_info import VadMaeInfo
from vad_core.i_vad_frame_wrapper import IVadFrameWrapper


class VadMaeHelper(IVadFrameWrapper):
    """
    帧级别视频异常检测器
    """

    def __init__(self, config_path: str):
        self.config: VadMaeInfo = VadMaeInfo(ConfigKit.load(config_path))
        self.pname = f"[ {os.getpid()}:vad_mae ]"
        self.model = mae_cvt_patch16(norm_pix_loss=self.config.vad_mae_norm_pix_loss,
                                     img_size=self.config.vad_mae_input_size,
                                     use_only_masked_tokens_ab=self.config.vad_mae_use_only_masked_tokens_ab,
                                     abnormal_score_func=self.config.vad_mae_abnormal_score_func,
                                     masking_method=self.config.vad_mae_masking_method,
                                     grad_weighted_loss=self.config.vad_mae_grad_weighted_rec_loss).float()
        self.on_start()

    def on_start(self):
        self.model.to(self.config.vad_mae_device)
        student = torch.load(self.config.vad_mae_ckpt_folder + "/checkpoint-best-student.pth")['model']
        teacher = torch.load(self.config.vad_mae_ckpt_folder + "/checkpoint-best.pth")['model']
        for key in student:
            if 'student' in key:
                teacher[key] = student[key]
        self.model.load_state_dict(teacher, strict=False)
        self.model.eval()
        self.model.train_TS = True

    def preprocess(self, frames):
        """
        :param frames: ndarray(n=5,h,w,3)
        Returns:
        """
        siz = frames.shape[0]
        assert siz >= 3, f"Size must be at least 3, but got {siz}"
        target = frames[int(siz / 2)]
        previous_img = frames[int(siz / 2) - 1]  # (h, w, 3)
        previous_img = previous_img.astype(np.int32)
        next_img = frames[int(siz / 2) + 1]  # (h, w, 3)
        next_img = next_img.astype(np.int32)
        gradient = np.abs(previous_img - next_img)
        gradient = gradient.astype(np.uint8)

        frames = np.concatenate([frames[0], frames[int(siz / 2)], frames[int(siz - 1)]], axis=-1)
        # (h, w, c)
        if frames.shape[:2] != tuple(self.config.vad_mae_input_size):
            frames = cv2.resize(frames, self.config.vad_mae_input_size[::-1])
            target = cv2.resize(target, self.config.vad_mae_input_size[::-1])
            gradient = cv2.resize(gradient, self.config.vad_mae_input_size[::-1])
        mask = np.zeros((frames.shape[0], frames.shape[1], 1), dtype=np.uint8)
        target = np.concatenate((target, mask), axis=-1)

        frames = frames.astype(np.float32)
        gradient = gradient.astype(np.float32)
        target = target.astype(np.float32)
        frames = (frames - 127.5) / 127.5
        target = (target - 127.5) / 127.5

        frames = np.swapaxes(frames, 0, -1).swapaxes(1, -1)
        frames = torch.tensor(frames).unsqueeze(0).to(self.config.vad_mae_device)
        gradient = np.swapaxes(gradient, 0, 1).swapaxes(0, -1)
        gradient = torch.tensor(gradient).unsqueeze(0).to(self.config.vad_mae_device)
        target = np.swapaxes(target, 0, -1).swapaxes(1, -1)
        target = torch.tensor(target).unsqueeze(0).to(self.config.vad_mae_device)
        return frames, gradient, target

    def inference_batch(self, frames_batch):
        """

        Args:
            frames_batch: List[ndarray(n, h, w, 3)] len=batch

        Returns:

        """
        if frames_batch is None or len(frames_batch) == 0:
            return []
        samples = []
        grads = []
        targets = []
        for i in range(len(frames_batch)):
            frames, grad, target = self.preprocess(frames_batch[i])
            samples.append(frames)
            grads.append(grad)
            targets.append(target)
        samples = torch.stack(samples, dim=0).squeeze(1)
        grads = torch.stack(grads, dim=0).squeeze(1)
        targets = torch.stack(targets, dim=0).squeeze(1)
        with torch.no_grad():
            _, _, _, recon_error_st_tc = self.model(samples, targets=targets, grad_mask=grads,
                                                    mask_ratio=self.config.vad_mae_mask_ratio)
            pred_teacher = recon_error_st_tc[0].detach().cpu().numpy()
            pred_student = recon_error_st_tc[1].detach().cpu().numpy()
            pred_cls = recon_error_st_tc[2].detach().cpu().numpy()
            score = 10.5 * pred_teacher + 5.3 * pred_student + 5.3 * pred_cls
            return list(score)

    def inference(self, frames):
        """
        :param frames: ndarray(n=5,h,w,3)
        Returns:
        """
        frames, grad, target = self.preprocess(frames)
        with torch.no_grad():
            _, _, _, recon_error_st_tc = self.model(frames, targets=target, grad_mask=grad,
                                                    mask_ratio=self.config.vad_mae_mask_ratio)
            pred_teacher = recon_error_st_tc[0].detach().cpu().numpy()
            pred_student = recon_error_st_tc[1].detach().cpu().numpy()
            pred_cls = recon_error_st_tc[2].detach().cpu().numpy()
            score = 1.05 * pred_teacher + 0.53 * pred_student + 0.53 * pred_cls
            return score.item()


if __name__ == '__main__':
    # 创建一个示例数组，形状为 (n, h, w, 3)
    n, h, w = 5, 320, 640
    arr = np.random.randn(n, h, w, 3)

    # 通过 reshape 和 transpose 转换为 (h, w, 9)
    arr_reshaped = arr.reshape(n, h, w, 3).transpose(1, 2, 0, 3).reshape(h, w, -1)
    arr_reshaped = (arr_reshaped - 127.5) / 127.5
    arr_reshaped = np.swapaxes(arr_reshaped, 0, -1).swapaxes(1, -1)

    print(arr_reshaped.shape)  # 输出 (h, w, 9)

    input_size = [320, 640]
    print(input_size[::-1])

    vmh = VadMaeHelper("")
    x1, x2, x3 = vmh.preprocess(arr)
    print(x1.shape)
    print(x2.shape)
    print(x3.shape)

    print(vmh.inference(arr))

    b, n, h, w = 2, 5, 320, 640
    arr = np.random.randn(b, n, h, w, 3)
    res = vmh.inference_batch(arr)
    print(res)
