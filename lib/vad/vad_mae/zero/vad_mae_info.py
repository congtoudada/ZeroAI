from zero.info.base_info import BaseInfo


class VadMaeInfo(BaseInfo):
    def __init__(self, data: dict = None):
        self.vad_mae_ckpt_folder = "pretrained/vad/vad_mae/avenue"  # ckpt目录(结尾勿带/)
        self.vad_mae_device = "cuda"
        self.vad_mae_mask_ratio = 0.5
        self.vad_mae_input_size = [320, 640]
        self.vad_mae_norm_pix_loss = False
        self.vad_mae_use_only_masked_tokens_ab = False
        self.vad_mae_abnormal_score_func = 'L2'
        self.vad_mae_masking_method = "random_masking"
        self.vad_mae_grad_weighted_rec_loss = True
        super().__init__(data)  # 前面是声明，一定要最后调用这段赋值


