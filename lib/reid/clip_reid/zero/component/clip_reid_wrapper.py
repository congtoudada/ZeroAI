import os
import faiss
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms

from clip_reid.datasets.make_dataloader_clipreid import make_dataloader
from clip_reid.model.make_model_clipreid import make_model
from clip_reid.utils.logger import setup_logger
from clip_reid.zero.info.clip_reid_info import ClipReidInfo
from clip_reid.config import cfg


class ClipReidWrapper(object):
    def __init__(self, config: ClipReidInfo):
        self.config = config
        self.model, self.logger, self.device = self.make_model()

    def make_model(self):
        if self.config.clip_reid_config_file != "":
            cfg.merge_from_file(self.config.clip_reid_config_file)
        # cfg.merge_from_list(args.opts)
        cfg.freeze()
        output_dir = cfg.OUTPUT_DIR
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        logger = setup_logger("transreid", output_dir, if_train=False)
        if self.config.clip_reid_config_file != "":
            logger.info("Loaded configuration file {}".format(self.config.clip_reid_config_file))
            with open(self.config.clip_reid_config_file, 'r') as cf:
                config_str = "\n" + cf.read()
                logger.info(config_str)
        logger.info("Running with config:\n{}".format(cfg))

        train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(
            cfg)

        model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
        model.load_param(cfg.TEST.WEIGHT)

        # 获取模型的设备
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        return model, logger, device

    def extract_feature(self, img):
        """
        抽特征，输入必须是 ndarray (w,h,3) 0-255
        """
        if not img.ndim == 3 or not img.shape[2] == 3:
            self.logger.error("Unsupported image shape: {}".format(img.shape))
        #     # 确保数值范围是0-255，转换为uint8类型
        #     img = np.clip(img, 0, 255).astype(np.uint8)
            return None

        pil_img = Image.fromarray(img).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize(cfg.INPUT.SIZE_TEST),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ])

        img_tensor = preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(img_tensor)

        ret = feat.to('cpu').detach().numpy()
        faiss.normalize_L2(ret)
        return ret
