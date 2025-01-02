import os
import torch
from PIL import Image
from torchvision.transforms import transforms

from clip_reid.datasets.make_dataloader_clipreid import make_dataloader
from clip_reid.model.make_model_clipreid import make_model
from clip_reid.utils.logger import setup_logger
from clip_reid.config import cfg
from reid_core.i_reid_wrapper import IReidWrapper
from reid_core.reid_info import ReidInfo


class ClipReidWrapper(IReidWrapper):
    def __init__(self, config: ReidInfo):
        self.config = config
        self.model, self.logger, self.device = self.make_model()

    def make_model(self):
        if self.config.reid_config_file != "":
            cfg.merge_from_file(self.config.reid_config_file)
        # cfg.merge_from_list(args.opts)
        cfg.freeze()
        output_dir = cfg.OUTPUT_DIR
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        logger = setup_logger("transreid", output_dir, if_train=False)
        if self.config.reid_config_file != "":
            logger.info("Loaded configuration file {}".format(self.config.reid_config_file))
            with open(self.config.reid_config_file, 'r') as cf:
                config_str = "\n" + cf.read()
                logger.info(config_str)
        logger.info("Running with config:\n{}".format(cfg))

        train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(
            cfg)

        model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
        model.load_param(cfg.TEST.WEIGHT)

        # 获取模型的设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        return model, logger, device

    def inference(self, img):
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
        return ret
