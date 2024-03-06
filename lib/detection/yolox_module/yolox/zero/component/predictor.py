import os.path as osp
import cv2
import torch

from loguru import logger
from yolox.data.data_augment import preproc
from yolox.models import YOLOX
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.zero.info.yolox_info import YoloxInfo


class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            trt_file=None,
            decoder=None,
            device=torch.device("cpu"),
            fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            if timer is not None:
                timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            # logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info


def create_zero_predictor(info: YoloxInfo, exp, pname=""):
    if info.yolox_args_trt:
        info.yolox_args_device = "gpu"
    info.yolox_args_device = torch.device("cuda" if info.yolox_args_device == "gpu" else "cpu")
    logger.info("{} Args: {}".format(pname, info.__dict__))

    # 设置模型参数
    if info.yolox_args_conf is not None:
        exp.test_conf = info.yolox_args_conf
    if info.yolox_args_nms is not None:
        exp.nmsthre = info.yolox_args_nms
    if info.yolox_args_tsize is not None:
        exp.test_size = (info.yolox_args_tsize, info.yolox_args_tsize)

    # 初始化模型
    model: YOLOX = exp.get_model().to(info.yolox_args_device)  # 初始化yolox模型
    logger.info("{} Model Summary: {}".format(pname, get_model_info(model, exp.test_size)))
    model.eval()

    if not info.yolox_args_trt:
        # if args.ckpt is None:
        #     ckpt_file = osp.join(output_dir, "yolox_s.pth.tar")
        # else:
        ckpt_file = info.yolox_args_ckpt
        logger.info(f"{pname} loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info(f"{pname} loaded checkpoint done.")

    if info.yolox_args_fuse:
        logger.info(f"{pname} Fusing model...")
        model = fuse_model(model)

    if info.yolox_args_fp16:
        model = model.half()

    if info.yolox_args_trt:
        assert not info.yolox_args_fuse, "TensorRT model is not support model fusing!"
        # trt_file = osp.join(output_dir, "model_trt.pth")
        trt_file = osp.join(osp.dirname(info.yolox_args_ckpt), "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 bytetrack/face.py first!"
        # 使用TensorRT时，模型的输出可能已经被TensorRT处理过，因此不需要再次解码。所以，这里设置为False，避免在推理时发生错误。
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info(f"{pname} Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    # yolox模型
    return Predictor(model, exp, trt_file, decoder, info.yolox_args_device, info.yolox_args_fp16)


def create_predictor(args, exp):
    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    # 初始化模型
    model = exp.get_model().to(args.device)  # 初始化yolox模型
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        # if args.ckpt is None:
        #     ckpt_file = osp.join(output_dir, "yolox_s.pth.tar")
        # else:
        ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        # trt_file = osp.join(output_dir, "model_trt.pth")
        trt_file = osp.join(osp.dirname(args.ckpt), "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 bytetrack/face.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    # yolox模型
    return Predictor(model, exp, trt_file, decoder, args.device, args.fp16)