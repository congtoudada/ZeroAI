# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author :
# @E-mail :
# @Date   : 2018-04-03 18:38:34
# --------------------------------------------------------
"""
import os, sys

sys.path.insert(0, os.path.dirname(__file__))

import torch
import cv2
import numpy as np
from models.config.config import cfg_mnet, cfg_slim, cfg_rfb
from models.nets.retinaface import RetinaFace
from models.nets.net_slim import Slim
from models.nets.net_rfb import RFB
from models.layers.functions.prior_box import PriorBox
from models.layers.box_utils import decode, decode_landm
from models.nms.py_cpu_nms import py_cpu_nms
from pybaseutils import image_utils, file_utils
from loguru import logger

root = os.path.dirname(__file__)


class UltraLightFaceDetector(object):
    def __init__(self,
                 model_file: str = "",
                 net_name: str = "RFB",
                 input_size: list = [320, None],
                 conf_thresh: float = 0.5,
                 iou_thresh: float = 0.3,
                 top_k: int = 500,
                 keep_top_k: int = 750,
                 device="cuda:0"):
        """
        :param model_file: model file
        :param net_name:"RFB",
        :param input_size:input_size,
        :param network: Backbone network mobile0.25 or slim or RFB
        :param conf_thresh: confidence_threshold
        :param iou_thresh: nms_threshold
        :param top_k: keep top_k results. If k <= 0, keep all the results.
        :param keep_top_k:
        :param device:
        """
        self.pname = f"[ {os.getpid()}:face_detector ]"
        self.model_file = model_file if model_file else os.path.join(root, "pretrained/pth/face_detection_rbf.pth")
        self.device = device
        self.network = net_name
        self.conf_threshold = conf_thresh
        self.iou_threshold = iou_thresh
        self.top_k = top_k
        self.keep_top_k = keep_top_k
        self.input_size = input_size
        self.cfg = self.get_model_cfg(self.network)
        self.net = self.build_model(self.cfg, self.network, self.model_file)
        torch.set_grad_enabled(False)
        logger.info(f'{self.pname} Finished loading model!')

    def build_model(self, cfg: dict, network: str, model_path: str):
        """
        :param cfg: <dict> model config
        :param network: mobile0.25,slim or RFB
        :param model_path: model path
        :return:
        """
        net = None
        if network == "mobile0.25":
            net = RetinaFace(cfg=cfg, phase='test')
        elif network == "slim":
            net = Slim(cfg=cfg, phase='test')
        elif network == "RFB":
            net = RFB(cfg=cfg, phase='test')
        else:
            logger.info(f"{self.pname} Don't support network!")
            exit(0)
        net = self.load_model(net, model_path)
        net = net.to(self.device)
        net.eval()
        return net

    def get_model_cfg(self, network: str):
        """
        get model config
        :param network: mobile0.25,slim or RFB
        :return:
        """
        if network == "mobile0.25":
            cfg = cfg_mnet
        elif network == "slim":
            cfg = cfg_slim
        elif network == "RFB":
            cfg = cfg_rfb
        else:
            logger.info(f"{self.pname} Don't support network!")
            exit(0)
        return cfg

    def load_model(self, model, model_path: str):
        """
        :param model: model
        :param model_path: model file
        :param load_to_cpu:
        :return:
        """
        logger.info('{} Loading pretrained model from {}'.format(self.pname, model_path))
        pretrained_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(pretrained_dict, strict=False)
        return model

    def pre_process(self, image: np.ndarray, input_size: list, img_mean=(104, 117, 123)):
        """
        :param image:
        :param input_size: model input size [W,H]
        :param img_mean:
        :return:image_tensor: out image tensor[1,channels,W,H]
                input_size  : model new input size [W,H]
        """
        if not input_size: input_size = [None, None]
        out_image = image_utils.resize_image(image, size=tuple(input_size))
        shape = out_image.shape
        input_size = [shape[1], shape[0]]
        out_image = np.float32(out_image)
        out_image -= img_mean
        out_image = out_image.transpose(2, 0, 1)
        image_tensor = torch.from_numpy(out_image).unsqueeze(0)
        return image_tensor, input_size

    @staticmethod
    def get_priorbox(cfg, input_size):
        """
        :param cfg: model config
        :param input_size: model input size [W,H]
        :return:
        """
        # get priorbox
        # input_size=[W,H]-->image_size=[H,W]
        priorbox = PriorBox(cfg, image_size=(input_size[1],  # Height
                                             input_size[0]  # width
                                             ))
        priors = priorbox.forward()
        # priors = priors.to(self.device)
        # get boxes and scores
        prior_data = priors.data
        return prior_data

    def pose_process(self, loc, conf, landms, image_size, input_size, variance):
        """
        :param loc:
        :param conf:
        :param landms:
        :param image_size: input orig-image size [W,H]
        :param input_size: model input size [W,H]
        :return:
        """
        priorbox = self.get_priorbox(self.cfg, input_size=input_size)
        priorbox = priorbox.to(self.device)
        # logger.info("input size:{}".format(self.input_size))
        # logger.info("priorbox:{}".format(priorbox.shape))

        bboxes_scale = np.asarray(image_size * 2, dtype=np.float32)
        landms_scale = np.asarray(image_size * 5, dtype=np.float32)
        # get boxes
        boxes = decode(loc.data.squeeze(0), priorbox, variance)
        boxes = boxes.cpu().numpy()
        boxes = boxes * bboxes_scale
        # get scores
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        # get landmarks
        landms = decode_landm(landms.data.squeeze(0), priorbox, variance)
        landms = landms.cpu().numpy()
        landms = landms * landms_scale
        dets, landms = self.nms_process(boxes,
                                        scores,
                                        landms,
                                        conf_threshold=self.conf_threshold,
                                        iou_threshold=self.iou_threshold,
                                        top_k=self.top_k,
                                        keep_top_k=self.keep_top_k)
        return dets, landms

    @staticmethod
    def nms_process(boxes, scores, landms, conf_threshold, iou_threshold, top_k, keep_top_k):
        """
        :param boxes: face boxes, (xmin,ymin,xmax,ymax)
        :param scores:scores
        :param landms: face landmark
        :param conf_threshold:
        :param iou_threshold:
        :param top_k:keep top_k results. If k <= 0, keep all the results.
        :param keep_top_k:
        :return: dets:shape=(num_bboxes,5),[xmin,ymin,xmax,ymax,scores]
                 landms:(num_bboxes,10),[x0,y0,x1,y1,...,x4,y4]
        """
        # ignore low scores
        inds = np.where(scores > conf_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]
        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, iou_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]
        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]
        return dets, landms

    def inference(self, img_tensor):
        with torch.no_grad():
            img_tensor = img_tensor.to(self.device)
            loc, conf, landms = self.net(img_tensor)  # forward pass
        return loc, conf, landms

    def adapter_bbox_score_landmarks(self, dets, landms):
        if len(dets) > 0:
            landms = landms.reshape(len(landms), -1, 2)
            bboxes = dets[:, 0:4]
            scores = dets[:, 4:5]
            # bboxes = self.get_square_bboxes(bboxes,fixed="H")
        else:
            bboxes, scores, landms = np.array([]), np.array([]), np.array([])
        return bboxes, scores, landms

    @staticmethod
    def get_square_bboxes(bboxes, fixed="W"):
        '''
        :param bboxes:
        :param fixed: (W)width (H)height
        :return:
        '''
        new_bboxes = []
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = bbox
            w = xmax - xmin
            h = ymax - ymin
            cx, cy = (int((xmin + xmax) / 2), int((ymin + ymax) / 2))
            if fixed == "H":
                dd = h / 2
            elif fixed == 'W':
                dd = w / 2
            elif fixed > 0:
                dd = h / 2 * fixed
            fxmin = int(cx - dd)
            fymin = int(cy - dd)
            fxmax = int(cx + dd)
            fymax = int(cy + dd)
            new_bbox = (fxmin, fymin, fxmax, fymax)
            new_bboxes.append(new_bbox)
        new_bboxes = np.asarray(new_bboxes)
        return new_bboxes

    def detect(self, bgr, vis=False):
        """
        :param bgr:
        :return:
            bboxes: <np.ndarray>: (num_boxes, 4)
            scores: <np.ndarray>: (num_boxes, 1)
            scores: <np.ndarray>: (num_boxes, 5, 2)
        """
        shape = bgr.shape
        img_tensor, input_size = self.pre_process(bgr, input_size=self.input_size)
        loc, conf, landms = self.inference(img_tensor)
        dets, landms = self.pose_process(loc,
                                         conf,
                                         landms,
                                         image_size=[shape[1], shape[0]],
                                         input_size=input_size,
                                         variance=self.cfg["variance"])
        bboxes, scores, landms = self.adapter_bbox_score_landmarks(dets, landms)
        if vis: self.show_landmark_boxes("Det", bgr, bboxes, scores, landms)
        return bboxes, scores, landms

    @staticmethod
    def show_landmark_boxes(title, image, bboxes, scores, landms):
        """
        显示landmark和boxes
        :param title:
        :param image:
        :param landms: [[x1, y1], [x2, y2]]
        :param bboxes: [[ x1, y1, x2, y2],[ x1, y1, x2, y2]]
        :return:
        """
        image = image_utils.draw_landmark(image, landms, radius=2, vis_id=False)
        text = scores.reshape(-1).tolist()
        text = ["{:3.3f}".format(t) for t in text]
        image = image_utils.draw_image_bboxes_text(image, bboxes, text, thickness=2, fontScale=1.0, color=(255, 0, 0))
        image_utils.cv_show_image(title, image)
        return image


if __name__ == '__main__':
    image_dir = "test.jpg"
    image_dir = "test_image"
    input_size = [320, None]
    device = "cuda:0"
    detector = UltraLightFaceDetector(net_name="RFB",
                                      input_size=input_size,
                                      device=device)
    image_list = file_utils.get_files_lists(image_dir)
    for image_file in image_list:
        image = cv2.imread(image_file)
        bboxes, scores, landms = detector.detect(image, vis=True)
        logger.info("bboxes:\n{}\nscores:\n{}\nlandms:\n{}".format(bboxes, scores, landms))
