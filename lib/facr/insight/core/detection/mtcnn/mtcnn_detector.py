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
import cv2
import numpy as np
import models.mtcnn_model as mtcnn
from pybaseutils import image_utils, file_utils


class MTCNNDetector():
    def __init__(self,
                 input_size: list = [320, None],
                 conf_thresh=[0.75, 0.85, 0.95],
                 iou_thresh=[0.7, 0.7, 0.7],
                 device="cuda:0"):
        self.input_size = input_size
        min_face_size = 20.0
        self.detecror = mtcnn.MTCNN(min_face_size, conf_thresh, iou_thresh, device=device)

    def detect(self, bgr, vis=False):
        """
        @param bgr:
        @param vis:
        @return:
        """
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width, _ = bgr.shape
        input = image_utils.resize_image(rgb, size=tuple(self.input_size))
        inputH, inputW = input.shape[:2]
        bbox_score, landms = self.detecror.detect_image(input)
        if len(bbox_score) > 0:
            boxe_scale = [width / inputW, height / inputH] * 2 + [1.0]
            land_scale = [width / inputW, height / inputH] * 5
            land_scale = np.asarray(land_scale).reshape(5, 2)
            bbox_score = bbox_score * boxe_scale
            landms = landms * land_scale
            bboxes = bbox_score[:, 0:4]
            scores = bbox_score[:, 4:5]
        else:
            bboxes, scores, landms = np.array([]), np.array([]), np.array([])
        if vis: self.show_landmark_boxes("Det", bgr, bboxes, scores, landms)
        return bboxes, scores, landms

    def detect_image_dir(self, image_dir, vis=True):
        image_list = file_utils.get_files_lists(image_dir)
        for image_file in image_list:
            image = cv2.imread(image_file)
            boxes, scores, landms = self.detect(image, vis=vis)
            print("boxes:\n{}\nscores:\n{}\nlandms:\n{}".format(boxes, scores, landms))

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
        image = image_utils.draw_landmark(image, landms, radius=3, vis_id=False)
        text = scores.reshape(-1).tolist()
        text = ["{:3.3f}".format(t) for t in text]
        image = image_utils.draw_image_bboxes_text(image, bboxes, text, thickness=2, fontScale=1.0, color=(255, 0, 0))
        image_utils.cv_show_image(title, image)
        return image


if __name__ == "__main__":
    image_dir = "test.jpg"
    image_dir = "test_image"
    mt = MTCNNDetector(device="cpu")
    mt.detect_image_dir(image_dir)
