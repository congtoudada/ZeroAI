# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author :
# @E-mail :
# @Date   : 2018-06-09 15:25:36
# --------------------------------------------------------
"""

import os, sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import cv2
from alignment import cv_face_alignment
from pybaseutils import image_utils


def show_landmark_boxes(title, image, landmarks, boxes, color=(0, 255, 0)):
    '''
    显示landmark和boxes
    :param title:
    :param image:
    :param landmarks: [[x1, y1], [x2, y2]]
    :param boxes:     [[ x1, y1, x2, y2],[ x1, y1, x2, y2]]
    :return:
    '''
    point_size = 1
    thickness = 4  # 可以为 0 、4、8
    for lm in landmarks:
        for landmark in lm:
            # 要画的点的坐标
            point = (int(landmark[0]), int(landmark[1]))
            cv2.circle(image, point, point_size, color, thickness * 2)
    for box in boxes:
        x1, y1, x2, y2 = box
        point1 = (int(x1), int(y1))
        point2 = (int(x2), int(y2))
        cv2.rectangle(image, point1, point2, color, thickness=thickness)
    image_utils.cv_show_image(title, image, delay=0)


def face_alignment(image, landmarks, vis=False):
    """
    face alignment and crop face ROI
    :param image:输入RGB/BGR图像
    :param landmarks:人脸关键点landmarks(5个点)
    :param vis: 可视化矫正效果
    :return:
    """
    output_size = [112, 112]
    alig_faces = []
    kpts_ref = cv_face_alignment.get_reference_facial_points(square=True, vis=vis)
    # kpts_ref = align_trans.get_reference_facial_points(output_size, default_square=True)
    for landmark in landmarks:
        warped_face = cv_face_alignment.alignment_and_crop_face(np.array(image), output_size, kpts=landmark,
                                                                kpts_ref=kpts_ref)
        alig_faces.append(warped_face)
    if vis:
        for face in alig_faces: image_utils.cv_show_image("face_alignment", face)
    return alig_faces


if __name__ == "__main__":
    image_file = "test.jpg"
    image = cv2.imread(image_file)
    # face detection from MTCNN
    boxes = np.asarray([[200.27724761, 148.9578526, 456.70521605, 473.52968433]])
    landmarks = np.asarray([[[287.86636353, 306.13598633],
                             [399.58618164, 272.68032837],
                             [374.80252075, 360.95596313],
                             [326.71264648, 409.12332153],
                             [419.06210327, 381.41421509]]])
    alig_faces = face_alignment(image, landmarks, vis=True)
    # show bbox and bounding boxes
    show_landmark_boxes("image", np.array(image), landmarks, boxes)
