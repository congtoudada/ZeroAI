# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: python-learning-notes
# @Author :
# @E-mail :
# @Date   : 2018-08-06 17:19:21
# --------------------------------------------------------
"""
import numpy as np
import cv2
from pybaseutils import image_utils


def point_affine_transform(point, trans):
    """
    输入原坐标点，进行仿射变换，获得变换后的坐标
    :param point: 输入坐标点 (x,y)
    :param trans: 仿射变换矩阵shape=(2,3),通过OpenCV的estimateAffine2D或者estimateAffine2D获得
    :return: 变换后的新坐标
    """
    new_point = np.array([point[0], point[1], 1.]).T
    new_point = np.dot(trans, new_point)  # 矩阵相乘
    return new_point[:2]


def image_affine_transform(image, dsize, trans):
    """
    输入原始图像，进行仿射变换，获得变换后的图像
    :param image: 输入图像
    :param dsize: 输入目标图像大小
    :param trans: 仿射变换矩阵shape=(2,3),通过OpenCV的estimateAffine2D或者estimateAffine2D获得
    :return:
    """
    out_image = cv2.warpAffine(image, M=trans, dsize=tuple(dsize))
    return out_image


def get_affine_transform(kpts, kpts_ref, trans_type="estimate"):
    """
    估计最优的仿射变换矩阵
    :param kps: 实际关键点
    :param kpts_ref: 参考关键点
    :param trans_type:变换类型
    :return: 仿射变换矩阵
    """
    kpts = np.float32(kpts)
    kpts_ref = np.float32(kpts_ref)
    if trans_type == "estimate":
        # estimateAffine2D()可以用来估计最优的仿射变换矩阵
        trans, _ = cv2.estimateAffine2D(kpts, kpts_ref)
    elif trans_type == "affine":
        # 通过3点对应关系获仿射变换矩阵
        trans = cv2.getAffineTransform(kpts[0:3], kpts_ref[0:3])
    else:
        raise Exception("Error:{}".format(trans_type))
    return trans


def alignment_and_crop_face(image, face_size, kpts, kpts_ref=None, align_type="estimate"):
    """
    apply affine transform
    :param image: input image
    :param face_size: out face size
    :param kpts: face landmark,shape=(5, 2).float32
    :param kpts_ref: reference facial points

    :param align_type: transform type, could be one of
            1) 'affine': use the first 3 points to do affine transform,by calling cv2.getAffineTransform()
            2) 'estimate': use all points to do affine transform
    :return:
    """

    kpts = np.float32(kpts)
    kpts_ref = np.float32(kpts_ref)
    if align_type == "estimate":
        # estimateAffine2D()可以用来估计最优的仿射变换矩阵
        retval, _ = cv2.estimateAffine2D(kpts, kpts_ref)
    elif align_type == "affine":
        # 通过3点对应关系获仿射变换矩阵
        retval = cv2.getAffineTransform(kpts[0:3], kpts_ref[0:3])
    else:
        raise Exception("Error:{}".format(align_type))
    # 进行仿射变换
    face_alig = cv2.warpAffine(image, M=retval, dsize=tuple(face_size))
    return face_alig


def get_reference_facial_points(square=True, vis=False):
    """
    获得人脸参考关键点,目前支持两种输入的参考关键点,即[96, 112]和[112, 112]
    face_size_ref = [96, 112]
    kpts_ref = [[30.29459953, 51.69630051],
                [65.53179932, 51.50139999],
                [48.02519989, 71.73660278],
                [33.54930115, 92.3655014],
                [62.72990036, 92.20410156]]
    ==================
    face_size_ref = [112, 112]
    kpts_ref = [[38.29459953 51.69630051]
                [73.53179932 51.50139999]
                [56.02519989 71.73660278]
                [41.54930115 92.3655014 ]
                [70.72990036 92.20410156]]

    ==================
    square = True, crop_size = (112, 112)
    square = False,crop_size = (96, 112),
    :param square: True is [112, 112] or False is [96, 112]
    :param vis: True or False,是否显示
    :return:
    """
    # face size[96_112] reference facial points
    face_size_ref = [96, 112]
    kpts_ref = [[30.29459953, 51.69630051],
                [65.53179932, 51.50139999],
                [48.02519989, 71.73660278],
                [33.54930115, 92.3655014],
                [62.72990036, 92.20410156]]
    kpts_ref = np.asarray(kpts_ref)  # kpts_ref_96_112
    # for output_size=[112, 112]
    if square:
        face_size_ref = np.array(face_size_ref)
        size_diff = max(face_size_ref) - face_size_ref
        kpts_ref += size_diff / 2
        face_size_ref += size_diff

    if vis:
        tmp = np.zeros(shape=(face_size_ref[1], face_size_ref[0], 3), dtype=np.uint8)
        tmp = image_utils.draw_landmark(tmp, [kpts_ref], vis_id=False)
        cv2.imshow("ref-Landmark", tmp)
        cv2.waitKey(0)
    return kpts_ref
