# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project:
# @Author : Pan
# @E-mail : 390737991@qq.com
# @Date   : 2018-12-31 09:11:25
# --------------------------------------------------------
"""
import numpy as np
from pybaseutils import image_utils
from models import box_utils, onet_landmark


class ONetLandmarkDet(onet_landmark.ONetLandmark):
    def __init__(self, device):
        super(ONetLandmarkDet, self).__init__(onet_path=None, device=device)

    def get_image_crop(self, bounding_boxes, image, size=48):
        if not isinstance(image, np.ndarray):
            rgb_image = np.asarray(image)
        else:
            rgb_image = image
        # resize
        bboxes = bounding_boxes[:, :4]
        scores = bounding_boxes[:, 4:]
        num_boxes = len(bboxes)
        img_boxes = np.zeros((num_boxes, 3, size, size), 'float32')
        for i, box in enumerate(bboxes):
            img_box = image_utils.get_bboxes_image(rgb_image, [box], size=size)
            img_box = img_box[0]
            img_boxes[i, :, :, :] = box_utils._preprocess(img_box)
        return img_boxes

    def face_alignment(self, faces, face_resize):
        landmarks = self.get_faces_landmarks(faces)
        faces = align_trans.get_alignment_faces_list_resize(faces, landmarks, face_size=face_resize)
        # image_processing.cv_show_image("image", faces[0])
        return faces


if __name__ == "__main__":
    pass
