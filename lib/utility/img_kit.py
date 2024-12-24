import numpy as np
import cv2
from PIL import Image
from PIL import ImageDraw


class ImgKit:
    @staticmethod
    def crop_img(im, ltrb):
        """
        裁剪图片
        """
        x1, y1, x2, y2 = ltrb[0], ltrb[1], ltrb[2], ltrb[3]
        if x1 < 0 or y1 < 0 or x2 > im.shape[1] or y2 > im.shape[0]:
            return im
        if x1 > x2 or y1 > y2:
            return im
        return np.ascontiguousarray(np.copy(im[int(y1): int(y2), int(x1): int(x2)]))

    @staticmethod
    def crop_img_border(im, ltrb, border=0):
        """
        裁剪图片（带border）
        """
        x1, y1, w, h = ltrb[0], ltrb[1], ltrb[2] - ltrb[0], ltrb[3] - ltrb[1]
        x2 = x1 + w + border
        x1 = x1 - border
        y2 = y1 + h + border
        y1 = y1 - border
        x1 = 0 if x1 < 0 else x1
        y1 = 0 if y1 < 0 else y1
        x2 = im.shape[1] if x2 > im.shape[1] else x2
        y2 = im.shape[0] if y2 > im.shape[0] else y2
        return np.ascontiguousarray(np.copy(im[int(y1): int(y2), int(x1): int(x2)]))

    @staticmethod
    def draw_img_box(im, ltrb, color='red', line_thickness=1):
        """
        画包围框
        """
        if im is not None:
            if ltrb[0] < 0 or ltrb[1] > 0 or ltrb[2] > im.shape[1] or ltrb[3] > im.shape[0]:
                return
            cv2.rectangle(im, pt1=(int(ltrb[0]), int(ltrb[1])), pt2=(int(ltrb[2]), int(ltrb[3])),
                          color=color, thickness=line_thickness)

