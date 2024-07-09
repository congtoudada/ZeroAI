import numpy as np
import cv2
from PIL import Image
from PIL import ImageDraw


class ImgKit:
    @staticmethod
    def crop_img(im, ltrb):
        x1, y1, x2, y2 = ltrb[0], ltrb[1], ltrb[2], ltrb[3]
        return np.ascontiguousarray(np.copy(im[int(y1): int(y2), int(x1): int(x2)]))

    @staticmethod
    def crop_img_border(im, ltrb, border=0):
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


class ImgKit_img_box:
    @staticmethod
    def draw_img_box(im, ltrb):
        x1, y1, x2, y2 = ltrb[0], ltrb[1], ltrb[2], ltrb[3]
        xy = ((x1, y1), (x2, y2))
        im_pil = Image.fromarray(im)

        # 创建一个可以在图像上绘图的对象
        draw = ImageDraw.Draw(im_pil)

        # 使用给定的坐标绘制矩形
        draw.rectangle(xy, outline='red', width=5)

        # 将PIL图像转换回numpy数组
        im_with_rectangle = np.array(im_pil)
        return im_with_rectangle

