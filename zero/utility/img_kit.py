import numpy as np


class ImgKit:
    @staticmethod
    def crop_img(self, im, tlbr):
        x1, y1, x2, y2 = tlbr[0], tlbr[1], tlbr[2], tlbr[3]
        return np.ascontiguousarray(np.copy(im[int(y1): int(y2), int(x1): int(x2)]))

    @staticmethod
    def crop_img_border(self, im, tlbr, border=0):
        x1, y1, w, h = tlbr[0], tlbr[1], tlbr[2] - tlbr[0], tlbr[3] - tlbr[1]
        x2 = x1 + w + border
        x1 = x1 - border
        y2 = y1 + h + border
        y1 = y1 - border
        x1 = 0 if x1 < 0 else x1
        y1 = 0 if y1 < 0 else y1
        x2 = im.shape[1] if x2 > im.shape[1] else x2
        y2 = im.shape[0] if y2 > im.shape[0] else y2
        return np.ascontiguousarray(np.copy(im[int(y1): int(y2), int(x1): int(x2)]))