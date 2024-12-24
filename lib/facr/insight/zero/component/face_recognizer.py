import torch.cuda
import os
import traceback
import cv2
from typing import Dict, List
import numpy as np
from pybaseutils import file_utils, image_utils
from loguru import logger

from insight.core.face_detector import FaceDetector
from insight.core.face_feature import FaceFeature
from insight.core.face_register import FaceRegister
from insight.zero.info.insight_info import InsightInfo


class FaceRecognizer(object):
    """
    人脸识别算法接口
    """
    def __init__(self, config: InsightInfo):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pname = f"[ {os.getpid()}:insight_face ]"
        # 初始化人脸检测
        self.faceDet = FaceDetector(model_file=self.config.insight_det_ckpt,
                                    net_name=self.config.insight_det_detector,
                                    input_size=[320, None],
                                    conf_thresh=self.config.insight_det_thresh,
                                    device=self.device)
        # 初始化人脸识别特征提取
        self.faceFea = FaceFeature(model_file=self.config.insight_rec_ckpt,
                                   net_name=self.config.insight_rec_feature,
                                   input_size=(self.config.insight_rec_input_size, self.config.insight_rec_input_size),
                                   embedding_size=self.config.insight_rec_embedding_size,
                                   device=self.device)
        self.database_file = os.path.join(os.path.dirname(self.config.insight_database),
                                     "database-{}.json".format(self.config.insight_rec_feature))
        # 初始化人脸数据库，用于注册人脸
        self.faceReg = FaceRegister(self.database_file)
        # 如果不存在就重新建库，加载
        if self.config.insight_force_rebuild or not os.path.exists(self.database_file):
            self.create_database(self.config.insight_database)

    # 自定义
    def save(self):
        self.faceReg.save()

    def search_face_task(self, bgr, thickness=2, fontScale=1.5, delay=0, vis=False):
        """
        1:N人脸搜索任务
        :param bgr: BGR image
        :return:
        """
        face_info = self.detect_search(bgr, max_face=-1, vis=False)
        if vis:
            image = self.draw_result("Recognizer", image=bgr, face_info=face_info,
                                     thickness=thickness, fontScale=fontScale, delay=delay, vis=vis)
        else:
            image = None
        return image, face_info

    def search_face_path(self, image_path, vis=False):
        # image_list = file_utils.get_files_lists(image_dir, postfix=file_utils.IMG_POSTFIX)
        try:
            # print("image_file:{}\t".format(image_file), end=',', flush=True)
            image = image_utils.read_image_ch(image_path)
            image = image_utils.resize_image(image, size=(None, 640))
            image, face_info = self.search_face_task(image, vis=vis)
            if len(face_info['label']) > 0:
                if face_info['label'][0] == "unknown":
                    return 1, 0
                else:
                    return int(face_info['label'][0]), face_info['score'][0]
            else:
                return 1, 0
        except Exception:
            traceback.print_exc()
            logger.error(image_path, flush=True)
            return 1, 0

    def search_face_image(self, raw_image, vis=False):
        try:
            # image = image_utils.read_image_ch(image_path)
            if raw_image is None:
                return 1, 0
            if raw_image.size == 0:
                logger.error(f"{self.pname} raw_image size == 0")
                return 1, 0
            image = image_utils.resize_image(raw_image, size=(None, 640))
            image, face_info = self.search_face_task(image, vis=vis)
            if len(face_info['label']) > 0:
                if face_info['label'][0] == "unknown":
                    return 1, 0
                else:
                    return int(face_info['label'][0]), face_info['score'][0]
            else:
                return 1, 0
        except Exception:
            traceback.print_exc()
            return 1, 0

    # 源码
    def crop_faces_alignment(self, rgb, boxes, landm):
        """
        裁剪人脸，并进行矫正
        @param rgb:
        @param boxes:
        @param landm:
        @return:
        """
        faces = self.faceDet.crop_faces_alignment(rgb, boxes, landm, alignment=True)
        return faces

    def extract_feature(self, faces: List[np.ndarray]):
        """
        提取人脸特征(不会进行人脸检测和矫正)
        @param faces: 人脸图像列表(BGR格式)
        @return: feature 返回人脸特
        """
        rgb = [cv2.cvtColor(r, cv2.COLOR_BGR2RGB) for r in faces]  # 转换为RGB格式
        feature = self.faceFea.get_faces_embedding(rgb)
        if not isinstance(feature, np.ndarray): feature = feature.numpy()
        return feature

    def detect_extract_feature(self, bgr: np.ndarray, max_face=-1, vis=False):
        """
        进行人脸检测和矫正，同时提取人脸特征
        @param bgr: 输入BGR格式的图像
        @param max_face: 最大人脸个数，默认为-1，表示全部人脸
        @param vis: 是否显示人脸检测框和人脸关键点
        @return: face_info={"boxes": 人脸检测框, "landm": 人脸关键点, "feature": 人脸特征, "face": 人脸图像}
        """
        boxes, score, landm = self.detector(bgr, max_face=max_face, vis=False)
        feature, faces = np.array([]), []
        if len(boxes) > 0 and len(landm) > 0:
            faces = self.crop_faces_alignment(bgr, boxes, landm)
            feature = self.extract_feature(faces)
        face_info = {"boxes": boxes, "landm": landm, "feature": feature, "face": faces}
        if vis: self.draw_result("Detector", image=bgr, face_info=face_info, vis=vis)
        return face_info

    def detector(self, bgr, max_face=-1, vis=False):
        """
        进行人脸检测和关键点检测
        @param bgr: 输入BGR格式的图像
        @param max_face: 图像中最大人脸个数，-1表示全部人脸
        @param vis: 是否可视化检测效果
        @return:
        """
        boxes, score, landm = self.faceDet.detect_face_landmarks(bgr, vis=vis)
        if len(boxes) > 0 and max_face > 0:
            max_face = min(len(boxes), max_face)
            boxes = boxes[:max_face]
            score = score[:max_face]
            landm = landm[:max_face]
        return boxes, score, landm

    def detect_search(self, bgr, max_face=-1, vis=False):
        """
        进行人脸检测,1:N人脸搜索
        @param bgr:
        @param max_face:
        @param vis:
        @return:
        """
        # 人脸检测并提取人脸特征
        face_info: Dict = self.detect_extract_feature(bgr, max_face=max_face, vis=False)
        # 匹配人脸
        label, score = self.search_face(face_fea=face_info["feature"], rec_thresh=self.config.insight_rec_thresh)
        face_info.update({"label": label, "score": score})
        if vis: self.draw_result("Recognizer", image=bgr, face_info=face_info, vis=vis)
        return face_info

    def create_database(self, portrait, vis=False):
        """
        生成人脸数据库(Face Database)
        @param portrait:人脸数据库目录，要求如下：
                          (1) 图片按照[ID-XXXX.jpg]命名,如:张三-image.jpg，作为人脸识别的底图
                          (2) 人脸肖像照片要求五官清晰且正脸的照片，不能出现多个人脸的情况
        @param vis: 是否可视化人脸检测效果
        @return:
        """
        logger.info(f"{self.pname} >>>>>>>>>>>>>开始注册人脸<<<<<<<<<<<<<<")
        image_list = file_utils.get_images_list(portrait)
        for image_file in image_list:
            name = os.path.basename(image_file)
            label = name.split("-")[0]
            if len(name) == len(label):
                logger.info("{} file={},\t图片名称不合法，请将图片按照[ID-XXXX.jpg]命名,如:张三-image.jpg"
                            .format(self.pname, image_file))
                continue
            bgr = image_utils.read_image_ch(image_file)
            logger.info("{} file={},\t".format(self.pname, image_file), end="")
            self.add_face(face_id=label, bgr=bgr, vis=vis)
        # 更新人脸数据库并保存
        self.faceReg.update()
        self.faceReg.save()
        logger.info(f"{self.pname} >>>>>>>>>>>>>完成注册人脸<<<<<<<<<<<<<<")
        return

    def draw_result(self, title, image, face_info, thickness=2, fontScale=1.0, color=(0, 255, 0), delay=0, vis=False):
        """
        @param title:
        @param image:
        @param face_info:
        @param thickness:
        @param fontScale:
        @param delay:
        @param vis:
        @return:
        """
        boxes = face_info["boxes"]
        if "landm" in face_info:
            radius = int(2 * thickness)
            image = image_utils.draw_landmark(image, face_info["landm"], color=color, radius=radius, vis_id=False)
        if "score" in face_info:
            score = face_info["score"].reshape(-1).tolist()
            label = face_info["label"] if "label" in face_info else [""] * len(boxes)
            text = ["{},{:3.3f}".format(l, s) for l, s in zip(label, score)]
        else:
            text = [""] * len(boxes)
        image = image_utils.draw_image_bboxes_text(image, boxes, text, thickness=thickness, fontScale=fontScale,
                                                   color=color, drawType="chinese")
        if vis and "label" in face_info: logger.info("{} pred label:{}".format(self.pname, text))
        if vis: image_utils.cv_show_image(title, image, delay=delay)
        return image

    def add_face(self, face_id: str, bgr: np.ndarray, vis=False):
        """
        :param face_id:  人脸ID(如姓名、学号，身份证等标识)
        @param bgr: 原始图像
        @param vis: 是否可视化人脸检测效果
        @return:
        """
        info = self.detect_extract_feature(bgr, max_face=1, vis=vis)
        if len(info["boxes"]) == 0:
            logger.info("{} Warning:{}".format(self.pname, "no face"))
            return False
        self.faceReg.add_face(face_id, face_fea=info["feature"])
        logger.info("{} 注册人脸:ID={}\t".format(self.pname, face_id))
        return True

    def del_face(self, face_id: str):
        """
        注销(删除)人脸
        :param face_id:  人脸ID(如姓名、学号，身份证等标识)
        :return:
        """
        return self.faceReg.del_face(face_id)

    def search_face(self, face_fea, rec_thresh):
        """
        1:N人脸搜索，比较人脸特征相似程度,如果相似程度小于rec_thresh的ID,则返回unknown
        :param face_fea: 人脸特征,shape=(nums_face, embedding_size)
        :param rec_thresh: 人脸识别阈值
        :return:返回预测pred_id的ID和距离分数pred_scores
        """
        label, score = self.faceReg.search_face(face_fea=face_fea, score_thresh=rec_thresh)
        return label, score

    def compare_face(self, image1, image2):
        """
        1:1人脸比对，compare pair image
        :param  image1: BGR image
        :param  image2: BGR image
        :return: similarity
        """
        # 进行人脸检测和矫正，同时提取人脸特征
        face_info1 = self.detect_extract_feature(image1, max_face=1, vis=False)
        face_info2 = self.detect_extract_feature(image2, max_face=1, vis=False)
        # 比较人脸特征的相似性(similarity)
        score = 0
        if len(face_info1['face']) > 0 and len(face_info2['face']) > 0:
            score = self.compare_feature(face_fea1=face_info1["feature"], face_fea2=face_info2["feature"])
        return face_info1, face_info2, score

    def compare_feature(self, face_fea1, face_fea2):
        """
        1:1人脸比对，compare two faces vector
        :param  face_fea1: face feature vector
        :param  face_fea2: face feature vector
        :return: similarity
        """
        score = self.faceReg.compare_feature(face_fea1, face_fea2)
        return score

    def detect_image_dir(self, image_dir, out_dir=None, vis=False):
        """
        @param image_dir:
        @param out_dir:
        @param vis:
        @return:
        """
        image_list = file_utils.get_files_list(image_dir, postfix=["*.jpg"])
        for image_file in image_list:
            try:
                logger.info("{} image_file:{}\t".format(self.pname, image_file), end=',', flush=True)
                image = image_utils.read_image_ch(image_file)
                face_info = self.detect_search(image, max_face=-1, vis=vis)
            except:
                traceback.print_exc()
                logger.info(f"{self.pname} {image_file}")
        return None
