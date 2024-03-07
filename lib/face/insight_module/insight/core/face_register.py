# -*-coding: utf-8 -*-
"""
    @Author :
    @E-mail :
    @Date   : 2018-04-30 14:23:46
"""
import os
import traceback
import numpy as np
from typing import Dict
from pybaseutils import json_utils, file_utils
from insight.core.face_matcher import EmbeddingMatching
from loguru import logger


class FaceRegister(object):
    def __init__(self, data_file=""):
        """
        @param data_file: 人脸数据库的文件，默认为空
        """
        self.pname = f"[ {os.getpid()}:face_register ]"
        self.data_file = data_file
        self.face_database = {}
        if os.path.exists(self.data_file):
            self.face_database = self.load(file=self.data_file)
        self.matcher = EmbeddingMatching(None, None)
        self.update()

    def update(self):
        """更新人脸数据库"""
        face_id, face_fea = self.get_database_list()
        if len(face_fea) > 0: face_fea = np.array(face_fea, dtype=np.float32).reshape(len(face_id), -1)
        self.matcher = EmbeddingMatching(face_fea, face_id)

    def add_face(self, face_id: str, face_fea: np.ndarray, update=False):
        """
        注册(增加)人脸
        :param face_id:  人脸ID(如姓名、学号，身份证等标识)
        :param face_fea: 人脸特征
        :param update: 是否更新人脸数据库
        :return:
        """
        if isinstance(face_fea, np.ndarray): face_fea = face_fea.tolist()
        self.face_database[face_id] = face_fea
        if update: self.update()

    def del_face(self, face_id: str, update=False):
        """
        注销(删除)人脸
        :param face_id: 人脸ID(如姓名、学号，身份证等标识)
        :param update: 是否更新人脸数据库
        :return:
        """
        face_fea = None
        try:
            face_fea = self.face_database.pop(face_id)
            if update: self.update()
        except Exception as e:
            traceback.logger.info_exc()
            logger.info(f"{self.pname} face_id不存在")
        return face_fea

    def get_face(self, face_id: str):
        """
        获得人脸ID的人脸特征
        :param face_id: 人脸ID(如姓名、学号，身份证等标识)
        :return:
        """
        face_fea = None
        try:
            face_fea = self.face_database[face_id]
        except Exception as e:
            traceback.logger.info_exc()
            logger.info(f"{self.pname} face_id不存在")
        return face_fea

    def get_database(self) -> Dict:
        """获得人脸数据库"""
        return self.face_database

    def get_database_list(self):
        """以列表形式获得人脸数据库"""
        face_id = list(self.face_database.keys())
        face_fea = list(self.face_database.values())
        return face_id, face_fea

    def save(self, file=None):
        """保存人脸数据库"""
        if not file: file = self.data_file
        file_utils.create_file_path(file)
        json_utils.write_json_path(file, self.face_database)
        logger.info("{} save database:{}".format(self.pname, self.data_file))

    def load(self, file=None):
        """加载人脸数据库"""
        if not file: file = self.data_file
        facebank = json_utils.read_json_data(file,)
        logger.info("{} load database:{}".format(self.pname, self.data_file))
        return facebank

    def search_face(self, face_fea, score_thresh, use_fast=False):
        """
        1:N人脸搜索，比较人脸特征相似程度,如果相似程度小于score_thresh的ID,则返回unknown
        :param face_fea: 人脸特征,shape=(nums_face, embedding_size)
        :param score_thresh: 人脸识别阈值
        :param use_fast:
        :return:返回预测pred_id的ID和距离分数pred_scores
        """
        if face_fea is None or len(face_fea) == 0: return np.array([]), np.array([])
        if use_fast:
            pred_name, pred_scores = self.matcher.fast_embedding_matching(face_fea, score_thresh)
        else:
            # pred_name, pred_scores = self.matcher.frame_embedding_matching(face_embedding, dist_threshold)
            pred_name, pred_scores = self.matcher.embedding_matching(face_fea, score_thresh)
        return pred_name, pred_scores

    def compare_feature(self, face_fea1, face_fea2):
        """
        1:1人脸比对，compare two faces vector
        :param  face_fea1: face feature vector
        :param  face_fea2: face feature vector
        :return: similarity
        """
        # 计算两个特征的相似性
        score = self.matcher.compare_embedding_scores(face_fea1, face_fea2)
        return score


if __name__ == "__main__":
    data_name = "data.json"
    fd = FaceRegister(data_name)
    face_fea = np.zeros(shape=(1, 512), dtype=np.float32)
    fd.add_face("face_id0", face_fea + 0 / 10000)
    fd.add_face("face_id1", face_fea + 1 / 10000)
    fd.add_face("face_id2", face_fea + 2 / 10000)
    fd.add_face("face_id3", face_fea + 3 / 10000)
    facebank = fd.get_database()
    face_id, face_fea = fd.get_database_list()
    face_fea = np.array(face_fea)
    fd.save()
    facebank = fd.load()

