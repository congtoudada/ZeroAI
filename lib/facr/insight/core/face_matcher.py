# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author :
# @E-mail :
# @Date   : 2018-04-07 17:56:20
# --------------------------------------------------------
"""
import os
import sys
import time
import math
import numpy as np
from sklearn.cluster import KMeans
import threading


def singleton(cls):
    _instance_lock = threading.Lock()
    instances = {}

    def _singleton(*args, **kwargs):
        with _instance_lock:
            if cls not in instances:
                instances[cls] = cls(*args, **kwargs)
            return instances[cls]

    return _singleton


@singleton
class FaceFeatureKmeans():
    """FaceFeature K-means(设置为单实例类)"""

    def __init__(self, embeddings, emb_id_list):
        self.embeddings, self.emb_id_list = embeddings, emb_id_list
        self.K = max(int(math.pow(len(self.emb_id_list), 1.0 / 2)), 1)
        print('please wait, do FaceFeatureKmeans')
        beg_time = time.time()
        self.centers, self.features_list, self.features_id_list = self._kmeans()
        end_time = time.time()
        print('kmeans init time: {:.5f}'.format(end_time - beg_time))

    def _kmeans(self):
        estimator = KMeans(
            n_clusters=self.K,
            max_iter=300,
            n_init=10,
            n_jobs=-1
        )
        estimator.fit(self.embeddings)

        centers = estimator.cluster_centers_
        labels = estimator.labels_
        features_list = [[] for i in range(self.K)]
        features_id_list = [[] for i in range(self.K)]
        for i in range(len(self.embeddings)):
            features_list[labels[i]].append(self.embeddings[i])
            features_id_list[labels[i]].append(self.emb_id_list[i])

        for i in range(self.K):
            features_list[i] = np.array(features_list[i])
            features_id_list[i] = np.array(features_id_list[i])
        features_list = np.array(features_list)
        features_id_list = np.array(features_id_list)

        return centers, features_list, features_id_list

    def fast_embedding_matching(self, face_emb, score_threshold):
        """
        通过欧式距离,比较embeddings特征相似程度,如果未找到小于dist_threshold的ID,则返回pred_idx为-1
        :param face_embedding: 人脸特征
        :param score_threshold: 人脸识别阈值
        :param numpyData:
        :return:返回预测pred_id的ID和距离分数pred_scores
        :return:
        """
        face_emb = np.expand_dims(face_emb, axis=-1)  # (512,)->(512,1)

        centers_embeddings = self.centers.transpose(1, 0)  # (20, 512)->(512, 20)
        centers_embeddings = np.expand_dims(centers_embeddings, axis=0)  # (512, 20)->(1, 512, 20)
        centers_diff = face_emb - centers_embeddings  # (1, 512, 20)
        centers_dist = np.sum(np.square(centers_diff), axis=1)  # (1, 20)

        center_argsort = np.argsort(centers_dist[0])
        search_iter = max(int(self.K / 3), 1)
        for idx in range(search_iter):
            center_idx = center_argsort[idx]
            feature_embeddings = self.features_list[center_idx].transpose(1, 0)  # (20, 512)->(512, 20)
            feature_embeddings = np.expand_dims(feature_embeddings, axis=0)  # (512, 20)->(1, 512, 20)
            feature_diff = face_emb - feature_embeddings  # (1, 512, 20)
            feature_dist = np.sum(np.square(feature_diff), axis=1)  # (1, 20)

            feature_pred_idx = np.argmin(feature_dist, axis=1)
            feature_pred_name = self.features_id_list[center_idx][feature_pred_idx]
            pred_scores = np.min(feature_dist, axis=1)
            pred_scores = self.get_scores(pred_scores)

            if pred_scores > score_threshold:
                break

        feature_pred_name[pred_scores < score_threshold] = -1  # if no match, set idx to -1
        return feature_pred_name, pred_scores

    @staticmethod
    def get_scores(x, meam=1.40, std=0.2):
        x = -(x - meam) / std
        # sigmoid
        scores = 1.0 / (1.0 + np.exp(-x))
        return scores


class EmbeddingMatching(object):
    """人脸特征匹配算法"""

    def __init__(self, dataset_embeddings, dataset_id_list):
        """
        :param dataset_embeddings:人脸底库数据embeddings特征
        :param dataset_id_list:人脸底库数据的ID
        """
        self.dataset_embeddings = dataset_embeddings
        self.dataset_names_list = dataset_id_list

    def embedding_matching(self, face_emb, score_threshold):
        """
        通过欧式距离,比较embeddings特征相似程度,如果未找到小于dist_threshold的ID,则返回pred_idx为-1
        :param face_embedding: 人脸特征
        :param score_threshold: 人脸识别阈值
        :param numpyData:
        :return:返回预测pred_id的ID和距离分数pred_scores
        :return:
        """
        assert len(self.dataset_embeddings) > 0, Exception("请先注册人脸")
        face_emb = np.expand_dims(face_emb, axis=-1)  # (512,)->(512,1)
        dataset_embeddings = self.dataset_embeddings.transpose(1, 0)  # (20, 512)->(512, 20)
        dataset_embeddings = np.expand_dims(dataset_embeddings, axis=0)  # (512, 20)->(1, 512, 20)
        diff = face_emb - dataset_embeddings  # (1, 512, 20)
        dist = np.sum(np.square(diff), axis=1)  # (1, 20)
        pred_idx = np.argmin(dist, axis=1)
        pred_scores = np.min(dist, axis=1)
        pred_scores = EmbeddingMatching.get_scores(pred_scores)
        pred_idx[pred_scores < score_threshold] = -1  # if no match, set idx to -1
        pred_name, pred_scores = self.decode_label(pred_idx, pred_scores, self.dataset_names_list)
        return pred_name, pred_scores

    def fast_embedding_matching(self, face_emb, score_threshold):
        """
        通过欧式距离,比较embeddings特征相似程度,如果未找到小于dist_threshold的ID,则返回pred_idx为-1
        :param face_embedding: 人脸特征
        :param score_threshold: 人脸识别阈值
        :param numpyData:
        :return:返回预测pred_id的ID和距离分数pred_scores
        :return:
        """
        # 单实例
        face_feature_kmeans = FaceFeatureKmeans(self.dataset_embeddings, self.dataset_names_list)
        return face_feature_kmeans.fast_embedding_matching(face_emb, score_threshold)

    def frame_embedding_matching(self, face_emb, score_threshold):
        """
        比较一帧数据中,所有人脸的embeddings特征相似程度,如果相似程度小于dist_threshold的ID,则返回pred_idx为-1
        PS:由于是同一帧的人脸,因此需要考虑人脸ID的互斥性
        :param face_emb: 输入的人脸特征必须是同一帧的人脸特征数据,shape=(nums_face, embedding_size)
        :param score_threshold: 人脸识别阈值
        :return:返回预测pred_id的ID和距离分数pred_scores
        """
        nums = len(face_emb)
        face_emb = np.expand_dims(face_emb, axis=-1)
        dataset_embeddings = self.dataset_embeddings.transpose(1, 0)
        dataset_embeddings = np.expand_dims(dataset_embeddings, axis=0)
        diff = face_emb - dataset_embeddings
        dist = np.sum(np.square(diff), axis=1)
        scores_mat = self.get_scores(dist)
        pred_scores = np.max(scores_mat, axis=1)
        index = np.argsort(-pred_scores)  # 逆序
        # ===================================
        # pred_idx = np.argmax(scores_mat, axis=1)
        # pred_data = np.vstack((pred_idx, pred_scores)).T
        # pred_data1 = pred_data[index, :]  # 按最后一列进行排序
        dst_idx_score = np.zeros(shape=(nums, 2))
        record_label = []
        for i in index:
            score = scores_mat[i, :]
            id = np.argmax(score)
            label = self.dataset_names_list[id]
            # label = id
            s = score[id]
            while label in record_label:
                score[id] = 0  # warnning : 0 will assign to scores_mat
                id = np.argmax(score)
                label = self.dataset_names_list[id]
                # label = id
                s = score[id]
                if s < score_threshold or s == 0:
                    id = -1
                    break
            record_label.append(label)
            dst_idx_score[i, :] = (id, s)
        # dst_idx_score1 = dst_idx_score[index, :]
        dst_idx = dst_idx_score[:, 0]
        dst_score = dst_idx_score[:, 1]
        dst_idx = np.asarray(dst_idx, np.int32)
        dst_idx[dst_score < score_threshold] = -1  # if no match, set id to -1
        dst_name, dst_score = self.decode_label(dst_idx, dst_score, self.dataset_names_list)
        return dst_name, dst_score

    @staticmethod
    def get_scores(x, meam=1.40, std=0.2):
        x = -(x - meam) / std
        # sigmoid
        scores = 1.0 / (1.0 + np.exp(-x))
        return scores

    @staticmethod
    def compare_embedding_scores(vect1, vect2):
        """
        compare two faces
        Args:
            face1_vector: vector of face1
            face2_vector: vector of face2

        Returns: similarity

        """
        v1 = np.asarray(vect1)
        v2 = np.asarray(vect2)
        # dist = np.linalg.norm(v1 - v2)
        dist = np.sum(np.square(v1 - v2), axis=1)
        scores = EmbeddingMatching.get_scores(dist)
        return scores

    @staticmethod
    def decode_label(pred_id, pred_scores, dataset_id_list):
        """
        对ID进行解码,返回对应的lable名称,当pred_id为-1时,返回"-1"
        :param pred_id:
        :param pred_scores:
        :param dataset_id_list:
        :return:
        """
        pred_id_list = []
        for id in pred_id:
            id = int(id)
            name = 'unknown' if id == -1 else dataset_id_list[id]
            pred_id_list.append(name)
        return pred_id_list, pred_scores


if __name__ == "__main__":
    pass
