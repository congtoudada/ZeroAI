# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author :
# @E-mail :
# @Date   : 2018-12-31 09:11:25
# --------------------------------------------------------
"""
import os
import torch
import torchvision.transforms as transforms
from insight.core.feature.demo import IR_18, IR_50
from insight.core.feature.demo import MobileNetV2
from insight.core.face_matcher import EmbeddingMatching
from pybaseutils import image_utils
from loguru import logger

root = os.path.dirname(__file__)

MODEL_FILE = {
    "resnet50": os.path.join(root, "../../../../../pretrained/face/insight/feature/weight/pth/resnet50.pth"),
    "resnet18": os.path.join(root, "../../../../../pretrained/face/insight/feature/weight/pth/resnet18.pth"),
    "mobilenet_v2": os.path.join(root, "../../../../../pretrained/face/insight/feature/weight/pth/mobilenet_v2.pth")
}


class FaceFeature(object):
    def __init__(self, model_file="", net_name="resnet18", input_size=(112, 112), embedding_size=512, device="cuda:0"):
        """
        :param model_file: model files
        :param net_name:
        :param input_size: [112,112] or [64,64]
        :param embedding_size:
        :param device: cuda id
        """
        self.pname = f"[ {os.getpid()}:face_feature ]"
        self.model_file = model_file if model_file else MODEL_FILE[net_name]
        # 初始化insightFace
        self.input_size = input_size
        self.device = device
        self.embedding_size = embedding_size
        self.model = self.build_net(self.model_file, net_name, input_size, self.embedding_size)
        self.model = self.model.to(device)
        self.model.eval()  # switch to evaluation mode
        self.transform = self.default_transform(input_size)
        logger.info("{} model_file:{}".format(self.pname, self.model_file))
        logger.info("{} network   :{}".format(self.pname, net_name))
        logger.info("{} use device:{}".format(self.pname, device))

    def build_net(self, model_file, net_name, input_size, embedding_size):
        """
        :param model_file:
        :param net_name:
        :param input_size:
        :param embedding_size:
        :return:
        """
        if net_name.lower() == "resnet18":
            mdoel = IR_18(input_size, embedding_size)
        elif net_name.lower() == "resnet50":
            mdoel = IR_50(input_size, embedding_size)
        elif net_name.lower() == "mobilenet_v2":
            mdoel = MobileNetV2(input_size, embedding_size)
        else:
            raise Exception("Error:{}".format(net_name))
        state_dict = torch.load(model_file, map_location="cpu")
        mdoel.load_state_dict(state_dict)
        return mdoel

    def forward(self, image_tensor):
        out_tensor = self.model(image_tensor.to(self.device))
        return out_tensor

    def get_faces_embedding(self, faces_list):
        """
        获取人脸的特征
        :param faces_list:人脸图像(RGB)列表,必须是已经裁剪的人脸,shape=[-1,112,112,3]
        :param landmarks : (可选)mtcnn人脸检测的landmarks,当存在时,则进行alignment,否则不进行alignment
                            landmarks=[[x0,y0],...,[x4,y4]]
        :return: 返回faces_list的512维人脸特征embeddings,shape=[-1,512]
        """
        with torch.no_grad():
            batch_faces = self.pre_process(faces_list, self.transform)
            embeddings = self.model(batch_faces.to(self.device))
            embeddings = self.post_process(embeddings).cpu()
        return embeddings

    def predict(self, faces_list, dist_threshold):
        """
        预测人脸ID
        :param faces_list:人脸图像(RGB)列表,必须是已经裁剪的人脸,shape=[-1,112,112,3]
        :param dist_threshold:人脸识别阈值
        :return:返回预测pred_id的ID和距离分数pred_scores
        """
        face_embeddings = self.get_faces_embedding(faces_list)
        pred_id, pred_scores = self.emb_matching.embedding_matching(face_embeddings, dist_threshold)
        return pred_id, pred_scores

    def set_database(self, dataset_embeddings, dataset_id_list):
        """
        设置人脸底库数据
        :param dataset_embeddings:人脸底库数据embeddings特征
        :param dataset_id_list:人脸底库数据的ID
        :param fast_match
        :return:
        """
        self.emb_matching = EmbeddingMatching(dataset_embeddings, dataset_id_list)

    def get_embedding_matching(self, face_embedding, dist_threshold, use_fast=True):
        """
        比较embeddings特征相似程度,如果相似程度小于dist_threshold的ID,则返回pred_idx为-1
        PS:不考虑人脸ID的互斥性
        :param face_emb: 人脸特征,shape=(nums_face, embedding_size)
        :param score_threshold: 人脸识别阈值
        :param use_fast:
        :return:返回预测pred_id的ID和距离分数pred_scores
        """
        if use_fast:
            pred_name, pred_scores = self.emb_matching.fast_embedding_matching(face_embedding, dist_threshold)
        else:
            # pred_name, pred_scores = self.emb_matching.frame_embedding_matching(face_embedding, dist_threshold)
            pred_name, pred_scores = self.emb_matching.embedding_matching(face_embedding, dist_threshold)
        return pred_name, pred_scores

    @staticmethod
    def get_label_name(pred_id, pred_scores, dataset_id_list):
        """
        对ID进行解码,返回对应的lable名称,当pred_id为-1时,返回unknow
        :param pred_id:
        :param pred_scores:
        :param dataset_id_list:
        :return:
        """
        pred_id_list, pred_scores = EmbeddingMatching.decode_label(pred_id, pred_scores, dataset_id_list)
        return pred_id_list, pred_scores

    @staticmethod
    def default_transform(input_size, RGB_MEAN=[0.5, 0.5, 0.5], RGB_STD=[0.5, 0.5, 0.5]):
        """
        人脸识别默认的预处理方法
        :param input_size:resize大小
        :param RGB_MEAN:均值
        :param RGB_STD: 方差
        :return:
        """
        transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize([input_size[0], input_size[1]]),
            transforms.Resize([int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)]),
            transforms.CenterCrop([input_size[0], input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
        ])
        return transform

    @staticmethod
    def pre_process(faces_list, transform):
        """
        @param faces_list:
        @param transform:
        @return:
        """
        outputs = []
        for face in faces_list:
            face = transform(face)
            outputs.append(face.unsqueeze(0))  # 增加一个维度
        outputs = torch.cat(outputs)
        return outputs

    @staticmethod
    def post_process(input, axis=1):
        """
        l2_norm
        :param input:
        :param axis:
        :return:
        """
        norm = torch.norm(input, 2, axis, True)
        output = torch.div(input, norm)
        return output


if __name__ == "__main__":
    from basetrainer.utils.converter import pytorch2onnx

    fr = FaceFeature(model_file="", net_name="mobilenet_v2", input_size=(112, 112), embedding_size=512)
    input_shape = (1, 3, 112, 112)
    pytorch2onnx.convert2onnx(fr.model, input_shape, input_names=['input'], output_names=['output'], )
