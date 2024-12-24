# -*-coding: utf-8 -*-
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import PIL.Image as Image
import torchvision
import cv2
import torchvision.transforms as transforms
from net.model_resnet import IR_18, IR_50
from net.mobilenet_v2 import MobileNetV2

# print(torch.__version__)


def post_process(embeddings, axis=1):
    '''
    特征后处理函数,l2_norm
    :param embeddings:
    :param axis:
    :return:
    '''
    norm = torch.norm(embeddings, 2, axis, True)
    output = torch.div(embeddings, norm)
    return output


def pre_process(input_size):
    '''
    输入图像预处理函数
    :param input_size:
    :return:
    '''
    data_transform = transforms.Compose([
        transforms.Resize([int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)]),
        transforms.CenterCrop([input_size[0], input_size[1]]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return data_transform


def compare_embedding(emb1, emb2):
    '''
    使用欧式距离比较两个人脸特征的差异
    :param emb1:
    :param emb2:
    :return:返回欧式距离(0,+∞),值越小越相似
    '''
    diff = emb1 - emb2
    dist = np.sum(np.power(diff, 2), axis=1)
    return dist


def get_scores(x, meam=1.40, std=0.2):
    '''
    人脸距离到人脸相似分数的映射
    :param x:欧式距离的值
    :param meam:均值,默认meam=1.40
    :param std: 方差,默认std=0.2
    :return: 返回人脸相似分数(0,1),值越大越相似
    '''
    x = -(x - meam) / std
    # sigmoid
    scores = 1.0 / (1.0 + np.exp(-x))
    return scores


def build_net(model_file, net_name, input_size, embedding_size):
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


MODEL_FILE = {
    "resnet50": "weight/pth/resnet50.pth",
    "resnet18": "weight/pth/resnet18.pth",
    "mobilenet_v2": "weight/pth/mobilenet_v2.pth"
}

if __name__ == "__main__":
    embedding_size = 512
    net_name = "resnet50"
    model_file = MODEL_FILE[net_name]
    input_size = [112, 112]  # 模型输入大小
    device = "cuda:0"
    score_thresh = 0.75  # 相似人脸分数人脸阈值
    test_image1 = "./face1.jpg"  # 测试图片1
    test_image2 = "./face2.jpg"  # 测试图片2
    # face1 = Image.open(test_image1)
    # face2 = Image.open(test_image2)
    face1 = cv2.imread(test_image1)
    face2 = cv2.imread(test_image2)
    face1 = cv2.cvtColor(face1, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
    face2 = cv2.cvtColor(face2, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
    data_transform = pre_process(input_size)
    # 加载模型
    model = build_net(model_file, net_name, input_size, embedding_size)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        # 输入图像预处理
        face_tensor1 = data_transform(Image.fromarray(face1))
        face_tensor2 = data_transform(Image.fromarray(face2))
        face_tensor1 = face_tensor1.unsqueeze(0)  # 增加一个维度
        face_tensor2 = face_tensor2.unsqueeze(0)  # 增加一个维度
        # forward
        embeddings1 = model(face_tensor1.to(device))
        embeddings2 = model(face_tensor2.to(device))
        # 特征后处理函数
        embeddings1 = post_process(embeddings1)
        embeddings2 = post_process(embeddings2)
        # 计算两个特征的欧式距离
        dist = compare_embedding(embeddings1.cpu().numpy(), embeddings2.cpu().numpy())
        # 将距离映射为人脸相似性分数
        score = get_scores(dist)
        # 判断是否是同一个人
        same_person = score > score_thresh
    print("embeddings1.shape:{}\nembeddings1:{}".format(embeddings1.shape, embeddings1[0, 0:20]))
    print("embeddings2.shape:{}\nembeddings2:{}".format(embeddings2.shape, embeddings2[0, 0:20]))
    print("distance:{},score:{},same person:{}".format(dist, score, same_person))
