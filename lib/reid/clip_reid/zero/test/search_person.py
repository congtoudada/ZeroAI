import os
from pathlib import Path
import faiss
import numpy as np
import torch
from matplotlib import pyplot as plt, image as mpimg
from torchvision import transforms
from clip_reid.config import cfg
import argparse
from PIL import Image

from clip_reid.datasets.make_dataloader_clipreid import make_dataloader
from clip_reid.model.make_model_clipreid import make_model
from clip_reid.utils.logger import setup_logger
from zero.helper.faiss_helper import FaissHelper


def get_feat(model, img_path):
    # 打开图片
    img = Image.open(img_path).convert('RGB')  # 确保图片是 RGB 格式
    # 显示图片
    img_ndarray = np.array(img)
    val_transforms = transforms.Compose([
        transforms.Resize(cfg.INPUT.SIZE_TEST),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    # 显示图片
    # plt.imshow(image1)  # 使用 matplotlib 显示图片
    # plt.axis('off')  # 不显示坐标轴
    # plt.show()
    # img_tensor = torch.from_numpy(img_ndarray).permute(2, 0, 1).unsqueeze(0).float().to(device)
    img_tensor = val_transforms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(img_tensor)
    return feat.to('cpu').detach().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="lib/reid/clip_reid/configs/person/vit_clipreid.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(cfg.TEST.WEIGHT)

    # 获取模型的设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 指定文件夹路径
    folder_path = Path("res/images/reid/gallery")

    # 遍历文件夹内的所有文件，拼接成相对路径
    img_database = [
        str(file)  # 转为字符串格式
        for file in folder_path.iterdir()
        if file.is_file()  # 只保留文件
    ]

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # 构建向量库
    d = 1280  # vit
    # d = 3072  # res50
    faiss_helper = FaissHelper(d)
    for img_path in img_database:
        feat = get_feat(model, img_path)
        faiss.normalize_L2(feat)
        faiss_helper.add(feat, {"img_path": img_path})

    # 准备查询向量
    query_path = "res/images/reid/query/0002_000_01_02.jpg"
    print("query: " + query_path)
    xq = get_feat(model, query_path)
    faiss.normalize_L2(xq)

    # 相似向量查询
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    k = 4  # topK的K值
    results = faiss_helper.search(xq, k)  # xq为待检索向量，返回的I为每个待检索query最相似TopK的索引list，D为其对应的距离

    print("------------ match ------------")
    for item in results:
        # 读取图片
        str = item['img_path']
        print(str)
        img = mpimg.imread(str)
        plt.imshow(img)
        plt.axis('off')  # 关闭坐标轴
        plt.show()



