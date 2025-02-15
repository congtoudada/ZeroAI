import numpy as np

from jigsaw.tool.evaluate import GroundTruthLoader

if __name__ == "__main__":
    # load ground truth
    gt_loader = GroundTruthLoader()
    gt = gt_loader(dataset="ped2")
    for idx, item in enumerate(gt):
        # 将 ndarray 写入到 .txt 文件，每行一个整数
        np.savetxt(rf'H:\AI\dataset\VAD\Featurize_png\ped2\ped2_gt\Test{idx+1:03d}.txt', item, fmt='%d')