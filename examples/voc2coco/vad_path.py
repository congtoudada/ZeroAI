# 文件: mypath
# 作者: 聪头
# 时间: 2023/3/29 17:02
# 描述:
import platform
import os

# 导入枚举类
from enum import Enum

# 继承枚举类
class DS(Enum): # Dataset
    ped2 = 1
    avenue = 2
    shanghaitech = 3

class Ph(Enum): # Phase
    testing = 1
    training = 2

def getVADPath(dataset=None, phase=None, withLastSep=True):
    plat = platform.system().lower()
    datasetPath = ""

    # 数据集公共目录
    if plat == 'windows':
        datasetPath = ".\\dataset" # 默认带有 '\\'
    elif plat == 'linux':
        datasetPath = "/root/data1/lzc/dataset/VAD" # 默认带有 '/'

    # 特定数据集目录: ped2,avenue,shanghaitech
    if dataset:
        datasetPath = os.path.join(datasetPath, dataset)

    # 特定阶段: testing, training
    if phase:
        datasetPath = os.path.join(datasetPath, phase)

    if withLastSep:
        datasetPath += os.path.sep

    return datasetPath

