from __future__ import division, print_function, absolute_import
import re
import glob
import os.path as osp
import warnings

from ..dataset import ImageDataset


class Market1501(ImageDataset):
    """Market1501.

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_
    
    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    _junk_pids = [0, -1]
    dataset_dir = 'market1501'
    dataset_url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'Market-1501-v15.09.15')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn(
                'The current data structure is deprecated. Please '
                'put data folders such as "bounding_box_train" under '
                '"Market-1501-v15.09.15".'
            )

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')#market1501里面的三个文件夹
        self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        self.market1501_500k = market1501_500k

        required_files = [
            self.data_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        if self.market1501_500k:
            required_files.append(self.extra_gallery_dir)
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True) #为什么训练relabe是true  其他时候是false
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)
        if self.market1501_500k:
            gallery += self.process_dir(self.extra_gallery_dir, relabel=False)

        super(Market1501, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))#查找指定目录dir_path下所有以.jpg结尾的文件，并将文件路径存储在img_paths列表中
        pattern = re.compile(r'([-\d]+)_c(\d)') #使用正则表达式创建一个模式，用于匹配文件名中的PID和camid

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups()) #使用正则表达式模式匹配当前图像路径img_path中的PID，并将其转换为整数赋值给pid。下划线_用于忽略camid。
            if pid == -1:
                continue # junk images are just ignored
            pid_container.add(pid) #将非垃圾图像的PID添加到pid_container集合中。
        pid2label = {pid: label for label, pid in enumerate(pid_container)} #根据pid_container中的PID创建一个字典pid2label，将每个PID映射为一个唯一的标签。

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())#从当前图像路径中提取PID和camid，并将它们转换为整数赋值给pid和camid。
            if pid == -1:
                continue # junk images are just ignored
            assert 0 <= pid <= 1501 # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1 # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid)) #将包含图像路径、PID和camid的元组添加到data列表中。

        return data
