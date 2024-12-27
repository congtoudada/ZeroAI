import os.path as osp

from .bases import BaseImageDataset


class Cong2025(BaseImageDataset):
    """
    Cong2025
    """
    dataset_dir = 'Cong2025'
    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(Cong2025, self).__init__()
        self.pid_begin = pid_begin
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.test_dir = osp.join(self.dataset_dir, 'test')  # 既是测试集，也是gallery
        self.query_dir = osp.join(self.dataset_dir, 'test')
        self.list_train_path = osp.join(self.dataset_dir, 'list_train.txt')
        self.list_val_path = osp.join(self.dataset_dir, 'list_val.txt')
        self.list_gallery_path = osp.join(self.dataset_dir, 'list_gallery.txt')
        self.list_query_path = osp.join(self.dataset_dir, 'list_query.txt')

        self._check_before_run()
        train = self._process_dir(self.train_dir, self.list_train_path)
        val = self._process_dir(self.train_dir, self.list_val_path)
        train += val
        gallery = self._process_dir(self.test_dir, self.list_gallery_path)
        query = self._process_dir(self.query_dir, self.list_query_path)

        self.train = train
        self.query = query
        self.gallery = gallery

        # self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = 1041, 32621, 15, 1  # 使用MSMT17的训练权重
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

        if verbose:
            print("=> Cong2025 loaded")
            print("Dataset statistics:")
            print("  ----------------------------------------")
            print("  subset   | # ids | # images | # cameras")
            print("  ----------------------------------------")
            print(
                "  train    | {:5d} | {:8d} | {:9d}".format(self.num_train_pids, self.num_train_imgs,
                                                            self.num_train_cams))
            print(
                "  query    | {:5d} | {:8d} | {:9d}".format(self.num_query_pids, self.num_query_imgs,
                                                            self.num_query_cams))
            print("  gallery  | {:5d} | {:8d} | {:9d}".format(self.num_gallery_pids, self.num_gallery_imgs,
                                                              self.num_gallery_cams))
            print("  ----------------------------------------")

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _process_dir(self, dir_path, list_path):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        dataset = []
        pid_container = set()
        cam_container = set()
        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid)  # no need to relabel
            camid = int(img_path.split('_')[2])
            img_path = osp.join(dir_path, img_path)
            dataset.append((img_path, self.pid_begin+pid, camid-1, 0))
            pid_container.add(pid)
            cam_container.add(camid)
        print(cam_container, 'cam_container')
        # check if pid starts from 0 and increments with 1
        for idx, pid in enumerate(pid_container):
            assert idx == pid, "See code comment for explanation"
        return dataset

