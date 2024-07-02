import sys
import time
import os.path as osp
import argparse
import torch
import torch.nn as nn
#sys.path.insert(0, "/root/user66/anaconda3/envs/zuyi")
sys.path.append("/user66/zuyi/OSNet/torchreid")
#print(sys.path)
import torchreid
from torchreid.models import build_model #pzy
from torchreid.optim import (build_optimizer,build_lr_scheduler) #pzy
from torchreid.data.datamanager import ImageDataManager #pzy
from torchreid.engine import ImageSoftmaxEngine #pzy  
from torchreid.utils import (
    Logger, check_isfile, set_random_seed, collect_env_info,
    resume_from_checkpoint, load_pretrained_weights, compute_model_complexity
)

from default_config import (
    imagedata_kwargs, optimizer_kwargs, videodata_kwargs, engine_run_kwargs,
    get_default_config, lr_scheduler_kwargs
)
from pprint import pprint
import os
import shutil

class Reid:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('opts', default=None, nargs=argparse.REMAINDER, help='Modify config options using the command-line')
        self.parser.add_argument('-s','--sources',type=str,nargs='+',help='source datasets (delimited by space)')
        self.parser.add_argument('-t','--targets',type=str,nargs='+',help='target datasets (delimited by space)')
        self.parser.add_argument('--config-file', type=str, default='conf/dev/algorithm/reid/dut_test_1c.yaml', help='path to config file')
        self.parser.add_argument('--transforms', type=str, nargs='+', default=['random_flip', 'random_erase'], help='data augmentation')
        self.parser.add_argument('--root', type=str, default='lib/reid/reid_module/root_datasets/dut_test', help='path to data root')
        self.parser.add_argument('--model_weights', type=str, default='pretrained/reid/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth', help='path to model weights')
        self.parser.add_argument('--test_evaluate', action='store_true', default=True, help='whether to perform evaluation')
        self.args = self.parser.parse_args()
        
    def run(self,demo_data1,demo_data2):
        cfg = get_default_config()
        cfg.use_gpu = torch.cuda.is_available()
        if self.args.config_file:
            cfg.merge_from_file(self.args.config_file)
        self.reset_config(cfg)
        cfg.merge_from_list(self.args.opts)
        set_random_seed(cfg.train.seed)
        self.check_cfg(cfg)
        
        print(self.args.config_file,self.args.root,self.args.test_evaluate,"调试Line47")
        print(cfg.test.evaluate, cfg.model.load_weights)
        pprint(cfg)
        print(dir(self),"line54调试待删除")
        
        # 如果 demo_data1 是目录，则调用 replace_query_contents_with_directory 方法。
            #自己定义一个self.query_dir
        self.query_dir = os.path.join(cfg.data.root, 'tmp_file', 'query')
        if os.path.isdir(demo_data1):
            print(demo_data1,"调试代码待删除")
            self.replace_query_contents_with_directory(demo_data1, self.query_dir)
        else:
            print(f"The demo_data1 {demo_data1} is not a directory.")
        # 如果 demo_data2 是目录，则调用 replace_query_contents_with_directory 方法。
            #自己定义一个self.query_dir
        self.gallery_dir = os.path.join(cfg.data.root, 'tmp_file', 'bounding_box_test')
        if os.path.isdir(demo_data2):
            print(demo_data2,"调试代码待删除")
            self.replace_query_contents_with_directory(demo_data2, self.gallery_dir)
        else:
            print(f"The demo_data2 {demo_data2} is not a directory.")

        print(dir(self),"line73调试待删除")
        
        log_name = 'test.log' if cfg.test.evaluate else 'train.log'
        log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
        sys.stdout = Logger(osp.join(cfg.data.save_dir, log_name))

        print('Show configuration\n{}\n'.format(cfg))
        print('Collecting env info ...')
        print('** System info **\n{}\n'.format(collect_env_info()))

        if cfg.use_gpu:
            torch.backends.cudnn.benchmark = True

        datamanager = self.build_datamanager(cfg)

        print('Building model: {}'.format(cfg.model.name))
        model = build_model(
            name=cfg.model.name,
            num_classes=datamanager.num_train_pids,
            loss=cfg.loss.name,
            pretrained=False,  # pzy 原：True
            use_gpu=cfg.use_gpu,
        )

        num_params, flops = compute_model_complexity(
            model, (1, 3, cfg.data.height, cfg.data.width)
        )
        print('Model complexity: params={:,} flops={:,}'.format(num_params, flops))

        if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
            load_pretrained_weights(model, cfg.model.load_weights)

        if cfg.use_gpu:
            model = nn.DataParallel(model).cuda()

        optimizer = build_optimizer(model, **optimizer_kwargs(cfg))  # pzy torchreid.optim.
        scheduler = build_lr_scheduler(  # pzy torchreid.optim.
            optimizer, **lr_scheduler_kwargs(cfg)
        )

        if cfg.model.resume and check_isfile(cfg.model.resume):
            cfg.train.start_epoch = resume_from_checkpoint(
                cfg.model.resume, model, optimizer=optimizer, scheduler=scheduler
            )
        print(
            'Building {}-engine for {}-reid'.format(cfg.loss.name, cfg.data.type)
        )
        engine = self.build_engine(cfg, datamanager, model, optimizer, scheduler)
        # engine.run(**engine_run_kwargs(cfg))
        n_topn_gallery_image_names, rank1, mAP = engine.run(**engine_run_kwargs(cfg))


        # 需求:
        # for row in n_topn_gallery_image_names:
        #     # 使用生成器表达式将元组中的每个元素转换为字符串
        #     print(' '.join(str(item) for item in row))
        
        # 暂时不用，可能有问题 
        #print(find_indices_with_target_pid(n_topn_gallery_image_names, '03'))
        formatted_rows = self.get_formatted_rows(n_topn_gallery_image_names)
        return formatted_rows


    def build_datamanager(self,cfg):
        if cfg.data.type == 'image':
            return ImageDataManager(**imagedata_kwargs(cfg)) #torchreid.data.
        else:
            return torchreid.data.VideoDataManager(**videodata_kwargs(cfg))


    def build_engine(self,cfg, datamanager, model, optimizer, scheduler):
        if cfg.data.type == 'image':
            if cfg.loss.name == 'softmax':
                engine = ImageSoftmaxEngine(   
                    datamanager,
                    model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    use_gpu=cfg.use_gpu,
                    label_smooth=cfg.loss.softmax.label_smooth
                )

            else:
                engine = torchreid.engine.ImageTripletEngine(
                    datamanager,
                    model,
                    optimizer=optimizer,
                    margin=cfg.loss.triplet.margin,
                    weight_t=cfg.loss.triplet.weight_t,
                    weight_x=cfg.loss.triplet.weight_x,
                    scheduler=scheduler,
                    use_gpu=cfg.use_gpu,
                    label_smooth=cfg.loss.softmax.label_smooth
                )

        else:
            if cfg.loss.name == 'softmax':
                engine = torchreid.engine.VideoSoftmaxEngine(
                    datamanager,
                    model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    use_gpu=cfg.use_gpu,
                    label_smooth=cfg.loss.softmax.label_smooth,
                    pooling_method=cfg.video.pooling_method
                )

            else:
                engine = torchreid.engine.VideoTripletEngine(
                    datamanager,
                    model,
                    optimizer=optimizer,
                    margin=cfg.loss.triplet.margin,
                    weight_t=cfg.loss.triplet.weight_t,
                    weight_x=cfg.loss.triplet.weight_x,
                    scheduler=scheduler,
                    use_gpu=cfg.use_gpu,
                    label_smooth=cfg.loss.softmax.label_smooth
                )

        return engine


    def reset_config(self, cfg):
        if self.args.root:
            cfg.data.root = self.args.root
        if self.args.sources:
            cfg.data.sources = self.args.sources
        if self.args.targets:
            cfg.data.targets = self.args.targets
        if self.args.transforms:
            cfg.data.transforms = self.args.transforms
        #pzy
        if self.args.test_evaluate:
            cfg.test.evaluate = self.args.test_evaluate
        if self.args.model_weights: 
            #cfg.model.weights = self.args.model_weights
            cfg.model.load_weights = self.args.model_weights
        # if self.args.config-file: 
        #     cfg.config-file = self.args.config-file


    def check_cfg(self,cfg):
        if cfg.loss.name == 'triplet' and cfg.loss.triplet.weight_x == 0:
            assert cfg.train.fixbase_epoch == 0, \
                'The output of classifier is not included in the computational graph'
                
    # 请注意，这段代码假设 directory_path 已经被定义，并且 cfg.data.root 是一个有效的路径。
    # 此外，这段代码将复制目录中的所有文件，而不会检查文件类型。如果您需要复制特定类型的文件（例如，只复制 .jpg 图片），您需要在复制之前添加适当的文件类型检查。
    def replace_query_contents_with_directory(self, directory_path, target_directory):
        """
        清空 target_directory 目录并将指定目录中的所有文件复制到该目录。
        """
        if not os.path.exists(directory_path):
            print(f"Error: The directory path provided ({directory_path}) does not exist.")
            return

        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        # 清空 target_directory 目录
        for filename in os.listdir(target_directory):
            file_to_delete = os.path.join(target_directory, filename)
            try:
                if os.path.isfile(file_to_delete) or os.path.islink(file_to_delete):
                    os.unlink(file_to_delete)
                elif os.path.isdir(file_to_delete):
                    shutil.rmtree(file_to_delete)
            except Exception as e:
                print(f"Failed to delete {file_to_delete}. Reason: {e}")

        # 复制新的文件到 target_directory 目录
        for filename in os.listdir(directory_path):
            file_to_copy = os.path.join(directory_path, filename)
            if os.path.isfile(file_to_copy):
                try:
                    shutil.copy(file_to_copy, target_directory)
                    print(f"File {file_to_copy} has been copied to {target_directory}.")
                except Exception as e:
                    print(f"Failed to copy {file_to_copy} to {target_directory}. Reason: {e}")

    
    def find_indices_with_target_pid(self, filenames, target_pid):
        indices = []
        for index, row in enumerate(filenames):
            # 确保row是一个列表并且至少有一个元素
            if not isinstance(row, list) or not row:
                continue  # 如果不是列表或列表为空，则跳过此项
            
            # 获取每一行的第n+1个元素（因为每一行首个元素是query，第二个开始才是gallery），即第topn个检索项的地址，先设置topn=1，所以是[1+1-1]=[1]也就是第二列
            filename = row[1]
            # 确保filename是一个字符串
            if not isinstance(filename, str):
                continue  # 如果不是字符串，则跳过此项
            
            # 分割文件名以获取'/'后的最后一部分
            parts = filename.split('/')
            # 获取路径的最后一部分，其中包含'xx'部分
            last_part = parts[-1]
            # 分割最后一部分以'_'并获取'xx'部分
            xx_part = last_part.split('_')[0]
            # 检查'xx'部分是否与传入的target_pid相等
            if xx_part == target_pid:
                indices.append(index)
        return indices
    
    def get_formatted_rows(self, n_topn_gallery_image_names):
        # 使用列表推导式来构建每一行的字符串表示
        formatted_rows = [' '.join(str(item) for item in row) for row in n_topn_gallery_image_names]
        return formatted_rows


        
    
if __name__ == '__main__':
    reid_instance = Reid()
    #demo_data1 = r"C:\Users\zuyi\Downloads\10_c1s1_000290_00.jpg"
    #reid_instance.run(demo_data1)