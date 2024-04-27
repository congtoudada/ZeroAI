import sys
import time
import os.path as osp
import argparse
import torch
import torch.nn as nn
#
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

def find_indices_with_target_pid(filenames, target_pid):
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

def build_datamanager(cfg):
    if cfg.data.type == 'image':
        return ImageDataManager(**imagedata_kwargs(cfg)) #torchreid.data.
    else:
        return torchreid.data.VideoDataManager(**videodata_kwargs(cfg))


def build_engine(cfg, datamanager, model, optimizer, scheduler):
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


def reset_config(cfg, args):
    if args.root:
        cfg.data.root = args.root
    if args.sources:
        cfg.data.sources = args.sources
    if args.targets:
        cfg.data.targets = args.targets
    if args.transforms:
        cfg.data.transforms = args.transforms


def check_cfg(cfg):
    if cfg.loss.name == 'triplet' and cfg.loss.triplet.weight_x == 0:
        assert cfg.train.fixbase_epoch == 0, \
            'The output of classifier is not included in the computational graph'


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config-file', type=str, default='', help='path to config file'
    )
    parser.add_argument(
        '-s',
        '--sources',
        type=str,
        nargs='+',
        help='source datasets (delimited by space)'
    )
    parser.add_argument(
        '-t',
        '--targets',
        type=str,
        nargs='+',
        help='target datasets (delimited by space)'
    )
    parser.add_argument(
        '--transforms', type=str, nargs='+', help='data augmentation'
    )
    parser.add_argument(
        '--root', type=str, default='', help='path to data root'
    )
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='Modify config options using the command-line'
    )
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)
    set_random_seed(cfg.train.seed)
    check_cfg(cfg)

    log_name = 'test.log' if cfg.test.evaluate else 'train.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(cfg.data.save_dir, log_name))

    print('Show configuration\n{}\n'.format(cfg))
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    datamanager = build_datamanager(cfg)

    print('Building model: {}'.format(cfg.model.name))
    model = build_model( #torchreid.models.
        name=cfg.model.name,
        num_classes=datamanager.num_train_pids,
        loss=cfg.loss.name,
        pretrained=False,   #pzy 原：True  
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

    optimizer = build_optimizer(model, **optimizer_kwargs(cfg)) #pzy torchreid.optim.
    scheduler = build_lr_scheduler(  #pzy torchreid.optim.
        optimizer, **lr_scheduler_kwargs(cfg)
    )

    if cfg.model.resume and check_isfile(cfg.model.resume):
        cfg.train.start_epoch = resume_from_checkpoint(
            cfg.model.resume, model, optimizer=optimizer, scheduler=scheduler
        )

    print(
        'Building {}-engine for {}-reid'.format(cfg.loss.name, cfg.data.type)
    )
    engine = build_engine(cfg, datamanager, model, optimizer, scheduler)
    #engine.run(**engine_run_kwargs(cfg))
    n_topn_gallery_image_names, rank1, mAP = engine.run(**engine_run_kwargs(cfg))
   
    #需求2
    print(n_topn_gallery_image_names) 
    #需求1
    print(find_indices_with_target_pid(n_topn_gallery_image_names, '02'))
    
if __name__ == '__main__':
    main()
