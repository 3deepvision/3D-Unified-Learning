
import random
import time
import argparse
import calendar
import yaml
import os
import open3d as o3d
import sys
import logging
import wandb
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import distributed as dist, multiprocessing as mp
import torch.nn.functional as F
import warnings
import numpy as np
from collections import defaultdict, Counter
from datautils.forafterpointDataLoader import ShapeNetPartNormal
from models.pcnet_part_seg import get_model
import cv2


torch.backends.cudnn.benchmark = False
warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))

from openpoints.models import build_model_from_cfg
from openpoints.models.layers import torch_grouping_operation, knn_point
from openpoints.loss import build_criterion_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.dataset import build_dataloader_from_cfg, get_class_weights, get_features_by_keys
from openpoints.transforms import build_transforms_from_cfg
from openpoints.utils import AverageMeter, ConfusionMatrix
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb, generate_exp_directory, resume_exp_directory, EasyConfig, dist_utils, find_free_port
from openpoints.models.layers import furthest_point_sample



def batched_bincount(x, dim, max_value):
    target = torch.zeros(x.shape[0], max_value, dtype=x.dtype, device=x.device)
    values = torch.ones_like(x)
    target.scatter_add_(dim, x, values)
    return target


def part_seg_refinement(pred, pos, cls, cls2parts, n=10):
    pred_np = pred.cpu().data.numpy()
    for shape_idx in range(pred.size(0)):  # sample_idx
        parts = cls2parts[cls[shape_idx]]
        counter_part = Counter(pred_np[shape_idx])
        if len(counter_part) > 1:
            for i in counter_part:
                if counter_part[i] < n or i not in parts:
                    less_idx = np.where(pred_np[shape_idx] == i)[0]
                    less_pos = pos[shape_idx][less_idx]
                    knn_idx = knn_point(n + 1, torch.unsqueeze(less_pos, axis=0),
                                        torch.unsqueeze(pos[shape_idx], axis=0))[1]
                    neighbor = torch_grouping_operation(pred[shape_idx:shape_idx + 1].unsqueeze(1), knn_idx)[0][0]
                    counts = batched_bincount(neighbor, 1, cls2parts[-1][-1] + 1)
                    counts[:, i] = 0
                    pred[shape_idx][less_idx] = counts.max(dim=1)[1]
    return pred


def get_ins_mious(pred, target, cls, cls2parts,
                  multihead=False,
                  ):
    """Get the Shape IoU
    shape IoU: the mean part iou for each shape
    Args:
        pred (_type_): _description_
        target (_type_): _description_
        num_classes (_type_): _description_
    Returns:
        _type_: _description_
    """
    ins_mious = []
    for shape_idx in range(pred.shape[0]):  # sample_idx
        part_ious = []
        parts = cls2parts[cls[shape_idx]]
        if multihead:
            parts = np.arange(len(parts))

        for part in parts:
            pred_part = pred[shape_idx] == part
            target_part = target[shape_idx] == part
            I = torch.logical_and(pred_part, target_part).sum()
            U = torch.logical_or(pred_part, target_part).sum()
            if U == 0:
                iou = torch.tensor(100, device=pred.device, dtype=torch.float32)
            else:
                iou = I * 100 / float(U)
            part_ious.append(iou)
        ins_mious.append(torch.mean(torch.stack(part_ious)))
    return ins_mious


feature_map = []
def forward_hook(module, inp, outp):     # 定义hook
    feature_map.append(outp)    # 把输出装入字典feature_map

grad = []     # 建立列表容器，用于盛放特征图的梯度

def backward_hook(module, inp, outp):    # 定义hook
    grad.append(outp)    # 把输出装入列表grad

def main(gpu, cfg):
    if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        dist.init_process_group(backend=cfg.dist_backend,
                                init_method=cfg.dist_url,
                                world_size=cfg.world_size,
                                rank=cfg.rank)
        dist.barrier()
    # logger
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    if cfg.rank == 0:
        Wandb.launch(cfg, cfg.wandb.use_wandb)
        writer = None
    else:
        writer = None
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    logging.info(cfg)

    # build dataset



    root = 'D:/pythonproject/data/ShapeNetPart/'
    numpoints = 2048
    batch_size = 16
    TEST_DATASET = ShapeNetPartNormal(data_root=root, num_points=numpoints, split='test',transform=cfg.datatransforms)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=batch_size, shuffle=False, num_workers=4)
    cfg.cls2parts = testDataLoader.dataset.cls2parts



    validate_fn = eval(cfg.get('val_fn', 'validate'))

    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels

    model = get_model(50).cuda()
    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))

    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        model = nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        logging.info('Using Distributed Data parallel ...')

    ############## hook
    model.leader_pct.convs3.register_forward_hook(forward_hook)
    model.leader_pct.convs3.register_full_backward_hook(backward_hook)  # 对net.features这一层注册反向传播
    ###############3

    # transforms
    if 'vote' in cfg.datatransforms:
        voting_transform = build_transforms_from_cfg('vote', cfg.datatransforms)
    else:
        voting_transform = None



    pretrained_path = r"D:\pythonproject\shapenetpart\log\shapenetpart\bothpthfile\checkpoint\ckpt_best.pth"
    load_checkpoint(model, pretrained_path=pretrained_path)
    validate_fn(model, testDataLoader, cfg,
                 num_votes=0,
                data_transform=voting_transform)



    wandb.finish(exit_code=True)

import copy
def get_label2color():
    # 每个种类所对应的part类别

    map = [
       [255, 130, 23],
       [0, 0, 255],
       [0, 255, 255],
       [255, 255, 0],
       [255, 0, 255],
       [100, 100, 255],
       [200, 200, 100],
       [170, 120, 200],
        [255, 0, 0],
       [200, 100, 100],
       [10, 200, 100],
        [255, 192, 203],
       [50, 50, 50],
    ]

    lab2seg = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11],
               'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21],
               'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3],
               'Pistol': [38, 39, 40],
               'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
    # 获取所要绘制点云数据的类别
    cls = []
    # 随机生成颜色映射
    idx = np.arange(0, len(map))
    idx_all = []
    np.random.seed(123)
    for _ in range(len(lab2seg)):
        np.random.shuffle(idx)
        idx_all.append(copy.deepcopy(idx))
    idx_all = np.array(idx_all)
    # 将生成的颜色映射表对应到不同种类的part类别上
    for i, key in enumerate(lab2seg.keys()):
        lab2seg[key] = dict(zip(set(lab2seg[key]), [map[idx_all[i, j]] for j in range(len(set(lab2seg[key])))]))

    all_points_color_map = np.zeros((50,3))
    for key,val in lab2seg.items():
        for subkey,subval in val.items():
            all_points_color_map[subkey] = subval


    return lab2seg, all_points_color_map

def _normalize(cams):
    """CAM normalization"""
    lowerval = cams.min(-1)
    #lowerval = lowerval.view(*lowerval.shape,1)
  #  cams.sub_(lowerval)
    cams = cams - lowerval
    maxval = cams.max(-1)
  # maxval = maxval.view(*maxval.shape, 1)
    cams = cams / maxval
    #cams.div_(maxval)
    return cams

#@torch.no_grad()
def validate(model, val_loader, cfg, num_votes=0, data_transform=None):
    number = 1
    lab2seg, all_points_color = get_label2color()  ########
    model.eval()  # set model to eval mode
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())

    lab2seg = get_label2color() ########
    object_class = 47  # 该场景下要cam的对象类别  47为table所属

    cam_file_path = "D:/pythonproject/data/visualdata/partsegcam/"

    for idx, data in pbar:

        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        target = data['y']
        cls = data['cls']
        batch_size, num_point, _ = data['pos'].size()

        ori_data = data['ori_data'] # 1 2048 7
        ori_xyz = ori_data[0,:,:3].cpu().numpy()
       # print(ori_xyz.shape)
        data['pos'] = data['pos'].permute(0, 2, 1)
        data['x'] = data['x'].permute(0, 2, 1)
        # 在第二个维度上进行拼接
        data_all = torch.cat((data['pos'], data['x']), dim=1)

        label_one_hot = torch.eye(16)[data['cls'].cpu().data.numpy(),]
        label_one_hot = label_one_hot.cuda()
        label_one_hot = label_one_hot.squeeze(1)


        logits = model(data_all, label_one_hot)


       # print(logits.shape)
        preds = logits.contiguous().cpu().data.max(dim=1)[1].numpy()  # preds即为所求类别  B N

        ################## CAM
       # preds.squeeze(0)
        indexarr = preds[0] == object_class
        # print(seg_pred[0,indexarr,object_class])
        score = logits[0, object_class, indexarr].sum()
        model.zero_grad()
        # print(score)
        score.backward(retain_graph=True)  # 由预测类别分数反向传播


        weights = np.mean(grad[0][0].cpu().data.numpy().squeeze(0), axis=1)
        activations = feature_map[0].cpu().data.numpy().squeeze(0)  #
        #print(weights.shape)  50
        #print(activations.shape)  50 2048
        weights = weights.reshape(-1, 1) #50 1
        grad_cam = (weights * activations).sum(0)  # 2048
       # print(grad_cam.shape)
        grad_cam = np.maximum(grad_cam, 0)  # ReLU
        grad_cam = _normalize(grad_cam)

        grad_cam = ((255 * grad_cam).astype(np.uint8))
        grad_cam = grad_cam.reshape(grad_cam.shape[0], 1)  # 2048 1
     #   np.set_printoptions(threshold=np.inf)
     #   print(grad_cam)
        # grad_cam = 1 - grad_cam
        heatmap = cv2.applyColorMap(grad_cam, cv2.COLORMAP_JET) # 2048 1 3
      #  print(heatmap)
        heatmap = heatmap.reshape(grad_cam.shape[0], -1) #2048 3
        heatmap[:, [0, 2]] = heatmap[:, [2, 0]]  # BGR -> RGB
        heatmap = heatmap.reshape(grad_cam.shape[0], -1) #2048 3
        ################## CAM

        cam_data = np.concatenate((ori_xyz,heatmap),axis=1)
        np.savetxt(cam_file_path + f"{number}.txt", cam_data, fmt="%.4f")
        number = number + 1




if __name__ == "__main__":
    #get_label2color()
    parser = argparse.ArgumentParser('ShapeNetPart Part segmentation training')
    parser.add_argument('--cfg', type=str, default="../../cfgs/shapenetpart/pcnetdefault.yaml", help='config file')
    #args.cfg = "D:/pythonproject/Pointnext/cfgs/shapenetpart/default.yaml"
    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)
    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)
    # init distributed env first, since logger depends on the dist info.
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.world_size > 1

    # logger
    # init log dir
    cfg.task_name = args.cfg.split('.')[-2].split('/')[-2]  # task/dataset name, \eg s3dis, modelnet40_cls
    cfg.cfg_basename = args.cfg.split('.')[-2].split('/')[-1]  # cfg_basename, \eg pointnext-xl
    tags = [
        cfg.task_name,  # task name (the folder of name under ./cfgs
        cfg.mode,
        cfg.cfg_basename,  # cfg file name
        f'ngpus{cfg.world_size}',
        f'seed{cfg.seed}',
    ]
    opt_list = [] # for checking experiment configs from logging file
    for i, opt in enumerate(opts):
        if 'rank' not in opt and 'dir' not in opt and 'root' not in opt and 'pretrain' not in opt and 'path' not in opt and 'wandb' not in opt and '/' not in opt:
            opt_list.append(opt)
    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)
    cfg.opts = '-'.join(opt_list)

    cfg.is_training = cfg.mode not in ['test', 'testing', 'val', 'eval', 'evaluation']

    if cfg.mode in ['resume', 'test', 'val']:
        resume_exp_directory(cfg, pretrained_path=cfg.pretrained_path)
        cfg.wandb.tags = [cfg.mode]
    else:
        generate_exp_directory(cfg, tags, additional_id=os.environ.get('MASTER_PORT', None))
        cfg.wandb.tags = tags
    os.environ["JOB_LOG_DIR"] = cfg.log_dir
    cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, indent=2)
        os.system('cp %s %s' % (args.cfg, cfg.run_dir))
    cfg.cfg_path = cfg_path

    # wandb config
    cfg.wandb.name = cfg.run_name


    main(0, cfg)
