"""
Distributed evaluation script for outlier detection with EDFM dataset
"""
import argparse
import os
import sys
import time
import json
import random
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import torch
import torch.nn as nn
from torchvision import transforms
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel

import pp_data_utils as d_utils
from displacement_dataset import OutlierSegmentationDataset
from models import build_offset_regression
from offset_dataset import OffsetDataset
import glob
from utils.util import AverageMeter, get_metrics_and_print, get_metrics_train_and_print
from utils.logger import setup_logger
from utils.config import config, update_config

from sklearn.neighbors import KDTree

from data_utils import write_ply

def softmax(x,axis=0):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x,axis=axis,keepdims=True))
    return e_x / e_x.sum(axis=axis,keepdims=True)


def parse_option():
    parser = argparse.ArgumentParser('Outlier segmentation TEST')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--batch_size', type=int, help='batch_size')
    parser.add_argument('--num_points', type=int, help='num_points')
    parser.add_argument('--num_steps', type=int, help='num_steps')
    parser.add_argument('--base_learning_rate', type=float, help='base learning rate')
    parser.add_argument('--weight_decay', type=float, help='weight_decay')
    parser.add_argument('--epochs', type=int, help='number of training epochs')
    parser.add_argument('--start_epoch', type=int, help='used for resume')
    parser.add_argument('--job_name',type=str,required=True)

    parser.add_argument('--DEBUG', type=int, required=True, help='Whether to debug or not (i.e. 1 ply file in datasets)')

    parser.add_argument('--dataset', type=str, required=True, help='Dataset type ("PCN" or "EDFS" or "EDFL")')
    parser.add_argument('--local_aggregator', type=str, required=True, help='Local aggregator type ("pseudogrid", "pospool", "adaptiveweight", "pointwisemlp", "minkowski")')
    parser.add_argument('--diameter_percent', type=int, required=True, help='Dataset type ("PCN" or "EDFS" or "EDFL")')

    # data augmnetations
    parser.add_argument('-ar','--aug_rot', action='append', help='List of rotation augmentation angles (in rad)', required=True)
    parser.add_argument('-as','--aug_sym', action='append', help='List of symetry augmentation axis (bool)', required=True)


    # io
    parser.add_argument('--load_path', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--val_freq', type=int, default=10, help='val frequency')
    parser.add_argument('--log_dir', type=str, default='log', help='log dir [default: log]')

    # misc
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument("--rng_seed", type=int, default=0, help='manual seed')

    args, unparsed = parser.parse_known_args()


    local_aggregator = args.local_aggregator

    # infer cfg from local aggregator given ("pool","grid","adap","mlp")
    cfg = glob.glob("./cfgs/*{}*.yaml".format(local_aggregator))[0]

    update_config(cfg)

    print("WARNING: NO FEATURE PIPELINE FOR NOW")
    config.features = []
    config.katz_params = []

    config.dataset = args.dataset

    if "EDF" in config.dataset:
        shape_diameter = 10. # 5m radius spherical scenes
        sampleDl = 0.04
        config.data_root = "../EDFM_dataset/"
    else:
        shape_diameter = 1. # normalized shapes
        sampleDl = 0
        config.data_root = "../pointCleanNetOutliersDataset_PLY/"



    config.DEBUG = args.DEBUG

    config.in_radius = 0.5*shape_diameter*args.diameter_percent/100.
    config.sampleDl = config.in_radius/32. # in total, 32 upsamples --> first layer = pointwise mlp (no downsampling)
    if config.in_radius==2.:
        config.radius = 0.1
    else:
        config.radius = max(config.in_radius*np.sqrt(3)/32.,0.025) # before, 0.1

    if args.num_points==15000:
        config.nsamples = [26,31,38,41,39]
        config.npoints = [4096,1152,304,88]
    else:
        config.nsamples = [2*26,int(1.5*26),int(1.25*26),26,26]
        config.npoints = [max(int(args.num_points/4.),1),max(int(args.num_points/16.),1),max(int(args.num_points/32.),1),max(int(args.num_points/128.),1)]



    print("RADII: in_radius={:.2f}, smallest radius={:.5f}".format(config.in_radius,config.radius))

    config.x_angle_range = float(args.aug_rot[0])
    config.y_angle_range = float(args.aug_rot[1])
    config.z_angle_range = float(args.aug_rot[2])
    config.augment_symmetries = [int(float(aus)) for aus in args.aug_sym]
    config.scale_low = 0.7
    config.scale_high = 1.3
    config.noise_std = 0.001*config.in_radius*0.5
    config.noise_clip = 0.05*config.in_radius*0.5


    config.num_workers = args.num_workers
    config.load_path = args.load_path
    config.print_freq = args.print_freq
    config.save_freq = args.save_freq
    config.val_freq = args.val_freq
    config.rng_seed = args.rng_seed

    config.input_features_dim = 0
    for f in config.features:
        if f=="normal":
            config.input_features_dim += 3
        if "katz" in f:
            config.input_features_dim += len(config.katz_params)
        if f=="intensity":
            config.input_features_dim += 1
    rem = abs(3 - config.input_features_dim%3)%3

    config.input_features_dim += rem

    config.local_rank = args.local_rank

    # config.log_dir = os.path.join(args.log_dir, '{}_{}'.format(args.job_name,int(time.time())))
    config.log_dir = os.path.join(args.log_dir, args.job_name)

    config.job_name = args.job_name

    if args.batch_size:
        config.batch_size = args.batch_size
    if args.num_points:
        config.num_points = args.num_points
    if args.num_steps:
        config.num_steps = args.num_steps
    if args.base_learning_rate:
        config.base_learning_rate = args.base_learning_rate
    if args.weight_decay:
        config.weight_decay = args.weight_decay
    if args.epochs:
        config.epochs = args.epochs
    if args.start_epoch:
        config.start_epoch = args.start_epoch

    torch.manual_seed(args.rng_seed)
    torch.cuda.manual_seed_all(args.rng_seed)
    random.seed(args.rng_seed)
    np.random.seed(args.rng_seed)

    return args, config


def get_loader(config):
    test_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])

    if "EDF" not in config.dataset:
        dataset_sampleDl = config.sampleDl
    else:
        dataset_sampleDl = 0

    # val_dataset = OutlierSegmentationDataset(input_features=config.features, katz_params=config.katz_params, katz_type=config.katz_type,
    #                        subsampling_parameter=dataset_sampleDl, feature_drop=config.color_drop,
    #                        in_radius=config.in_radius, num_points=config.num_points,
    #                        num_steps=config.num_steps, num_epochs=21,
    #                        data_root=config.data_root, transforms=test_transforms,
    #                        split='test',dataset_type=config.dataset, DEBUG=config.DEBUG)

    val_dataset = OffsetDataset(input_features=config.features, katz_params=config.katz_params, 
                                katz_type=config.katz_type,
                                subsampling_parameter=dataset_sampleDl,
                                in_radius=config.in_radius, num_points=config.num_points,
                                num_steps=config.num_steps, num_epochs=config.epochs,
                                feature_drop=0, data_root=None, transforms=None, 
                                split='test',dataset_type="PCN", noise_level=5.*1e-3, noise_type="gaussian", 
                                num_points_per_shape=140000, 
                                outlier_proportion=0.4, DEBUG=False)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             num_workers=config.num_workers,
                                             pin_memory=True,
                                             sampler=val_sampler,
                                             drop_last=False)

    return val_loader


def load_checkpoint(config, model):
    logger.info("=> loading checkpoint '{}'".format(config.load_path))

    checkpoint = torch.load(config.load_path, map_location='cpu')
    config.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])

    logger.info("=> loaded successfully '{}' (epoch {})".format(config.load_path, checkpoint['epoch']))

    del checkpoint
    torch.cuda.empty_cache()


def main(config):
    val_loader = get_loader(config)
    n_data = len(val_loader.dataset)
    logger.info(f"length of validation dataset: {n_data}")

    model, criterion = build_offset_regression(config)
    model.cuda()
    criterion.cuda()

    model = DistributedDataParallel(model, device_ids=[config.local_rank], broadcast_buffers=False)

    # optionally resume from a checkpoint
    if config.load_path:
        assert os.path.isfile(config.load_path)
        load_checkpoint(config, model)
        logger.info("==> checking loaded ckpt")
        validate('resume', val_loader, model, criterion, config, num_votes=1)#20 votes ?

    #validate('Last', val_loader, model, criterion, config, num_votes=20)


def validate(epoch, test_loader, model, criterion, config, num_votes=10):
    vote_offset_sum = [np.zeros((3, l.shape[0]), dtype=np.float32) for l in
                       test_loader.dataset.clouds_points_labels]
    vote_counts = [np.zeros((1, l.shape[0]), dtype=np.float32) + 1e-6 for l in
                   test_loader.dataset.clouds_points_labels]
    vote_offset = [np.zeros((3, l.shape[0]), dtype=np.float32) for l in
                   test_loader.dataset.clouds_points_labels]
    validation_proj = test_loader.dataset.projections
    validation_labels = test_loader.dataset.clouds_points_labels

    val_proportions = np.zeros(config.num_classes, dtype=np.float32)
    for label_value in range(config.num_classes):
        val_proportions[label_value] = np.sum(
            [np.sum(labels == label_value) for labels in test_loader.dataset.clouds_points_labels])

    batch_time = AverageMeter()
    losses = AverageMeter()

    model.eval()
    with torch.no_grad():
        end = time.time()
        RT = d_utils.BatchPointcloudRandomRotate(x_range=config.x_angle_range, y_range=config.y_angle_range,
                                                 z_range=config.z_angle_range)
        TS = d_utils.BatchPointcloudScaleAndJitter(scale_low=config.scale_low, scale_high=config.scale_high,
                                                   std=config.noise_std, clip=config.noise_clip,
                                                   augment_symmetries=config.augment_symmetries)

        for v in range(1):#num_votes):
            test_loader.dataset.epoch = v
            for idx, (points, mask, features, points_labels, cloud_label, input_inds) in enumerate(test_loader):
                # augment for voting
                if v > 0:
                    points = RT(points)
                    points = TS(points)
                    if config.input_features_dim <= 5:
                        pass
                    elif config.input_features_dim == 6:
                        color = features[:, :3, :]
                        features = torch.cat([color, points.transpose(1, 2).contiguous()], 1)
                    elif config.input_features_dim == 7:
                        color_h = features[:, :4, :]
                        features = torch.cat([color_h, points.transpose(1, 2).contiguous()], 1)
                    else:
                        raise NotImplementedError(
                            f"input_features_dim {config.input_features_dim} in voting not supported")

                # forward
                points = points.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)
                features = features.cuda(non_blocking=True)
                points_labels = points_labels.cuda(non_blocking=True)
                cloud_label = cloud_label.cuda(non_blocking=True)
                input_inds = input_inds.cuda(non_blocking=True)

                pred = model(points, mask, features)
                loss = criterion(pred, points_labels, mask)
                losses.update(loss.item(), points.size(0))

                # collect
                bsz = points.shape[0]
                for ib in range(bsz):
                    mask_i = mask[ib].cpu().numpy().astype(np.bool)
                    logits = pred[ib].cpu().numpy()[:, mask_i]
                    inds = input_inds[ib].cpu().numpy()[mask_i]
                    c_i = cloud_label[ib].item()
                    vote_offset_sum[c_i][:, inds] = vote_offset_sum[c_i][:, inds] + logits
                    vote_counts[c_i][:, inds] += 1
                    vote_offset[c_i] = vote_offset[c_i] / vote_counts[c_i]

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # if (idx % config.print_freq == 0):
                #     metrics = get_metrics_train_and_print(logger.info, config.num_classes, pred.detach().cpu().numpy(), points_labels.int().detach().cpu().numpy(), mask.bool().detach().cpu().numpy(), verbose=1)

        for name,points,logits,proj,lbls in zip(test_loader.dataset.cloud_names,test_loader.dataset.clouds_points,vote_offset,validation_proj,validation_labels):
            pos = points[proj,:]
            preds = np.argmax(logits[:, proj], axis=0).astype(np.int32)
            probas = softmax(logits[:,proj], axis=0)

            uncertain_indica = (probas[1,:]==0.5).astype(np.bool).squeeze()#((probas[1,:]>=0.5-1e-5).astype(np.float32)*(probas[1,:]<=0.5+1e-5).astype(np.float32)).astype(np.bool).squeeze()

            if np.sum(uncertain_indica)>0:
                certain_indica = (1-uncertain_indica.astype(np.float32)).astype(np.bool)

                uncertain_pos = pos[uncertain_indica]

                certain_pos = pos[certain_indica]
                probas_certain = probas[:,certain_indica]
                preds_certain = preds[certain_indica]

                certain_tree = KDTree(certain_pos,leaf_size=50)
                nearest_certain = certain_tree.query(uncertain_pos,1)[1][:,0]

                probas[:,uncertain_indica] = probas_certain[:,nearest_certain]
                preds[uncertain_indica] = preds_certain[nearest_certain]

            save_dir = os.path.dirname(os.path.join(config.log_dir,name))
            os.makedirs(save_dir, exist_ok=True)
            write_ply("{}/{}_E{:02d}_votes_{:02d}.ply".format(config.log_dir,name,config.start_epoch-1,num_votes),[pos,probas[1,:].squeeze(),preds.squeeze(),lbls.squeeze()],["vertex","probas_01","y_hat","GT"])
        # metrics = get_metrics_and_print(logger.info, config.num_classes, vote_offset, validation_proj, validation_labels, verbose=True)
    # return metrics


if __name__ == "__main__":
    opt, config = parse_option()

    torch.cuda.set_device(config.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    os.makedirs(opt.log_dir, exist_ok=True)
    os.environ["JOB_LOAD_DIR"] = os.path.dirname(config.load_path)

    logger = setup_logger(output=config.log_dir, distributed_rank=dist.get_rank(), name="EDF_eval")

    main(config)
