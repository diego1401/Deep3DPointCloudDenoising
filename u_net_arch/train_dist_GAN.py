"""
Distributed training script for outlier detection with EDFM dataset
"""
import numpy as np

import argparse
import glob
import os
import sys
import time
import json
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import torch
import torch.nn as nn
from torchvision import transforms
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from qualitative_inference_test import denoise_shape

import data_utils as d_utils
from models import build_offset_regression, build_discriminator
# from displacement_dataset import OutlierSegmentationDataset
from offset_dataset import OffsetDataset
from utils.util import AverageMeter, accuracy, get_metrics_and_print, get_metrics_train_and_print

from utils.lr_scheduler import get_scheduler
from utils.logger import setup_logger
from utils.config import config, update_config

from data_utils import write_ply




SAVED_METRICS = ["prec","rec","f_b","miou"]
REAL_LABEL = 1
FAKE_LABEL = 1-REAL_LABEL
ALPHA = 0.01


####################################
#                                  #
#   Parsing and preparing model    #
#                                  #
####################################

def parse_option():
    parser = argparse.ArgumentParser('Offset regression training')
    parser.add_argument('--config_file',type=str,required=True)
    parser.add_argument("--local_rank", type=int, required=True,help='local rank for DistributedDataParallel')

    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--batch_size', type=int, help='batch_size')
    parser.add_argument('--num_points', type=int, help='num_points')
    parser.add_argument('--num_steps', type=int, help='num_steps')
    parser.add_argument('--base_learning_rate', type=float, help='base learning rate')
    parser.add_argument('--weight_decay', type=float, help='weight_decay')
    parser.add_argument('--epochs', type=int, help='number of training epochs')
    parser.add_argument('--start_epoch', type=int, help='used for resume')
    parser.add_argument('--job_name',type=str,required=False)
    parser.add_argument('--DEBUG', type=int, required=False, help='Whether to debug or not (i.e. 1 ply file in datasets)')
    parser.add_argument('--dataset', type=str, required=False, help='Dataset type ("PCN" or "EDFS" or "EDFL")')
    parser.add_argument('--local_aggregator', type=str, required=False, help='Local aggregator type ("pseudogrid", "pospool", "adaptiveweight", "pointwisemlp", "minkowski")')
    parser.add_argument('--diameter_percent', type=int, required=False, help='Dataset type ("PCN" or "EDFS" or "EDFL")')

    # data augmnetations
    parser.add_argument('-ar','--aug_rot', action='append', help='List of rotation augmentation angles (in rad)', required=False)
    parser.add_argument('-as','--aug_sym', action='append', help='List of symetry augmentation axis (bool)', required=False)


    # io
    parser.add_argument('--load_path', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--val_freq', type=int, default=10, help='val frequency')
    parser.add_argument('--log_dir', type=str, default='log', help='log dir [default: log]')

    # misc
    parser.add_argument("--rng_seed", type=int, default=0, help='manual seed')

    args, unparsed = parser.parse_known_args()
    print(f"Using {args.config_file} as config file")
    assert os.path.exists(f"u_net_arch/cfgs/{args.config_file}.yaml") 
    cfg = glob.glob(f"u_net_arch/cfgs/{args.config_file}.yaml")[0]

    update_config(cfg)  
    #Setting some arguments we don't change that much
    args.job_name = config.experiment_name
    args.DEBUG = 0
    args.local_aggregator = 'pseudo_grid'
    args.dataset = 'PCN'
    args.diameter_percent = 10

    
    assert config.noise_level != -1 and config.outlier_percentage != -1
    #Extracting arguments from config file
    
    print(f"Using GPU {args.local_rank}")
    config.local_rank = args.local_rank

    local_aggregator = args.local_aggregator

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
        config.data_root = 'offset_dataset/PCN_SHAPES/'



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
        config.npoints = [max(int(config.num_points/4.),1),max(int(config.num_points/16.),1),max(int(config.num_points/32.),1),max(int(config.num_points/128.),1)]



    print("RADII: in_radius={:.2f}, smallest radius={:.5f}".format(config.in_radius,config.radius))


    aug_rot = np.array([np.pi,np.pi,np.pi])
    aug_sym = np.array([0.,0.,0.])

    config.x_angle_range = float(aug_rot[0])
    config.y_angle_range = float(aug_rot[1])
    config.z_angle_range = float(aug_rot[2])

    config.augment_symmetries = [int(float(aus)) for aus in aug_sym]

    # config.scale_low = 0.7
    # config.scale_high = 1.3
    # config.noise_std = 0.001*config.in_radius*0.5
    # config.noise_clip = 0.05*config.in_radius*0.5


    config.num_workers = args.num_workers
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
    # set the data loader
    train_trans_list = [ d_utils.PointcloudToTensor(),
        d_utils.PointcloudRandomRotate(x_range=config.x_angle_range, y_range=config.y_angle_range,
                                       z_range=config.z_angle_range)]
    if config.jitter:
        print("Using jitter!")
        assert config.scale_low == 1
        print(config.scale_low)
        train_trans_list.append(d_utils.PointcloudScaleAndJitter(scale_low=config.scale_low, scale_high=config.scale_high,
                                         std=config.noise_std, clip=config.noise_clip,
                                         augment_symmetries=config.augment_symmetries))

    train_transforms = transforms.Compose(train_trans_list)

    test_transforms = transforms.Compose([
        d_utils.PointcloudToTensor(),
    ])

    dataset_sampleDl = 0

    train_dataset = OffsetDataset(input_features=config.features, katz_params=config.katz_params, 
                                katz_type=config.katz_type,
                                subsampling_parameter=dataset_sampleDl,
                                in_radius=config.in_radius, num_points=config.num_points,
                                num_steps=config.num_steps, num_epochs=config.epochs,
                                feature_drop=config.color_drop, data_root=config.data_root, transforms=train_transforms, 
                                split='train',dataset_type=config.dataset, noise_level=config.noise_level, noise_type=config.noise_type, 
                                num_points_per_shape=140000, 
                                outlier_proportion=config.outlier_percentage, DEBUG=config.DEBUG)

    val_dataset = OffsetDataset(input_features=config.features, katz_params=config.katz_params, 
                                katz_type=config.katz_type,
                                subsampling_parameter=dataset_sampleDl,
                                in_radius=config.in_radius, num_points=config.num_points,
                                num_steps=config.num_steps, num_epochs=1,
                                feature_drop=config.color_drop, data_root=config.data_root, transforms=test_transforms, 
                                split='val',dataset_type=config.dataset, noise_level=config.noise_level, noise_type=config.noise_type, 
                                num_points_per_shape=140000, 
                                outlier_proportion=config.outlier_percentage, DEBUG=config.DEBUG)


    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=False)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.batch_size,
                                               shuffle=False,
                                               num_workers=config.num_workers,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             num_workers=config.num_workers,
                                             pin_memory=True,
                                             sampler=val_sampler,
                                             drop_last=False)

    return train_loader, val_loader


def load_checkpoint(config, model, optimizer, scheduler,block):
    if block == "Discriminator":
        load_path = config.load_path_discriminator
    elif block == "Generator":
        load_path = config.load_path_generator
        
    else:
        raise ValueError(f"Block {block} not implemented in loading")

    logger.info("=> loading checkpoint '{}'".format(load_path))

    checkpoint = torch.load(load_path, map_location='cpu')
    # config.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # scheduler.load_state_dict(checkpoint['scheduler'])

    logger.info("=> loaded successfully '{}' (epoch {})".format(load_path, checkpoint['epoch']))

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(config, epoch, model, optimizer, scheduler,block):
    logger.info('==> Saving...')
    state = {
        'config': config,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, os.path.join(config.log_dir, f'{block}_current.pth'))
    if epoch % config.save_freq == 0:
        torch.save(state, os.path.join(config.log_dir, '{}_ckpt_epoch_{}.pth'.format(block,epoch)))
        logger.info("Saved in {}".format(os.path.join(config.log_dir, '{}_ckpt_epoch_{}.pth'.format(block,epoch))))

###########################
#                         #
#   Forward and losses    #
#                         #
###########################

def forward_generator(model,to_unpack):
    points,mask,features = to_unpack
    pred = model(points, mask, features)
    return torch.transpose(pred,1,2)

def aux_compute_loss_generator(config,criterion,to_unpack):
    pred, points_offsets, mask, points = to_unpack
    if config.loss == "L1":
        return criterion(pred, points_offsets, mask)
    elif config.loss in ["chamfer_L1","chamfer","chamfer_sparse","l1_chamfer_sparse",
                         "l1_chamfer_adaptive_to_chamfer","l1_chamfer_adaptive_to_l1"]:
        return criterion(pred, points_offsets, mask, points)
    else:
        raise ValueError(f"Loss {config.loss} is not implemented in training method")

def get_loss_generator(config,criterion,model,to_unpack,return_pred=False):
    points,mask,features,points_offsets = to_unpack
    # forward
    points = points.cuda(non_blocking=True)
    mask = mask.cuda(non_blocking=True)
    features = features.cuda(non_blocking=True)
    points_offsets = points_offsets.cuda(non_blocking=True)

    # forward
    pred = forward_generator(model,(points,mask,features))

    #loss
    loss = aux_compute_loss_generator(config,criterion,(pred, points_offsets, mask, points))

    if return_pred:
        return torch.transpose(pred.detach(),1,2),loss
    return loss




###########################
#                         #
#   Main train and val    #
#                         #
###########################

def main(config):
    print("Getting loader...")
    train_loader, val_loader = get_loader(config)
    n_data = len(train_loader.dataset)
    logger.info("length of training dataset: {}".format(n_data))
    n_data = len(val_loader.dataset)
    logger.info("length of validation dataset: {}".format(n_data))

    print("Creating model...")
    if config.GAN == 0:
        raise ValueError("To train w/o GAN use 'train_dist.py' file")

    model_generator,criterion_generator = build_offset_regression(config)
    model_discriminator, criterion_discriminator = build_discriminator(config)

    model_generator.cuda()
    criterion_generator.cuda()
    model_discriminator.cuda()
    criterion_discriminator.cuda()

    if config.optimizer == 'sgd':
        optimizer_generator = torch.optim.SGD(model_generator.parameters(),
                                    lr=config.batch_size * dist.get_world_size() / 8 * config.base_learning_rate,
                                    momentum=config.momentum,
                                    weight_decay=config.weight_decay)
        
    elif config.optimizer == 'adam':
        optimizer_generator = torch.optim.Adam(model_generator.parameters(),
                                     lr=config.base_learning_rate,
                                     weight_decay=config.weight_decay)
    elif config.optimizer == 'adamW':
        optimizer_generator = torch.optim.AdamW(model_generator.parameters(),
                                      lr=config.base_learning_rate,
                                      weight_decay=config.weight_decay)
    else:
        raise NotImplementedError("Optimizer {} not supported".format(config.optimizer))

    # SGD always for discriminator
    optimizer_discriminator = torch.optim.SGD(model_discriminator.parameters(),
                                    lr=config.batch_size * dist.get_world_size() / 8 * config.base_learning_rate,
                                    momentum=config.momentum,
                                    weight_decay=config.weight_decay)

    scheduler_generator = get_scheduler(optimizer_generator, len(train_loader), config)
    scheduler_discriminator = get_scheduler(optimizer_discriminator, len(train_loader), config)

    print(f"Distributing model... local rank {config.local_rank}")
    model_generator = DistributedDataParallel(model_generator, device_ids=[config.local_rank], broadcast_buffers=False)
    model_discriminator = DistributedDataParallel(model_discriminator, device_ids=[config.local_rank], broadcast_buffers=False)
    
    # optionally resume from a checkpoint
    if config.load_path_generator:
        assert os.path.isfile(config.load_path_generator)
    
        load_checkpoint(config, model_generator, optimizer_generator, scheduler_generator,block="Generator")
        logger.info("==> checking loaded ckpt")
        print("Loading previous model!")

    if config.load_path_discriminator:
        assert os.path.isfile(config.load_path_discriminator)
        load_checkpoint(config, model_discriminator, optimizer_discriminator,scheduler_discriminator,block="Discriminator")
        logger.info("==> checking loaded ckpt")
        print("Loading previous model!")
        

    # tensorboard
    if dist.get_rank() == 0:
        summary_writer = SummaryWriter(log_dir=os.path.join(config.log_dir.replace(config.job_name,""), "TENSORBOARD_SUMMARIES", config.job_name))
    else:
        summary_writer = None

    # routine
    print("Starting routine...")
    for epoch in range(config.start_epoch, config.epochs + 1):
    # for epoch in range(1):
        print(f"epoch {epoch}")
        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)
        train_loader.dataset.epoch = epoch - 1
        tic = time.time()
        loss = train(epoch, train_loader, model_generator,model_discriminator,criterion_generator, criterion_discriminator, 
                     optimizer_generator, optimizer_discriminator, scheduler_generator, scheduler_discriminator, config)



        logger.info('epoch {}, total time {:.2f}, lr {:.5f}'.format(epoch,
                                                                    (time.time() - tic),
                                                                    optimizer_generator.param_groups[0]['lr']))
        if epoch % config.val_freq == 0:
            loss_val = validate(epoch, val_loader, model_generator,criterion_generator, config)
            if summary_writer is not None:
                summary_writer.add_scalar('loss_val', loss_val, epoch)

        if dist.get_rank() == 0:
            # save model
            save_checkpoint(config, epoch, model_generator, optimizer_generator, scheduler_generator,block="generator")
            save_checkpoint(config, epoch, model_discriminator,optimizer_discriminator, scheduler_discriminator,block="discriminator")

        # if epoch % config.val_freq == 0:
        #     # Checking results
        #     config.epoch_model_used = epoch
        #     denoise_shape(config,unpack=epoch)

        if summary_writer is not None:
            summary_writer.add_scalar('loss_train', loss, epoch)
            summary_writer.add_scalar('learning_rate', optimizer_generator.param_groups[0]['lr'], epoch)

        # torch.cuda.empty_cache()

    loss_val = validate(epoch, val_loader, model_generator,criterion_generator, config)
    if summary_writer is not None:
        summary_writer.add_scalar('loss_val', loss_val, epoch)


# From https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

def update_GAN(config,model_generator,model_discriminator,criterion_discriminator,criterion_generator,
                         optimizer_generator, optimizer_discriminator,to_unpack):

    points,mask,features,points_offsets = to_unpack

    points = points.cuda(non_blocking=True)
    mask = mask.cuda(non_blocking=True)
    # features = features.cuda(non_blocking=True)
    points_offsets = points_offsets.cuda(non_blocking=True)
    b_size = points.size(0)

   
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    # Train with all-real batch
    model_discriminator.zero_grad()
    # label = torch.full((b_size,), REAL_LABEL, dtype=torch.float).cuda(non_blocking=True)
    # label *= (1-((torch.rand(b_size) < 0.05) * 1).cuda(non_blocking=True))

    # # Real Points
    # clean_points = points + points_offsets
    # clean_features = clean_points.transpose(1,2).clone()
    # # Forward pass real batch through D
    # output = model_discriminator(clean_points, mask, clean_features).view(-1)
    # err_real = criterion_discriminator(output, label) * ALPHA
    # err_real.backward()

    # # Fake Points
    # # fake_points = points
    # pred_offsets = model_generator(points, mask, features).transpose(1,2)
    # fake_points = points + pred_offsets.detach()
    # fake_features = fake_points.transpose(1,2).clone()
    # label.fill_(FAKE_LABEL)
    # label += ((torch.rand(b_size) < 0.05) * 1).cuda(non_blocking=True)

    # output = model_discriminator(fake_points, mask, fake_features).view(-1)
    # err_fake = criterion_discriminator(output, label)* ALPHA
    # err_fake.backward()

    # # Getting accuracy on the fake patches

    # output = (output > 0.5).float()
    # accuracy_discriminator = 1 - torch.abs(output-label).mean()

    #!

    label_clean = torch.full((b_size,), REAL_LABEL, dtype=torch.float).cuda(non_blocking=True)
    label_noisy = torch.full((b_size,), FAKE_LABEL, dtype=torch.float).cuda(non_blocking=True)

    clean_points = points + points_offsets

    pred_offsets = model_generator(points, mask, features).transpose(1,2)
    noisy_points = points + pred_offsets.detach()
    # with torch.no_grad():
    #     pred_offsets = model_generator(points, mask, features).transpose(1,2)
    # noisy_points = points + pred_offsets.detach()

    train_points = torch.cat((clean_points, noisy_points))
    train_features = train_points.transpose(-2,-1).clone()
    train_mask = torch.cat((mask,mask))
    train_label = torch.cat((label_clean,label_noisy))
    
    output = model_discriminator(train_points, train_mask, train_features).view(-1)
    errD = criterion_discriminator(output, train_label)*ALPHA
    errD.backward()

    #!

    # Update D
    optimizer_discriminator.step()

    output = (output > 0.5).float()
    accuracy_discriminator = 1 - torch.abs(output-train_label).mean()

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    model_generator.zero_grad()

    label = torch.full((b_size,), REAL_LABEL, dtype=torch.float).cuda(non_blocking=True)
    label *= (1-((torch.rand(b_size) < 0.05) * 1).cuda(non_blocking=True))
    predicted_points = points + pred_offsets
    predicted_features = predicted_points.transpose(1,2).clone()

    output = model_discriminator(predicted_points, mask, predicted_features).view(-1)
    # Calculate G's loss based on this output
    errG1 = criterion_discriminator(output, label) 
    errG2 =  aux_compute_loss_generator(config,criterion_generator,(pred_offsets, points_offsets, mask, points))
    # predicted_distance = points + pred_offsets
    # err_min_dist = torch.max(torch.cdist(clean_points,predicted_distance,p=2))
    errG = errG1* ALPHA + errG2
    # + err_min_dist/config.in_radius
    # Calculate gradients for G
    errG.backward()
    
    optimizer_generator.step()


    return accuracy_discriminator,errG1,errG2


def train(epoch, train_loader, model_generator,model_discriminator,criterion_generator, criterion_discriminator,
          optimizer_generator, optimizer_discriminator,  scheduler_generator, scheduler_discriminator, config):
    """
    One epoch training
    """
    model_generator.train()
    model_discriminator.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter_l1 = AverageMeter()
    loss_meter_bce = AverageMeter()
    accuracy_meter = AverageMeter()


    end = time.time()
    for idx,(points, mask, features,
                       points_labels, points_offsets, current_cloud_index, input_inds) in enumerate(train_loader):

        data_time.update(time.time() - end)
        bsz = points.size(0)
        
        accuracy,err_discr,err_l1  = update_GAN(config,model_generator,model_discriminator,
                    criterion_discriminator,criterion_generator,
                    optimizer_generator, optimizer_discriminator,
                    (points,mask,features,points_offsets))

        if not config.freeze_gen:
            scheduler_generator.step()
        scheduler_discriminator.step()

        # update meters
        loss_meter_l1.update(err_l1.item(), bsz)
        loss_meter_bce.update(err_discr.item(), bsz)
        accuracy_meter.update(accuracy.item(),bsz)

        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % config.print_freq == 0:
            logger.info('Train: [{}/{}][{}/{}]\t'.format(epoch,config.epochs + 1, idx, len(train_loader))+
                        'T {:.3f} ({:.3f})\t'.format(batch_time.val, batch_time.avg)+
                        'DT {:.3f} ({:.3f})\t'.format(data_time.val, data_time.avg)+
                        'loss Generator L1 {:.3f} ({:.3f}) '.format(loss_meter_l1.val, loss_meter_l1.avg)+
                        'loss Discr for G {:.3f} ({:.3f}) '.format(loss_meter_bce.val, loss_meter_bce.avg)+
                        'Accuracy Discriminator {:.3f} ({:.3f})'.format(accuracy_meter.val, accuracy_meter.avg))
    return loss_meter_l1.avg


def validate(epoch,val_loader, model_generator, criterion_generator, config):
    """
    One epoch validating
    """

    batch_time = AverageMeter()
    losses_generator = AverageMeter()

    model_generator.eval()
    with torch.no_grad():
        end = time.time()
        val_loader.dataset.epoch = 0
        for idx, (points, mask, features,
                   points_labels, points_offsets, current_cloud_index, input_inds) in enumerate(val_loader):

            loss = get_loss_generator(config,criterion_generator,model_generator,((points,mask,features,points_offsets)))

            # measure elapsed time
            losses_generator.update(loss.item(), points.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if idx % config.print_freq == 0:
                logger.info(
                    'Test: [{}/{}]\t'.format(idx,len(val_loader))+
                    'Time {:.3f} ({:.3f})\t'.format(batch_time.val, batch_time.avg)+
                    'Loss {:.4f} ({:.4f})'.format(losses_generator.val, losses_generator.avg))

   
    

    return losses_generator.avg


if __name__ == "__main__":
    opt, config = parse_option()
    torch.cuda.set_device(config.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    os.makedirs(opt.log_dir, exist_ok=True)
    os.environ["JOB_LOG_DIR"] = config.log_dir
    print("RANK = {}".format(dist.get_rank()))
    logger = setup_logger(output=config.log_dir, distributed_rank=dist.get_rank(), name="PCN")
    if dist.get_rank() == 0:
        path = os.path.join(config.log_dir, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(opt), f, indent=2)
            json.dump(vars(config), f, indent=2)
        logger.info("Full config saved to {}".format(path))
    print(f"Using GPU {torch.cuda.current_device()}")
    main(config)



    # fake_points = points + pred_offsets.detach()
    # train_points = torch.cat((clean_points,fake_points)).clone()
    # train_features = train_points.transpose(-2,-1).clone()
    # train_mask = torch.cat((mask,mask)).clone()
    # train_label = torch.cat((label_clean,label_noisy)).clone()


