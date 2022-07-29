from ast import Str
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

import data_utils as d_utils
from models import build_offset_regression_PCN, build_offset_regression,build_complete_denoising
from offset_dataset import OffsetDataset
from utils.util import AverageMeter, get_metrics_and_print, get_metrics_train_and_print
from utils.lr_scheduler import get_scheduler
from utils.logger import setup_logger
from utils.config import config, update_config
from tqdm import tqdm

from models import chamfer_distance
# from pytorch3d.loss import chamfer_distance

from data_utils import write_ply
import time



def parse_option():
    parser = argparse.ArgumentParser('Offset regression qualitative test')

    parser.add_argument('--config_file',type=str,required=True)
    parser.add_argument('--epoch_for_model',type=int,required=True,help='Epoch on which the model was saved')
    parser.add_argument("--local_rank", type=int, required=True,help='local rank for DistributedDataParallel')

    parser.add_argument('--DEBUG', type=int, default=0, help='Whether to debug or not (i.e. 1 ply file in datasets)')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--batch_size', type=int, help='batch_size')
    parser.add_argument('--num_points', type=int, help='num_points')
    parser.add_argument('--num_steps', type=int, help='num_steps')
    parser.add_argument('--base_learning_rate', type=float, help='base learning rate')
    parser.add_argument('--weight_decay', type=float, help='weight_decay')
    parser.add_argument('--epochs', type=int, help='number of training epochs')
    parser.add_argument('--start_epoch', type=int, help='used for resume')
    parser.add_argument('--job_name',type=str,required=False)
    

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

    assert config.experiment_name != ''
    assert config.noise_level != -1 and config.outlier_percentage != -1
    #Extracting arguments from config file
    args.job_name = config.experiment_name
    print(f"Using GPU {args.local_rank}")
    config.local_rank = args.local_rank


    assert config.epoch_model_used != -1
    config.epoch_model_used = args.epoch_for_model
    #Setting some arguments we don't change that much
    args.local_aggregator = 'pseudo_grid'
    args.dataset = 'PCN'
    args.diameter_percent = 10
    config.aug_rot = np.array([np.pi,np.pi,np.pi])
    config.aug_sym = np.array([1.,1.,0.])


    
    # infer cfg from local aggregator given ("pool","grid","adap","mlp")
    local_aggregator = args.local_aggregator
    # cfg = glob.glob("u_net_arch/cfgs/*{}*.yaml".format(local_aggregator))
    
    
    # cfg = glob.glob("./cfgs/offset_reg.yaml")[0]
    print("WARNING: NO FEATURE PIPELINE FOR NOW")
    config.features = []
    config.katz_params = []

    config.dataset = args.dataset

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

    config.x_angle_range = float(config.aug_rot[0])
    config.y_angle_range = float(config.aug_rot[1])
    config.z_angle_range = float(config.aug_rot[2])
    config.augment_symmetries = [int(float(aus)) for aus in config.aug_sym]

    print("Augmentation used")
    print(f"Rotation range ({config.x_angle_range},{config.y_angle_range},{config.z_angle_range})")
    print(f"Symmetry {config.augment_symmetries}")
    # config.scale_low = 0.7
    # config.scale_high = 1.3
    # config.noise_std = 0.001*config.in_radius*0.5
    # config.noise_clip = 0.05*config.in_radius*0.5


    config.num_workers = args.num_workers
    # config.load_path = args.load_path
    config.print_freq = args.print_freq
    config.save_freq = args.save_freq
    config.val_freq = args.val_freq
    config.rng_seed = args.rng_seed

    # config.input_features_dim = 0
    # for f in config.features:
    #     if f=="normal":
    #         config.input_features_dim += 3
    #     if "katz" in f:
    #         config.input_features_dim += len(config.katz_params)
    #     if f=="intensity":
    #         config.input_features_dim += 1
    # rem = abs(3 - config.input_features_dim%3)%3

    # config.input_features_dim += rem

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
        d_utils.PointcloudToTensor(),
    ])

    dataset_sampleDl = 0

    dataset = OffsetDataset(input_features=config.features, katz_params=config.katz_params, 
                                katz_type=config.katz_type,
                                subsampling_parameter=dataset_sampleDl,
                                in_radius=config.in_radius, num_points=config.num_points,
                                num_steps=config.num_steps, num_epochs=1,
                                feature_drop=config.color_drop, data_root=config.data_root, transforms=test_transforms, 
                                split='qualitative_test',dataset_type=config.dataset, noise_level=config.noise_level, noise_type="gaussian", 
                                num_points_per_shape=140000, 
                                outlier_proportion=config.outlier_percentage, DEBUG=config.DEBUG,architecture=config.architecture,
                                sampleDl_patches=config.sample_Dl_patches, fourier_features=config.fourier_features)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             num_workers=config.num_workers,
                                             pin_memory=True,
                                             sampler=sampler,
                                             drop_last=False)

    return dataset,loader

def loading_model(config,loss=None, cuda=True):
    # We force the same loss
    if loss is not None:
        config.loss = loss
    #######
    if config.architecture == 'U-Net':
        model,criterion = build_offset_regression(config)
    elif config.architecture == 'PCN':
        model,criterion = build_offset_regression_PCN(config)
    elif config.architecture == 'U-Net_full':
        model,criterion = build_complete_denoising(config)
        criterion = None
    else:
        raise ValueError(f"Architecture {config.architecture} not implemented")
    print(f"Distributing model... local rank {config.local_rank}")

    model.cuda()
    if criterion is not None:
        criterion.cuda()
    model = DistributedDataParallel(model, device_ids=[config.local_rank], broadcast_buffers=False)

    print("Loading model...")
    if config.GAN:
        config.load_path = f"./log/{config.experiment_name}/generator_ckpt_epoch_{config.epoch_model_used}.pth"
    else:
        config.load_path = f"./log/{config.experiment_name}/ckpt_epoch_{config.epoch_model_used}.pth"
    print(f"from {config.load_path}")
    checkpoint = torch.load(config.load_path, map_location='cpu')
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model,criterion

#################
#               #
#     U Net     #
#               #
#################

def aux_compute_loss(config,criterion,to_unpack):
    pred, points_offsets, mask, points = to_unpack
    if config.loss == "l1":
        return criterion(pred, points_offsets, mask)
    elif config.loss == "chamfer_L1" or config.loss == "chamfer":
        return criterion(pred, points_offsets, mask, points)
    else:
        raise ValueError(f"Loss {config.loss} is not implemented in training method")

def offset_prediction_U_Net(loader, model,criterion, config):
    """
    Predicting offsets given a model and the shapes
    """
    vote_offset_sum = [np.zeros((l.shape[0],3), dtype=np.float32) for l in
                       loader.dataset.clouds_points_labels]
    counts = [np.zeros((l.shape[0],1), dtype=np.float32) + 1e-7 for l in
                   loader.dataset.clouds_points_labels]
    vote_clean = [np.zeros((l.shape[0],3), dtype=np.float32) for l in
                   loader.dataset.clouds_points_labels]

    vote_scalar_field = [np.zeros((l.shape[0],1), dtype=np.float32) for l in
                       loader.dataset.clouds_points_labels]

    print(vote_offset_sum[0].shape)
    print(vote_scalar_field[0].shape)

                   
    N_clouds = len(vote_offset_sum)
    print(f"Processing {N_clouds} cloud/s")
    model.eval()
    
    with torch.no_grad():
        for idx, (points, mask, features,
                   points_labels, points_offsets, current_cloud_index, input_inds)  in tqdm(enumerate(loader)):

            points = points.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            features = features.cuda(non_blocking=True)
            points_offsets = points_offsets.cuda(non_blocking=True)


            # Forward
            pred = model(points, mask, features)

            # Ensembling all predictions
            batch_size = points.shape[0]
            mask = mask.cpu().numpy().astype(bool)
            pred = pred.cpu().numpy()
            input_inds = input_inds.cpu().numpy()
            points = points.cpu().numpy()
            points_offsets = points_offsets.cpu().numpy()

            for ib in range(batch_size):
                mask_i = mask[ib]
                batch_pred = pred[ib][:, mask_i].T
                pts = points[ib][mask_i,:]
                pts_offset = points_offsets[ib][mask_i,:]
                inds = input_inds[ib][mask_i]
                c_i = current_cloud_index[ib].item()

                vote_offset_sum[c_i][inds, :] += batch_pred
                vote_scalar_field[c_i][inds, :] += np.linalg.norm(batch_pred)
                vote_clean[c_i][inds, :] += pts_offset
                counts[c_i][inds, :] += 1
            

        for c_i in range(N_clouds):    
            vote_offset_sum[c_i] = vote_offset_sum[c_i]/counts[c_i]
            vote_clean[c_i] = vote_clean[c_i]/counts[c_i]
            vote_scalar_field[c_i] = vote_scalar_field[c_i] /counts[c_i]

    return vote_offset_sum, vote_clean,vote_scalar_field


#################
#               #
#     PCN       #
#               #
#################

def offset_prediction_PCN(loader, model,criterion, config):
    """
    Predicting offsets given a model and the shapes
    """
    vote_offset_sum = [np.zeros((l.shape[0],3), dtype=np.float32) for l in
                       loader.dataset.clouds_points_labels]
    counts = [np.zeros((l.shape[0],1), dtype=np.float32) + 1e-7 for l in
                   loader.dataset.clouds_points_labels]
    vote_clean = [np.zeros((l.shape[0],3), dtype=np.float32) for l in
                   loader.dataset.clouds_points_labels]

    N_clouds = len(vote_offset_sum)
    print(f"Processing {N_clouds} cloud/s")
    model.eval()
    
    with torch.no_grad():
        for idx, (points, center_point_ind, points_offsets, current_cloud_index, input_inds)  in tqdm(enumerate(loader)):

            points = points.cuda(non_blocking=True)
            points_offsets = points_offsets.cuda(non_blocking=True)


            # Forward
            pred, trans, _, _ = model(points.transpose(2, 1))
            pred = torch.bmm(pred.unsqueeze(1), trans.transpose(2, 1)).squeeze(1)
            pred = pred * config.in_radius
            # Ensembling all predictions
            batch_size = points.shape[0]
            pred = pred.cpu().numpy()
            # print(pred)
            # center_point_ind = center_point_ind.cpu().numpy()
            input_inds = input_inds.cpu().numpy()
            points_offsets = points_offsets.cpu().numpy()
            for ib in range(batch_size):
                batch_pred = pred[ib]
                pts_offset = points_offsets[ib]
                center_inds = center_point_ind[ib]
                inds = input_inds[ib]
                c_i = current_cloud_index[ib].item()

                vote_offset_sum[c_i][inds[center_inds], :] += batch_pred
                vote_clean[c_i][inds[center_inds], :] += pts_offset
                counts[c_i][inds[center_inds], :] += 1

        for c_i in range(N_clouds):    
            vote_offset_sum[c_i] = vote_offset_sum[c_i]/counts[c_i]
            vote_clean[c_i] = vote_clean[c_i]/counts[c_i]

            

    return vote_offset_sum, vote_clean

########################################
#                                      #
#     U-Net regresion + outliers       #
#                                      #
########################################

def offset_prediction_full_cleaning(loader, model):
    """
    Predicting offsets given a model and the shapes
    """
    vote_offset_sum = [np.zeros((l.shape[0],3), dtype=np.float32) for l in
                       loader.dataset.clouds_points_labels]
    counts = [np.zeros((l.shape[0],1), dtype=np.float32) + 1e-7 for l in
                   loader.dataset.clouds_points_labels]
    vote_clean = [np.zeros((l.shape[0],3), dtype=np.float32) for l in
                   loader.dataset.clouds_points_labels]

    # vote_scalar_field = [np.zeros((l.shape[0],1), dtype=np.float32) for l in
    #                    loader.dataset.clouds_points_labels]

    outlierness_sum = [np.zeros((l.shape[0],1), dtype=np.float32) for l in
                       loader.dataset.clouds_points_labels]

    outlier_labels = [np.zeros((l.shape[0],1), dtype=np.float32) for l in
                       loader.dataset.clouds_points_labels]
                   
    N_clouds = len(vote_offset_sum)
    print(f"Processing {N_clouds} cloud/s")
    model.eval()
    
    with torch.no_grad():
        for idx, (points, mask, features,
                   points_labels, points_offsets, current_cloud_index, input_inds)  in tqdm(enumerate(loader)):

            points = points.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            features = features.cuda(non_blocking=True).float()
            points_offsets = points_offsets.cuda(non_blocking=True)
            points_labels = points_labels.cuda(non_blocking=True).float()


            # Forward
            pred = model(points, mask, features).transpose(1,2)

            oi = nn.Sigmoid()(pred[...,-1])
            di = nn.Tanh()(pred[...,:-1])


            # Ensembling all predictions
            batch_size = points.shape[0]
            mask = mask.cpu().numpy().astype(bool)
            # pred = pred.cpu().numpy()
            input_inds = input_inds.cpu().numpy()
            points = points.cpu().numpy()
            points_offsets = points_offsets.cpu().numpy()
            point_labels = points_labels.cpu().numpy()
            oi = oi.cpu().numpy()
            di = di.cpu().numpy()

            for ib in range(batch_size):
                mask_i = mask[ib]
                # batch_pred = pred[ib][:, mask_i].T
                batch_outlierness = oi[ib,mask_i].reshape(-1,1)
                batch_offsets = di[ib][mask_i,:]


                # pts = points[ib][mask_i,:]
                pts_offset = points_offsets[ib][mask_i,:]
                batch_label = point_labels[ib,mask_i].reshape(-1,1)
                inds = input_inds[ib][mask_i]
                c_i = current_cloud_index[ib].item()

                vote_offset_sum[c_i][inds, :] += batch_offsets
                outlierness_sum[c_i][inds,:] += batch_outlierness
                # vote_scalar_field[c_i][inds, :] += np.linalg.norm(batch_pred)

                vote_clean[c_i][inds, :] += pts_offset
                outlier_labels[c_i][inds,:] += batch_label
                counts[c_i][inds, :] += 1

        
        
        for c_i in range(N_clouds):
            vote_clean[c_i] /= counts[c_i]
            outlier_labels[c_i] /= counts[c_i]

        # We discretize the predicted outlierness
        inlier_prediction_labels = []
        for c_i in range(N_clouds):
            inlier_prediction_labels.append(((outlierness_sum[c_i]/counts[c_i])<0.5).flatten()) # (Number of Clouds, N, 3)
            outlier_labels[c_i] = (outlier_labels[c_i]>0.5).flatten()


        for c_i in range(N_clouds):
            vote_offset_sum[c_i] = vote_offset_sum[c_i] /(counts[c_i] - outlierness_sum[c_i])

        for c_i in range(N_clouds):

            vote_offset_sum[c_i] = vote_offset_sum[c_i][inlier_prediction_labels[c_i],:]
            vote_clean[c_i] = vote_clean[c_i][outlier_labels[c_i],:]

    return (vote_offset_sum,inlier_prediction_labels),(vote_clean, outlier_labels)

def safe_make(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
        print("Created Folder",directory)

def denoising(dataset,loader,prediction,target,config,unpack,scalar_field=None):

    PATH = 'cloud_points/denoised_clouds'
    PATH = os.path.join(PATH,config.experiment_name+"_test_5e3")
    
    safe_make(PATH)

    PATH_DENOISED = os.path.join(PATH,"denoised")
    PATH_NOISY = os.path.join(PATH,"noisy")
    PATH_LOSSES = os.path.join(PATH,"losses")
    PATH_CLEAN = os.path.join(PATH,"clean")

    safe_make(PATH_DENOISED)
    safe_make(PATH_NOISY)
    safe_make(PATH_LOSSES)
    safe_make(PATH_CLEAN)

    point_clouds = loader.dataset.clouds_points
    N_clouds = len(point_clouds)

    if "full" not in config.architecture:
        
        assert point_clouds[0].shape == prediction[0].shape
        denoised_clouds = [cloud + offset for cloud,offset in zip(point_clouds,prediction)]
        clean_clouds = [cloud + offset for cloud,offset in zip(point_clouds,target)]
        noisy_clouds = [cloud for cloud in point_clouds]
    else:
        prediction, pred_outlier = prediction
        target, target_outlier = target

        
        point_cloud_no_outliers_pred = [point_clouds[c_i][pred_outlier[c_i],:] for c_i in range(N_clouds)]
        point_cloud_no_outliers_target = [point_clouds[c_i][target_outlier[c_i],:] for c_i in range(N_clouds)]

        assert point_cloud_no_outliers_pred[0].shape == prediction[0].shape
        assert point_cloud_no_outliers_target[0].shape == target[0].shape

        denoised_clouds = [cloud + offset for cloud,offset in zip(point_cloud_no_outliers_pred,prediction)]
        clean_clouds = [cloud + offset for cloud,offset in zip(point_cloud_no_outliers_target,target)]
        noisy_clouds = [cloud for cloud in point_clouds]

    
    
    

    if scalar_field is not None:
        params_name_list = ["vertex","intensity"]
    else:
        params_name_list = ["vertex"]


    for idx in range(N_clouds):
        denoised_cloud = denoised_clouds[idx]
        denoised_cloud_info = [denoised_cloud]
        if scalar_field is not None:
            denoised_cloud_info.append(scalar_field[idx])

        noisy_cloud = noisy_clouds[idx]
        cloud_name = dataset.index_to_cloud_name[idx].split("/")[1]
        clean_cloud = clean_clouds[idx]

        if unpack is None:
            filename = f"{PATH_DENOISED}/{config.experiment_name}_{cloud_name}_denoised.ply"
            print("saving",filename)
            write_ply(filename,denoised_cloud_info,params_name_list)

            filename2 = f"{PATH_NOISY}/{config.experiment_name}_{cloud_name}_noisy.ply"
            write_ply(filename2,[noisy_cloud],["vertex"])

            filename3 = f"{PATH_CLEAN}/{config.experiment_name}_{cloud_name}_clean.ply"
            write_ply(filename3,[clean_cloud],["vertex"])
        else:
            iteration_number = unpack
            filename = f"{PATH_DENOISED}/{config.experiment_name}_{cloud_name}_{iteration_number}_denoised_GAN.ply"
            print("saving",filename)
            write_ply(filename,denoised_cloud_info,params_name_list)

def denoise_shape(config,unpack=None):
    '''
        Denoises the shapes in folder "qualitative test" using the config file.

    '''
    # old_loss = config.loss
    print("Creating model...")
    model, criterion = loading_model(config)

    print("Getting loader...")
    dataset, loader = get_loader(config)
    n_data = len(loader.dataset)
    print("length of training dataset: {}".format(n_data))

    
    
    print(f"Distributing model... local rank {config.local_rank}")
    model = DistributedDataParallel(model, device_ids=[config.local_rank], broadcast_buffers=False)

    # print("Predicting and writing")
    # offset_prediction_on_k_patches(loader,model,config,criterion,k=1)

    print("Computing predictions...")
    scalar_field = None
    if config.architecture == "U-Net":
        prediction,target,scalar_field = offset_prediction_U_Net(loader, model,criterion,config)
    elif config.architecture == "PCN":
        prediction,target = offset_prediction_PCN(loader, model,criterion,config)
    elif config.architecture == "U-Net_full":
        prediction,target = offset_prediction_full_cleaning(loader,model)
    else:
        raise ValueError(f"Architecture {config.architecture} not implemented")


    print("Applying predictions and saving...")
    denoising(dataset,loader,prediction,target,config,unpack,scalar_field)
        # end = time.time()
        # f.write(f"Execution time for value {v} is: {end-start}\n")
    # config.loss = old_loss

if __name__ == "__main__":

    
    opt, config = parse_option()
    

    
    print(f'config gpu set is {config.local_rank}')
    torch.cuda.set_device(config.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    print("RANK = {}".format(dist.get_rank()))
    print(f"Using GPU {torch.cuda.current_device()}")
    denoise_shape(config)



# Trash

# def offset_prediction_U_Net(loader, model,criterion, config):
#     """
#     Predicting offsets given a model and the shapes
#     """
#     vote_offset_sum = [np.zeros((l.shape[0],3), dtype=np.float32) for l in
#                        loader.dataset.clouds_points_labels]
#     counts = [np.zeros((l.shape[0],1), dtype=np.float32) + 1e-7 for l in
#                    loader.dataset.clouds_points_labels]
#     vote_offset = [np.zeros((l.shape[0],3), dtype=np.float32) for l in
#                    loader.dataset.clouds_points_labels]
#     vote_clean = [np.zeros((l.shape[0],3), dtype=np.float32) for l in
#                    loader.dataset.clouds_points_labels]
#     vote_loss = [np.zeros((l.shape[0],1), dtype=np.float32) for l in
#                    loader.dataset.clouds_points_labels]
#     vote_loss_noisy_points = [np.zeros((l.shape[0],1), dtype=np.float32) for l in
#                    loader.dataset.clouds_points_labels]
#     N_clouds = len(vote_offset)
#     print(f"Processing {N_clouds} cloud/s")
#     model.eval()
#     mean_loss = []
#     mean_loss_noisy = []
#     with torch.no_grad():
#         for idx, (points, mask, features,
#                    points_labels, points_offsets, current_cloud_index, input_inds, center_point)  in tqdm(enumerate(loader)):

#             points = points.cuda(non_blocking=True)
#             mask = mask.cuda(non_blocking=True)
#             features = features.cuda(non_blocking=True)
#             points_offsets = points_offsets.cuda(non_blocking=True)


#             # Forward
#             pred = model(points, mask, features)

#             #loss
#             loss_config = aux_compute_loss(config,criterion,(torch.transpose(pred,1,2), points_offsets, mask, points))
#             mean_loss.append(loss_config.item())

#             # Ensembling all predictions
#             batch_size = points.shape[0]
#             mask = mask.cpu().numpy().astype(bool)
#             pred = pred.cpu().numpy()
#             input_inds = input_inds.cpu().numpy()
#             points = points.cpu().numpy()
#             points_offsets = points_offsets.cpu().numpy()

#             cd = 0
#             for ib in range(batch_size):
#                 mask_i = mask[ib]
#                 batch_pred = pred[ib][:, mask_i].T
#                 pts = points[ib][mask_i,:]
#                 pts_offset = points_offsets[ib][mask_i,:]
#                 inds = input_inds[ib][mask_i]
#                 c_i = current_cloud_index[ib].item()
#                 vote_offset_sum[c_i][inds, :] += batch_pred
#                 vote_clean[c_i][inds, :] += pts_offset
#                 counts[c_i][inds, :] += 1
#                 original_points = pts + pts_offset

#                 # print(np.std(pts_offset))
#                 predicted_points = pts + batch_pred

#                 loss = np.linalg.norm(predicted_points - original_points,ord=2,axis=1)
#                 loss /= config.in_radius

#                 loss_noise = np.linalg.norm(pts - original_points,ord=2,axis=1)
#                 loss_noise /= config.in_radius

#                 vote_loss[c_i][inds, :] += np.expand_dims(loss,1)
#                 vote_loss_noisy_points[c_i][inds,:] += np.expand_dims(loss_noise,1)

#                 loss2, _ = chamfer_distance(torch.from_numpy(original_points).unsqueeze(0), torch.from_numpy(pts).unsqueeze(0),
#                                             batch_reduction="sum",point_reduction='mean',norm_type="L2")
#                 cd += loss2
            
#             cd /= batch_size
#             mean_loss_noisy.append(cd.item())
            

#         for c_i in range(len(vote_offset)):    
#             vote_offset[c_i] = vote_offset_sum[c_i]/counts[c_i]
#             vote_clean[c_i] = vote_clean[c_i]/counts[c_i]
#             vote_loss[c_i] = vote_loss[c_i]/counts[c_i]
#             vote_loss_noisy_points[c_i] = vote_loss_noisy_points[c_i]/counts[c_i]

            
        
#     print(f"Mean loss for {config.experiment_name}: {np.mean(mean_loss)}")
#     m_l = np.mean(mean_loss)
#     m_l_n = np.mean(mean_loss_noisy)

#     return vote_offset, vote_clean, vote_loss, vote_loss_noisy_points,(m_l,m_l_n)

# def offset_prediction_on_k_patches(loader, model, config, criterion, k):
#     #! Deprecated
#     """
#     Predicting offsets given a model and the shapes, on a limited k patches.
#     """
#     vote_offset = [np.zeros((3, l.shape[0]), dtype=np.float32) for l in
#                    loader.dataset.clouds_points_labels]
#     print(f"Processing {len(vote_offset)} cloud/s")
#     model.eval()
#     counter = 0
#     densities = []
#     errors = []
#     outliernesses = []
#     PATH = './cloud_points/qualitative_patch_test_iteration'
#     PATH = os.path.join(PATH,config.experiment_name)
#     safe_make(PATH)
#     with torch.no_grad():
#         for idx, (points, mask, features,
#                    points_labels, points_offsets, current_cloud_index, input_inds, center_point) in enumerate(loader):
                   
#             points = points.cuda(non_blocking=True)
#             mask = mask.cuda(non_blocking=True)
#             features = features.cuda(non_blocking=True)
#             points_offsets = points_offsets.cuda(non_blocking=True)
#             center_point = center_point.cuda(non_blocking=True) 

#             # Forward
#             orginal_pred = model(points, mask, features)
#             # new_points = []
#             # new_points.append(points + torch.transpose(orginal_pred,1,2))
#             # for _ in range(3):
#             #     new_points.append(
#             #         new_points[-1] + torch.transpose(
#             #                             model(new_points[-1], mask, features)
#             #                             ,1,2))


            


            
#             # Ensembling all predictions
#             mask = mask.cpu().numpy().astype(bool)
#             pred = orginal_pred.cpu().numpy()
#             input_inds = input_inds.cpu().numpy()
#             center_point = center_point.cpu().numpy()
#             points_offsets = points_offsets.cpu().numpy()
#             # loss = loss.cpu().numpy()
#             points = points.cpu().numpy()
#             batch_size = points.shape[0]
#             for ib in range(batch_size):
#                 mask_i = mask[ib]
#                 points_batch = points[ib][mask_i,:]
#                 pred_batch = pred[ib][:, mask_i].T
#                 p_offsets_batch = points_offsets[ib][mask_i,:]
#                 ct_batch = center_point[ib]
#                 pt_labels = points_labels[ib]
            
#                 assert points_batch.shape == pred_batch.shape
#                 assert points_batch.shape == p_offsets_batch.shape

#                 denoised_points = points_batch + pred_batch
#                 clean_points = points_batch + p_offsets_batch

#                 # Distance from the center of a given gt point
#                 l2_loss = np.linalg.norm(clean_points - ct_batch)

#                 # Corresponding loss of a gt point when predicted by the model
#                 # Trying to see if farther away points have a higher loss than the ones near the center
#                 loss_batch = np.linalg.norm(denoised_points - clean_points,ord=1,axis=1)
#                 loss_batch /= config.in_radius

                
                
#                 index = counter*batch_size + ib
#                 #write_ply(filename,params_in_ls,params_names_ls)
#                 write_ply(f"{PATH}/original_points_{index}.ply",[clean_points],['vertex'])
#                 print(clean_points.shape)
#                 write_ply(f"{PATH}/pred_points_{index}.ply",[denoised_points,loss_batch],['vertex','loss'])

#                 # for index_iteration,denoised_points_iter in enumerate(new_points):
#                 #     # 
#                 #     new_denoised = denoised_points_iter.cpu().numpy()
#                 #     # print(new_denoised.shape)
#                 #     write_ply(f"{PATH}/pred_points_{index_iteration+1}.ply",[new_denoised[ib][mask_i,:]],['vertex'])

#                 write_ply(f"{PATH}/noisy_points_{index}.ply",[points_batch],['vertex'])

#                 #computing some quantities

#                 density = mask_i.sum()/mask_i.shape[0]
#                 outlierness = pt_labels.sum()/pt_labels.shape[0]
                
#                 loss = np.linalg.norm(pred_batch-p_offsets_batch,ord=1)

#                 densities.append(density)
#                 errors.append(loss)
#                 outliernesses.append(outlierness)

#                 break


#             break
#             counter += 1
#             if counter > k:
#                 break
        

#     #fixing dim
#     vote_offset = [v.T for v in vote_offset]
#     #saving for plot
#     with open('u_net_arch/figures/test_graph.npy', 'wb') as f:

#         np.save(f, densities)

#         np.save(f, errors)

#         np.save(f,outliernesses)
    

#     #####################3
#     return vote_offset