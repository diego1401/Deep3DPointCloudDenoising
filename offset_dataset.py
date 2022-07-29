from dis import dis
from platform import architecture
import torch
import torch.utils.data as data
import numpy as np
import os
import time
import sys
import pickle
from sklearn.neighbors import KDTree
from data_utils import grid_subsampling, read_ply_ls

import glob
import trimesh

from sklearn.model_selection import KFold

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

from data_utils import write_ply


from torch_scatter import scatter_sum

# FROM: https://github.com/tancik/fourier-feature-networks/blob/master/Demo.ipynb
# Fourier feature mapping
def input_mapping(x, B): 
  if B is None:
    return x
  else:
    x_proj = (2.*np.pi*x) @ B.T
    return np.concatenate([np.sin(x_proj), np.cos(x_proj)], axis=-1)

######################

def get_class_count_samples(class_id, sample_count_for_class, all_clouds_indices, all_clouds_labels):
    indices = [cloud_inds[lbl==class_id] for (cloud_inds,lbl) in zip(all_clouds_indices,
                                                                    all_clouds_labels)]
    return get_count_samples(indices,sample_count_for_class)

def get_count_samples(indices,sample_count):
    cloud_ID = [i*np.ones((indices[i].shape[0],)) for i in range(len(indices))]
    indices = np.concatenate(indices)
    cloud_ID = np.concatenate(cloud_ID)
    if indices.shape[0] > sample_count:
        shuffle_choice = np.random.permutation(np.arange(indices.shape[0]))
        indices = indices[shuffle_choice][:sample_count]
        cloud_ID = cloud_ID[shuffle_choice][:sample_count]
    else:
        shuffle_choice = np.random.permutation(np.arange(indices.shape[0]))
        indices = indices[shuffle_choice]
        cloud_ID = cloud_ID[shuffle_choice]
        padding_choice = np.random.choice(indices.shape[0], sample_count - indices.shape[0])
        indices = np.hstack([indices, indices[padding_choice]])
        cloud_ID = np.hstack([cloud_ID, cloud_ID[padding_choice]])

    return indices.astype(np.int32), cloud_ID.astype(np.int32)


def softmax(x,axis=0):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x,axis=axis,keepdims=True))
    return e_x / e_x.sum(axis=axis,keepdims=True)


from scipy.spatial import ConvexHull
def HPR_op(pos, pos_norm, pos_dir, parameter,ktype="std"):
    if ktype=="std":
        # Original R
        R = np.max(pos_norm)*10**parameter # biggest distance from scanning device
        pos_hat = pos+2*(R-pos_norm)*pos_dir
    elif ktype=="exp":
        # Exp ktz
        pos_hat = pos_dir*(pos_norm/np.max(pos_norm))**parameter

    pos_hat = np.concatenate([pos_hat,np.zeros((1,3))],axis=0)

    hull = ConvexHull(pos_hat)

    visible_indices = hull.vertices

    return visible_indices[:-1] # removing vertex corresponding to zero point

# Standard Katz visibility
def compute_katz(pos,parameters,ktype="std"):
    pos_norm = np.linalg.norm(pos,axis=1,ord=2)[:,None]
    pos_norm[pos_norm<1e-12]=1e-12
    pos_dir = pos/pos_norm

    k_ls = []
    for parameter in parameters:
        cur = np.ones((pos.shape[0],1)).astype(np.float32)
        indices = HPR_op(pos, pos_norm, pos_dir, parameter,ktype)
        cur[indices] = 0.
        k_ls.append(cur)

    katz = np.concatenate(k_ls,axis=1)
    return katz,pos_norm,pos_dir


def pc_normalize(pc):
    # Center and rescale point for 1m radius
    pmin = np.min(pc, axis=0)
    pmax = np.max(pc, axis=0)
    pc -= (pmin + pmax) / 2
    scale = np.max(np.linalg.norm(pc, axis=1))
    pc *= 1.0 / scale

    return pc


def get_scene_seg_features(input_features_dim, features):
    rem = abs(3 - input_features_dim%3)%3

    if rem>0:
        ones = torch.ones((features.shape[0],rem)).type(torch.float32)
        features = torch.cat([ones,features], -1)

    return features.transpose(0, 1).contiguous()

def file_of_files_to_list(file):
    out_ls = []
    Lines = file.readlines()
    # Strips the newline character
    for line in Lines:
        out_ls.append(line.strip())
    return out_ls


def add_noise_and_get_offset(shape, pos, gt, noise_type, noise_level):
    num_inliers = int(gt.shape[0]-np.sum(gt.squeeze()))
    
    if num_inliers > 0:
        # add noise to inliers
        offset_noise = np.zeros((num_inliers,3))
        if noise_level>0:
            if noise_type=="gaussian":
                offset_noise = noise_level*np.random.randn(num_inliers,3) #torch.randn(pos[gt==0].shape)
            elif noise_type=="white":
                offset_noise = noise_level*np.random.rand(num_inliers,3)

        mx = 3/100
        offset_noise = np.clip(offset_noise,-mx,mx)

        pos[gt==0] = pos[gt==0] + offset_noise

        
    
    closest_on_surface, d, _ = trimesh.proximity.closest_point(shape, pos)
    
    offset = torch.from_numpy(closest_on_surface) - pos # denoised = noisy + offset

    
    
    return pos, offset, d

def add_noise_and_get_offset_diverse(shape, pos, gt):
    num_inliers = int(gt.shape[0]-np.sum(gt.squeeze()))
    
    if num_inliers > 0:
        # add noise to inliers
        noise_levels = [0,0.25,0.5,1,1.5,2.5]
        num_of_instances = int(num_inliers//len(noise_levels))
        curr_instances = 0
        noises = []
        print(f"Total {num_inliers}, each bin has {num_of_instances}")
        for idx,noise_level in enumerate(noise_levels):
            noise_level /= 100
            if idx == len(noise_levels) -1 :
                num_of_instances = num_inliers - curr_instances

            curr_instances += num_of_instances

            if noise_level > 0:
                offset_noise = noise_level*np.random.randn(num_of_instances,3) 
            else:
                offset_noise = np.zeros((num_of_instances,3))

            noises.append(offset_noise)

        offset_noise_final = np.concatenate(noises)
        mx = 3/100
        offset_noise_final = np.clip(offset_noise_final,-mx,mx)
        np.random.shuffle(offset_noise_final)


        pos[gt==0] = pos[gt==0] + offset_noise_final
    
    closest_on_surface, d, _ = trimesh.proximity.closest_point(shape, pos)
    
    offset = torch.from_numpy(closest_on_surface) - pos # denoised = noisy + offset
    
    return pos, offset, d

def process_off_file(filepath, num_points_per_shape, outlier_proportion, noise_type, noise_level,margin=0.1):
    num_outliers_per_shape = int(num_points_per_shape*outlier_proportion)
    num_inliers_per_shape = num_points_per_shape - num_outliers_per_shape
    
    basename = os.path.basename(filepath)
    shape_name = basename.replace(".off","")
    print(f"Processing {shape_name}")
    shape = trimesh.load_mesh(filepath)
    
    shape.vertices -= np.mean(shape.vertices, axis = 0)
    diagonal = np.linalg.norm(np.max(shape.vertices, axis = 0) - np.min(shape.vertices, axis = 0), ord=2)
    shape.vertices = shape.vertices/diagonal
    
    print("Sampling")
    in_pos, faces = trimesh.sample.sample_surface_even(shape, num_inliers_per_shape)
    # normals = shape.face_normals[faces].astype(np.float32)
    
    
    in_mean = in_pos.mean(0)
    in_amplitude = in_pos.max() - in_pos.min()
    in_pos = in_pos - in_mean
    in_pos = (1.-margin)*in_pos / in_amplitude
    shape.vertices = shape.vertices - in_mean
    shape.vertices = (1.-margin)*shape.vertices / in_amplitude
    
    low_pos = in_pos.min(0) - margin
    high_pos = in_pos.max(0) + margin
    
    out_pos = np.random.uniform(size=(num_outliers_per_shape,3),low=low_pos,high=high_pos)
    
    pos = np.concatenate([in_pos,out_pos],axis=0)
    mean_pos = pos.mean(0)
    pos = pos - mean_pos
    shape.vertices = shape.vertices - mean_pos
    gt = np.concatenate([np.zeros((num_inliers_per_shape,1)),np.ones((num_outliers_per_shape,1))]).squeeze()
    
    order = np.arange(num_points_per_shape)
    print("Shuffle")
    np.random.shuffle(order)
    pos = pos[order]
    gt = gt[order].squeeze()
    
    tree = KDTree(pos,leaf_size=50)
    
    print("Adding Noise")
    if noise_type == 'diverse':
        print("Using the diverse setting")
        noisy_pos, current_points_offsets, d = add_noise_and_get_offset_diverse(shape, pos, gt)
        max_std = 2.5 / 100
    # elif noise_type == "diverse_stable":
    #     noise_levels = [0,0.25,0.5,1,1.5,2.5]
    #     noisy_pos = []
    #     current_points_offsets = []
    #     for noise_level in noise_levels:
    #         noisy, offsets, d = add_noise_and_get_offset(shape, pos, gt, noise_type, noise_level)
    #         noisy_pos.append(noise_level)
    #         current_points_offsets.append(offsets)
    #     noisy_pos = np.concatenate(noisy_pos)
    #     current_points_offsets = np.concatenate(current_points_offsets)

        gt = np.zeros(num_inliers_per_shape+num_outliers_per_shape)
        print(current_points_offsets.shape)
        gt[np.where(np.linalg.norm(current_points_offsets,axis=1)>max_std)] = 1

    else:
        noisy_pos, current_points_offsets, d = add_noise_and_get_offset(shape, pos, gt, noise_type, noise_level)
    print("Done")

    return shape, noisy_pos.astype(np.float32), gt.astype(np.int32), current_points_offsets, tree

class OffsetDataset(data.Dataset):
    def __init__(self, input_features, katz_params, katz_type, subsampling_parameter,
                 in_radius, num_points, num_steps, num_epochs,
                 feature_drop=0, data_root=None, transforms=None, split='train',dataset_type="PCN", noise_level=5.*1e-3, noise_type="gaussian", 
                 num_points_per_shape=140000, 
                 outlier_proportion=0.4, DEBUG=False,architecture='U-Net',sampleDl_patches=None,fourier_features=False):
        """ Dataset for Offset prediction

        Args:
            input_features_dim: input features dimensions, used to choose input feature type
            subsampling_parameter: grid length for pre-subsampling point clouds.
            in_radius: radius of each input spheres.
            num_points: max number of points for the input spheres.
            num_steps: number of spheres for one training epoch.
            num_epochs: total epochs.
            feature_drop: probability ratio for random feature dropping.
            data_root: root path for data.
            transforms: data transformations.
            split: dataset split name.
        """
        super().__init__()
        
        assert num_steps*num_epochs%2==0 # necessary because balanced number inliers - outliers --> num_steps * num_epochs has to be EVEN.

        self.noise_level = noise_level
        self.noise_type = noise_type
        self.num_points_per_shape = num_points_per_shape
        self.outlier_proportion = outlier_proportion
        self.architecture = architecture

        self.DEBUG = DEBUG
        mapping_size = 32
        self.fourier_features = fourier_features
        self.B = np.random.normal(0,12.,size=(mapping_size,3))

        self.epoch = 0
        self.input_features = input_features
        self.katz_type = katz_type

        self.katz_params = katz_params

        # input_features_dim = 0
        # for f in input_features:
        #     if f=="normal":
        #         input_features_dim += 3
        #     if "katz" in f:
        #         input_features_dim += len(self.katz_params)
        #     if f=="intensity":
        #         input_features_dim += 1
        
        # self.input_features_dim = input_features_dim
        self.input_features_dim = None

        self.transforms = transforms
        self.subsampling_parameter = subsampling_parameter
        self.feature_drop = feature_drop
        print("SAMPLE DL PATCHES VALUE",sampleDl_patches)
        self.in_radius = in_radius
        if sampleDl_patches is None:
            sampleDl_patches = in_radius
        self.num_points = num_points
        self.num_steps = num_steps
        self.num_epochs = num_epochs
        self.label_to_names = {0: 'inlier',
                               1: 'outlier'}
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}

        # assert dataset_type in ["PCN"]

        self.dataset_type = dataset_type

        self.data_root = data_root
        self.data_dir = os.path.join(self.data_root, 'processed')
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self.init_datasets()

        self.split = split

        if split == 'train':
            self.cloud_names = self.train_shapes
        elif split == 'val':
            self.cloud_names = self.val_shapes
        elif split == 'test':
            self.cloud_names = self.test_shapes
        elif split == 'qualitative_test':
            self.cloud_names = self.qualitative_test
        else:
            self.cloud_names = self.val_shapes + self.train_shapes

        
        # prepare data
        # filename = os.path.join(self.data_dir, f'{split}_{subsampling_parameter:.3f}_data.pkl')
        # if not os.path.exists(filename):
        cloud_points_list, cloud_points_cluster_list, cloud_points_features_list, cloud_points_label_list = [], [], [], []
        sub_cloud_points_list, sub_cloud_points_density_list, sub_cloud_points_label_list, sub_cloud_points_features_list = [], [], [], []
        sub_cloud_tree_list = []
        cloud_tree_list = []
        shape_list = []
        cloud_offsets_list = []
        sub_cloud_offsets_list = []

        self.index_to_cloud_name = {}

        ############################################################










        FORCE_REMAKE = False
        



        ##################################################################3
        if noise_type == "diverse_stable":
            noise_levels = [0,0.25,0.5,1,1.5,2.5]
        else:
            noise_levels = [self.noise_level]
        for noise_level in noise_levels:
            for cloud_idx, cloud_name in enumerate(self.cloud_names):
                print(f"{cloud_name}:{cloud_idx}")
                self.index_to_cloud_name[cloud_idx] = cloud_name
                if "EDF" in self.dataset_type:
                    raise ValueError("Can't use EDF data for this task.")
                
                elif self.dataset_type=="PCN":
                    # Pass if the cloud has already been computed
                    if noise_type == 'diverse':
                        print("Using diverse!")
                        cloud_file = os.path.join(self.data_dir, cloud_name + '_diverse_{:06d}_{:.2f}.pkl'.format(self.num_points_per_shape,self.outlier_proportion))
                    else:
                        cloud_file = os.path.join(self.data_dir, cloud_name + '_{}_{:.2e}_{:06d}_{:.2f}.pkl'.format(self.noise_type,noise_level,self.num_points_per_shape,self.outlier_proportion))
                    os.makedirs(os.path.dirname(cloud_file), exist_ok=True)
                    if os.path.exists(cloud_file) and not FORCE_REMAKE:
                        while os.stat(cloud_file).st_size == 0:
                            time.sleep(0.1)
                        with open(cloud_file, 'rb') as f:
                            shape, cloud_points, cloud_classes, cloud_offsets, cloud_tree = pickle.load(f)

                        
                    else:
                        shape, cloud_points, cloud_classes, cloud_offsets, cloud_tree = process_off_file(os.path.join(self.data_root, cloud_name + '.off'), self.num_points_per_shape, self.outlier_proportion, self.noise_type, noise_level)

                    
                        with open(cloud_file, 'wb') as f:
                            pickle.dump((shape, cloud_points, cloud_classes, cloud_offsets, cloud_tree), f)

                    cloud_intensity = None

                if self.split == "qualitative_test":

                    #To visualize cloud point and tree points
                    PATH = "cloud_points/cloud_points_for_viz"
                    write_ply(f"{PATH}/{cloud_name.split('/')[1]}_noise_level_{noise_level}.ply",[cloud_points,cloud_points[:,2]],["vertex","height"])
                    # write_ply(f"{PATH}/{cloud_name.split('/')[1]}_tree_inlier.ply",[np.array(cloud_tree.data)],["vertex"])
                    print(f"{cloud_name.split('/')[1]}_noise_level_{noise_level} written!")
                    ###
                if len(self.input_features)==0:
                    cloud_features = np.ones((cloud_points.shape[0], 3), dtype=np.float32)
                else:
                    all_ls = []
                    for f in self.input_features:
                        if f=="intensity":
                            all_ls.append(cloud_intensity)
                        elif f=="normal":
                            all_ls.append(cloud_normals)
                        elif "katz" in f:
                            all_ls.append(cloud_katz)

                    cloud_features = np.concatenate(all_ls,axis=1)

                cloud_points_list.append(cloud_points)
                cloud_points_features_list.append(cloud_features)
                cloud_points_label_list.append(cloud_classes)

                cloud_tree_list.append(cloud_tree)
                
                shape_list.append(shape)
                
                cloud_offsets_list.append(cloud_offsets)

                if subsampling_parameter > 0:
                    raise ValueError("Not implemented")
                    # sub_points, sub_features, sub_labels = grid_subsampling(cloud_points,
                    #                                                       features=cloud_features,
                    #                                                       labels=cloud_classes,
                    #                                                       sampleDl=subsampling_parameter)
                    # sub_labels = np.squeeze(sub_labels)
                    # search_tree = KDTree(sub_points, leaf_size=50)
                else:
                    sub_points = cloud_points
                    sub_features = cloud_features
                    sub_labels = np.squeeze(cloud_classes)
                    sub_offsets = cloud_offsets
                    search_tree = cloud_tree

                cluster_file = os.path.join(self.data_dir, "1NN", cloud_name + '.npy')
                os.makedirs(os.path.dirname(cluster_file), exist_ok=True)
                if os.path.exists(cluster_file):
                    cluster = np.load(cluster_file)
                else:
                    cluster = search_tree.query(cloud_points,k=1,return_distance=False)
                    with open(cluster_file,"wb") as f:
                        np.save(f,cluster)
                sub_density = scatter_sum(torch.ones(cluster.shape[0],),torch.from_numpy(cluster).squeeze())
                sub_density = (sub_density/torch.max(sub_density)).numpy()

                sub_cloud_points_density_list.append(sub_density)
                cloud_points_cluster_list.append(cluster)

                sub_cloud_points_list.append(sub_points)
                sub_cloud_points_features_list.append(sub_features)
                sub_cloud_points_label_list.append(sub_labels)
                sub_cloud_tree_list.append(search_tree)
                
                sub_cloud_offsets_list.append(sub_offsets)
            
        self.shapes = shape_list
        self.clouds_points_offsets = cloud_offsets_list

        self.clouds_points = cloud_points_list
        self.clouds_points_cluster = cloud_points_cluster_list
        self.clouds_points_features = cloud_points_features_list
        self.clouds_points_labels = cloud_points_label_list

        self.clouds_trees = cloud_tree_list

        self.sub_clouds_points = sub_cloud_points_list
        self.sub_clouds_points_density = sub_cloud_points_density_list
        self.sub_clouds_points_features = sub_cloud_points_features_list
        self.sub_clouds_points_labels = sub_cloud_points_label_list
        self.sub_cloud_trees = sub_cloud_tree_list
        
        self.sub_cloud_offsets = sub_cloud_offsets_list

        self.sub_clouds_indices = [np.arange(cloud.shape[0]) for cloud in self.sub_clouds_points]
        self.sub_clouds_points_density_proba = [softmax(density,axis=0) for density in self.sub_clouds_points_density]



        # contrary to e00, we different sampling of pick points, based on random sampling + class interleaving (only if train/val).
        print("!!!!! BALANCED POINT SAMPLING (!= MIN ENERGY SAMPLING) !!!!!")
        total_number_of_patches = self.num_epochs*self.num_steps
        # if "test" not in self.split or self.split == "qualitative_test":
        if "test" not in self.split:
            if self.outlier_proportion>0:
                num_outliers = int(total_number_of_patches/2.)
                num_inliers = total_number_of_patches - num_outliers

                outlier_inds, outlier_cloud_inds = get_class_count_samples(1, num_outliers, self.sub_clouds_indices, self.sub_clouds_points_labels)
                inlier_inds, inlier_cloud_inds = get_class_count_samples(0, num_inliers, self.sub_clouds_indices, self.sub_clouds_points_labels)

                self.cloud_inds = np.empty((outlier_cloud_inds.size + inlier_cloud_inds.size,), dtype=np.int32)
                self.cloud_inds[0::2] = outlier_cloud_inds
                self.cloud_inds[1::2] = inlier_cloud_inds

                self.point_inds = np.empty((outlier_inds.size + inlier_inds.size,), dtype=np.int32)
                self.point_inds[0::2] = outlier_inds
                self.point_inds[1::2] = inlier_inds
            else:
                inlier_inds, inlier_cloud_inds = get_class_count_samples(0, total_number_of_patches, self.sub_clouds_indices, self.sub_clouds_points_labels)

                self.cloud_inds = inlier_cloud_inds
                self.point_inds = inlier_inds
        else:
            # self.point_inds, self.cloud_inds = get_count_samples(self.sub_clouds_indices,total_number_of_patches)
            cloud_inds_ls = []
            point_inds_ls = []
            for i, tup in enumerate(zip(self.sub_clouds_points,self.sub_cloud_trees)):
                #sampleDl: distance between 2 consecutive subsampled points
                # sampleDl_for_subsampling = min(0.25*self.in_radius,0.25*2.)
                sampleDl_for_subsampling = sampleDl_patches
                pc = tup[0]
                tree = tup[1]
                sub_pc, _, _ = grid_subsampling(pc,features=pc, 
                                                labels=np.ones(pc.shape[0]).astype(np.int32), 
                                                sampleDl=sampleDl_for_subsampling) # the subsampling value of 0.25*2. is fixed for radii > 2. m (i.e. >40% of shape diameter). For values below, use 0.25*self.in_radius
                if architecture == "PCN":
                    cur_indices = np.arange(0,len(pc))
                elif "U-Net" in self.architecture:
                    cur_indices = tree.query(sub_pc,k=1,return_distance=False)
                cloud_inds_ls.append(i*np.ones_like(cur_indices))
                point_inds_ls.append(cur_indices)

            self.point_inds = np.concatenate(point_inds_ls)
            self.cloud_inds = np.concatenate(cloud_inds_ls)

            self.num_steps = self.point_inds.shape[0]

        # if "test" not in self.split or self.split == "qualitative_test":
        if "test" not in self.split:
            self.noise = np.random.normal(scale=2*self.subsampling_parameter, 
                                          size=(self.point_inds.shape[0],3))# scale=self.in_radius / 10
        else:
            self.noise = np.zeros((self.point_inds.shape[0],3))

        self.cloud_inds = np.split(self.cloud_inds,self.cloud_inds.shape[0],axis=0)
        self.point_inds = np.split(self.point_inds,self.point_inds.shape[0],axis=0)
        self.noise = np.split(self.noise,self.noise.shape[0],axis=0)

    def init_datasets(self):
        self.train_shapes = ["train/"+os.path.basename(f).replace(".off","") for f in glob.glob(os.path.join(self.data_root,"train/")+"*.off")]
        self.val_shapes = ["val/"+os.path.basename(f).replace(".off","") for f in glob.glob(os.path.join(self.data_root,"val/")+"*.off")]
        self.test_shapes = ["test/"+os.path.basename(f).replace(".off","") for f in glob.glob(os.path.join(self.data_root,"test/")+"*.off")]
        self.qualitative_test = ["qualitative_test/"+os.path.basename(f).replace(".off","") for f in glob.glob(os.path.join(self.data_root,"qualitative_test/")+"*.off")]
        

        for split in ["train","test","val","qualitative_test"]:
            if split=="train":
                files = self.train_shapes
            elif split=="val":
                files = self.val_shapes
            elif split=="test":
                files = self.test_shapes
            elif split == "qualitative_test":
                files = self.qualitative_test


        if self.DEBUG:
            self.train_shapes = [self.train_shapes[0],self.train_shapes[1]]
            self.val_shapes = [self.val_shapes[0],self.val_shapes[1]]
            self.test_shapes = [self.test_shapes[0],self.test_shapes[1]]
            # self.qualitative_test stays the same since is user defined

    def __getitem__(self, idx):
        """
        Returns:
            current_points: (N, 3), a point cloud.
            mask: (N, ), 0/1 mask to distinguish padding points.
            features: (input_features_dim, N), input points features.
            current_points_labels: (N), point label.
            current_cloud_index: (1), cloud index.
            input_inds: (N), the index of input points in point cloud.
        """
        index = idx + self.epoch * self.num_steps

        cloud_ind = int(self.cloud_inds[index])
        point_ind = int(self.point_inds[index])
        noise = self.noise[index]


        if self.dataset_type=="PCN": # full
            cur_cloud_tree = self.clouds_trees[cloud_ind]
            cur_clouds_points_features = self.clouds_points_features[cloud_ind]
            cur_clouds_points_labels = self.clouds_points_labels[cloud_ind]
        else: # subsampled
            raise ValueError("Can't use EDF data for offset prediction.")
            
        cur_clouds_points_offsets = self.clouds_points_offsets[cloud_ind]

        points = np.array(cur_cloud_tree.data, copy=False)
        center_point = points[point_ind, :].reshape(1,3)#.reshape(1, -1)
        pick_point = center_point + noise.astype(center_point.dtype)


        # Indices of points in input region
        query_tmps = cur_cloud_tree.query_radius(pick_point,r=self.in_radius,
                                                            return_distance=True,
                                                            sort_results=True)
        query_inds = query_tmps[0][0]
     
        # Number collected
        cur_num_points = query_inds.shape[0]
        if point_ind not in query_inds:
            print("FAULTY BEFORE")
        
        if cur_num_points == 0:
            query_tmps = cur_cloud_tree.query_radius(pick_point,r=self.in_radius*2,
                                                            return_distance=True,
                                                            sort_results=True)
            distances = query_tmps[1][0]
            # print("distances",type(distances))
            # idx = np.where(distances<self.in_radius)
            query_inds = query_tmps[0][0]
            cur_num_points = query_inds.shape[0]
            assert cur_num_points > 0

            print(f"Index of faulty cloud {cloud_ind}, real points {cur_num_points}")
            
            # self.__getitem__(idx+1)
        if self.num_points < cur_num_points:
            shuffle_choice = np.random.permutation(np.arange(self.num_points))
            input_inds = query_inds[:self.num_points][shuffle_choice]
            # center_point_ind = np.argmin(shuffle_choice) # where 0 is
            mask = torch.ones(self.num_points,).type(torch.int32)
        else:
            shuffle_choice = np.random.permutation(np.arange(cur_num_points))
            if "U-Net" in self.architecture:
                query_inds = query_inds[shuffle_choice]
            # center_point_ind = np.argmin(shuffle_choice) # where 0 is
                padding_choice = np.random.choice(cur_num_points, self.num_points - cur_num_points)
                input_inds = np.hstack([query_inds, query_inds[padding_choice]])
            elif self.architecture == "PCN":
                padding_choice = np.zeros(self.num_points - cur_num_points).astype(int)
                input_inds = np.hstack([query_inds, padding_choice])
            
            
            mask = torch.zeros(self.num_points,).type(torch.int32)
            mask[:cur_num_points] = 1

        

            
        if point_ind not in input_inds:
            center_point_ind = np.where(input_inds==query_inds[0])[0][0]
            print("FAULTY")
        else:
            center_point_ind = np.where(input_inds==point_ind)[0][0]
            #MAKE SURE THE FIRST ELEMENT IS THE CENTER
        input_inds[0],input_inds[center_point_ind] = input_inds[center_point_ind],input_inds[0]
        # assert np.where(input_inds==point_ind)[0][0] == 0
        center_point_ind = 0

        # assert point_ind == input_inds[center_point_ind]



        original_points = points[input_inds]
        
        current_points_offsets = cur_clouds_points_offsets[input_inds]
        
        current_points = original_points - pick_point

        # current_features = cur_clouds_points_features[input_inds]
        # current_features = torch.from_numpy(current_features).type(torch.float32)

        # current_features_drop = (torch.rand(1) > self.feature_drop).type(torch.float32)
        # current_features = (current_features * current_features_drop).type(torch.float32)
        current_cloud_index = torch.from_numpy(np.array(cloud_ind)).type(torch.int64)

        
        
        # stacking just to make sure that the offsets are also transformed according to data augmentation
        point_stack = np.concatenate([current_points,current_points_offsets],axis=0)
        if self.transforms is not None:
            point_stack = self.transforms(point_stack)
        current_points = point_stack[0:original_points.shape[0],:]
        current_points_offsets = point_stack[original_points.shape[0]:,:]


        if self.architecture == "PCN":
            if "test" in self.split:
                return [current_points, center_point_ind, current_points_offsets[center_point_ind,:], current_cloud_index, input_inds]
            else:
                return [current_points, center_point_ind, current_points_offsets, current_cloud_index, input_inds]
            
        else:
            current_points_labels = torch.from_numpy(cur_clouds_points_labels[input_inds].squeeze()).contiguous().type(torch.int64)
            
            
            if not self.fourier_features:
                # OPTION 1: Trivial, use points as features
                features = current_points.clone().transpose(1,0)
            else:
                #OPTION 2: From https://github.com/tancik/fourier-feature-networks/blob/master/Demo.ipynb
                features = input_mapping(current_points.clone(),self.B).transpose(1,0)


            return [current_points, mask, features,
                        current_points_labels, current_points_offsets, current_cloud_index, input_inds]

    def __len__(self):
        return self.num_steps

if __name__=="__main__":
    root = "path/to/PCN_SHAPES/"
    diam_perc = 10
    transforms = None

    # For a first trial, use setting 1.
    # Setting 1: num_points_per_shape=50000, noise_type="white", noise_level=5.*1e-2,outlier_proportion=0.4
    # Setting 2: num_points_per_shape=50000, noise_type="gaussian", noise_level=5.*1e-4,outlier_proportion=0.4
    #
    # Example:
    # --------
    # dset = OffsetDataset(data_root=root,input_features=[],katz_params=[],katz_type='std',in_radius=0.5*diam_perc*10./100., subsampling_parameter=0., num_points=1000,num_steps=6, num_epochs=5, split='train', transforms=transforms,dataset_type="PCN",DEBUG=False, num_points_per_shape=50000, noise_type="white", noise_level=5.*1e-2,outlier_proportion=0.4)
    
    
    # "debug" setup (DEBUG flag on True)
    dset = OffsetDataset(data_root=root,input_features=[],katz_params=[],katz_type='std',in_radius=0.5*diam_perc*10./100., subsampling_parameter=0., num_points=1000,num_steps=6, num_epochs=5, split='train', transforms=transforms,dataset_type="PCN",DEBUG=True, num_points_per_shape=50000, noise_type="gaussian", noise_level=5.*1e-4,outlier_proportion=0.4)


    [current_points, mask, features,
                           current_points_labels, current_points_offsets, current_cloud_index, input_inds] = dset[1]

    write_ply("test.ply",[current_points,mask,current_points_labels,np.linalg.norm(current_points_offsets,axis=-1)],["vertex","m","GT","d"])