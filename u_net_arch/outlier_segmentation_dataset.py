import torch
import torch.utils.data as data
import numpy as np
import os
import time
import sys
import pickle
from sklearn.neighbors import KDTree
from data_utils import grid_subsampling, read_ply_ls

from sklearn.model_selection import KFold

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

from data_utils import write_ply


from torch_scatter import scatter_sum

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


class OutlierSegmentationDataset(data.Dataset):
    def __init__(self, input_features, katz_params, katz_type, subsampling_parameter,
                 in_radius, num_points, num_steps, num_epochs,
                 feature_drop=0, data_root=None, transforms=None, split='train',dataset_type="EDFS", DEBUG=False):
        """EDF dataset for scene segmentation task.

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

        self.DEBUG = DEBUG

        self.epoch = 0
        self.input_features = input_features
        self.katz_type = katz_type

        self.katz_params = katz_params

        input_features_dim = 0
        for f in input_features:
            if f=="normal":
                input_features_dim += 3
            if "katz" in f:
                input_features_dim += len(self.katz_params)
            if f=="intensity":
                input_features_dim += 1
        self.input_features_dim = input_features_dim
        self.transforms = transforms
        self.subsampling_parameter = subsampling_parameter
        self.feature_drop = feature_drop
        self.in_radius = in_radius
        self.num_points = num_points
        self.num_steps = num_steps
        self.num_epochs = num_epochs
        self.label_to_names = {0: 'inlier',
                               1: 'outlier'}
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}

        assert dataset_type in ["EDFM","EDFS","EDFS3f0","EDFS3f1","EDFS3f2","PCN"]

        self.dataset_type = dataset_type

        self.data_root = data_root
        self.data_dir = os.path.join(self.data_root, 'processed')
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self.init_datasets()

        self.split = split

        if split == 'train':
            self.cloud_names = self.train_clouds
        elif split == 'val':
            self.cloud_names = self.val_clouds
        elif split == 'test':
            self.cloud_names = self.test_clouds
        else:
            self.cloud_names = self.val_clouds + self.train_clouds

        # prepare data
        # filename = os.path.join(self.data_dir, f'{split}_{subsampling_parameter:.3f}_data.pkl')
        # if not os.path.exists(filename):
        cloud_points_list, cloud_points_cluster_list, cloud_points_features_list, cloud_points_label_list = [], [], [], []
        sub_cloud_points_list, sub_cloud_points_density_list, sub_cloud_points_label_list, sub_cloud_points_features_list = [], [], [], []
        sub_cloud_tree_list = []
        cloud_tree_list = []

        for cloud_idx, cloud_name in enumerate(self.cloud_names):

            if "EDF" in self.dataset_type:
                # Pass if the cloud has already been computed
                cloud_file = os.path.join(self.data_dir, cloud_name + '.pkl')
                if os.path.exists(cloud_file):
                    while os.stat(cloud_file).st_size == 0:
                        time.sleep(0.1)
                    with open(cloud_file, 'rb') as f:
                        cloud_points, cloud_intensity, cloud_normals, cloud_classes, cloud_tree = pickle.load(f)
                else:
                    try:
                        ply = read_ply_ls(os.path.join(self.data_root, cloud_name + '.ply'),["vertex","GT","normal","intensity"])
                    except Exception as e:
                        ply = read_ply_ls(os.path.join(self.data_root, cloud_name + '.ply'),["vertex","GT","nx","ny","nz","intensity"])
                        ply["normal"] = np.concatenate([ply["nx"],ply["ny"],ply["nz"]],axis=1)
                    cloud_points = ply["vertex"]
                    cloud_intensity = (ply["intensity"]/255.).astype(np.float32)
                    cloud_normals = ply["normal"]
                    cloud_classes = (ply["GT"]==2).astype(np.int32)

                    cloud_tree = KDTree(cloud_points, leaf_size=50)

                    with open(cloud_file, 'wb') as f:
                        pickle.dump((cloud_points, cloud_intensity, cloud_normals, cloud_classes, cloud_tree), f)


                if len(self.katz_params)>0:
                    katz_features_ls = []
                    for cur_val in self.katz_params:
                        katz_file = os.path.join(self.data_dir, "katz_values", "{}Ktz{:.3f}_".format(self.katz_type,cur_val) + cloud_name + '.ply.npy')

                        if os.path.exists(katz_file):
                            katz = np.load(katz_file)
                        else:
                            katz,_,_ = compute_katz(cloud_points,[cur_val],self.katz_type)
                            with open(katz_file,"wb") as f:
                                np.save(f,katz)
                        katz_features_ls.append(katz)

                    cloud_katz = np.concatenate(katz_features_ls,axis=1)

            elif self.dataset_type=="PCN":
                # Pass if the cloud has already been computed
                cloud_file = os.path.join(self.data_dir, cloud_name + '.pkl')
                os.makedirs(os.path.dirname(cloud_file), exist_ok=True)
                if os.path.exists(cloud_file):
                    while os.stat(cloud_file).st_size == 0:
                        time.sleep(0.1)
                    with open(cloud_file, 'rb') as f:
                        cloud_points, cloud_normals, cloud_classes, cloud_tree = pickle.load(f)
                else:
                    ply = read_ply_ls(os.path.join(self.data_root, cloud_name + '.ply'),["vertex","normal","GT"])
                    cloud_points = ply["vertex"]
                    cloud_normals = ply["normal"]
                    cloud_classes = (ply["GT"]==1).astype(np.int32)

                    cloud_points = cloud_points[cloud_classes==0] # clean points

                    cloud_tree = KDTree(cloud_points, leaf_size=50)

                    

                    with open(cloud_file, 'wb') as f:
                        pickle.dump((cloud_points, cloud_normals, cloud_classes, cloud_tree), f)

                cloud_intensity = None

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

            if subsampling_parameter > 0:
                sub_points, sub_features, sub_labels = grid_subsampling(cloud_points,
                                                                      features=cloud_features,
                                                                      labels=cloud_classes,
                                                                      sampleDl=subsampling_parameter)
                sub_labels = np.squeeze(sub_labels)
            else:
                sub_points = cloud_points
                sub_features = cloud_features
                sub_labels = np.squeeze(cloud_classes)

            # Get chosen neighborhoods
            search_tree = KDTree(sub_points, leaf_size=50)

            cluster_file = os.path.join(self.data_dir, "1NN", cloud_name + '.ply.npy')
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

        self.sub_clouds_indices = [np.arange(cloud.shape[0]) for cloud in self.sub_clouds_points]
        self.sub_clouds_points_density_proba = [softmax(density,axis=0) for density in self.sub_clouds_points_density]



        # contrary to e00, we different sampling of pick points, based on random sampling + class interleaving (only if train/val).
        print("!!!!! BALANCED POINT SAMPLING (!= MIN ENERGY SAMPLING) !!!!!")
        total_number_of_patches = self.num_epochs*self.num_steps
        if "test" not in self.split:
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
            # self.point_inds, self.cloud_inds = get_count_samples(self.sub_clouds_indices,total_number_of_patches)
            cloud_inds_ls = []
            point_inds_ls = []
            for i, tup in enumerate(zip(self.sub_clouds_points,self.sub_cloud_trees)):
                pc = tup[0]
                tree = tup[1]
                sub_pc, _, _ = grid_subsampling(pc,features=pc, labels=np.ones(pc.shape[0]).astype(np.int32), sampleDl=min(0.25*self.in_radius,0.25*2.)) # the subsampling value of 0.25*2. is fixed for radii > 2. m (i.e. >40% of shape diameter). For values below, use 0.25*self.in_radius
                cur_indices = tree.query(sub_pc,k=1,return_distance=False)
                cloud_inds_ls.append(i*np.ones_like(cur_indices))
                point_inds_ls.append(cur_indices)

            self.point_inds = np.concatenate(point_inds_ls)
            self.cloud_inds = np.concatenate(cloud_inds_ls)

            self.num_steps = self.point_inds.shape[0]

        if "test" not in self.split:
            self.noise = np.random.normal(scale=2*self.subsampling_parameter, size=(self.point_inds.shape[0],3))# scale=self.in_radius / 10
        else:
            self.noise = np.zeros((self.point_inds.shape[0],3))

        self.cloud_inds = np.split(self.cloud_inds,self.cloud_inds.shape[0],axis=0)
        self.point_inds = np.split(self.point_inds,self.point_inds.shape[0],axis=0)
        self.noise = np.split(self.noise,self.noise.shape[0],axis=0)


        # prepare validation projection inds
        filename = os.path.join(self.data_dir, '{}_{}_{}_{:.2f}_{:.2f}_proj.pkl'.format(self.DEBUG,self.dataset_type,self.split,
                                                                                        self.subsampling_parameter,self.in_radius))
        if not os.path.exists(filename):
            print("Computing projections...")
            proj_ind_list = []
            if self.dataset_type=="PCN":
                for points, search_tree in zip(self.clouds_points, self.clouds_trees):
                    proj_inds = np.arange(0,points.shape[0]).astype(np.int32)
                    proj_ind_list.append(proj_inds)
            else:
                for points, search_tree in zip(self.clouds_points, self.sub_cloud_trees):
                    proj_inds = np.squeeze(search_tree.query(points, return_distance=False))
                    proj_inds = proj_inds.astype(np.int32)
                    proj_ind_list.append(proj_inds)
            self.projections = proj_ind_list
            print("Done.")
            with open(filename, 'wb') as f:
                pickle.dump(self.projections, f)
                print(f"{filename} saved successfully")
        else:
            with open(filename, 'rb') as f:
                self.projections = pickle.load(f)
                print(f"{filename} loaded successfully")

    def init_datasets(self):
        small_dataset = ["pointcloud_00", "pointcloud_01", "pointcloud_02", "pointcloud_03", "pointcloud_04", "pointcloud_05", "pointcloud_06", "pointcloud_07", "pointcloud_08", "pointcloud_09", "pointcloud_10", "pointcloud_11", "pointcloud_12", "pointcloud_13"]

        val_dataset_extension = ["test_pointcloud_13", "test_pointcloud_14", "test_pointcloud_15", "test_pointcloud_16", "test_pointcloud_17", "test_pointcloud_18", "test_pointcloud_19", "test_pointcloud_20"]

        if self.dataset_type=="EDFM":
            self.train_clouds = small_dataset[0:11]
            self.val_clouds = small_dataset[11:]
            self.test_clouds = val_dataset_extension
        elif self.dataset_type=="EDFS": #EDF small
            self.train_clouds = small_dataset[0:9]
            self.val_clouds = small_dataset[9:11]
            self.test_clouds = small_dataset[11:]
        elif "EDFS" in self.dataset_type:
            num_folds,fold_id = self.dataset_type.split("EDFS")[-1].split("f")
            num_folds = int(num_folds)
            fold_id = int(fold_id)

            kf = KFold(n_splits=num_folds)
            # removing 9091
            splits = kf.split(small_dataset[0:-1])

            fold_ls = [(train,test) for train,test in splits]
            train_index,test_index = fold_ls[fold_id]
            print("************")
            print("************")
            print("K FOLD VALIDATION WITH K={}, USING FOLD ID {}.".format(num_folds, fold_id))
            print("TRAIN_INDEX = {}".format(train_index[2:]))
            print("additionally, we will add the index for the outlier (i.e. difficult) scan.")
            print("VAL_INDEX = {}".format(train_index[0:2]))
            print("TEST_INDEX = {}".format(test_index))
            print("************")
            print("************")
            self.train_clouds = [small_dataset[i] for i in train_index[2:]]+[small_dataset[-1]]
            self.val_clouds = [small_dataset[i] for i in train_index[0:2]]
            self.test_clouds = [small_dataset[i] for i in test_index]
        elif self.dataset_type=="PCN":
            with open(os.path.join(self.data_root,"outliers_TRAIN_W_NOR",'trainingset.txt'), 'r') as f:
                self.train_clouds = [os.path.join("outliers_TRAIN_W_NOR",l) for l in file_of_files_to_list(f)]

            with open(os.path.join(self.data_root,"outliers_TRAIN_W_NOR",'validationset.txt'), 'r') as f:
                self.val_clouds = [os.path.join("outliers_TRAIN_W_NOR",l) for l in file_of_files_to_list(f)]

            with open(os.path.join(self.data_root,"outliers_TEST_W_NOR",'testset.txt'), 'r') as f:
                self.test_clouds = [os.path.join("outliers_TEST_W_NOR",l) for l in file_of_files_to_list(f)]


        for split in ["train","test","val"]:
            if split=="train":
                files = self.train_clouds
            elif split=="val":
                files = self.val_clouds
            elif split=="test":
                files = self.test_clouds
            num_classes_file = os.path.join(self.data_dir, '_num_inliers_num_outliers_{}.pkl'.format(split))


            if os.path.exists(num_classes_file):
                while os.stat(num_classes_file).st_size == 0:
                    time.sleep(0.1)
                with open(num_classes_file, 'rb') as f:
                    num_inliers,num_outliers = pickle.load(f)
                print("[LOADED] {:012d} INLIERS AND {:012d} OUTLIERS IN SPLIT {}".format(num_inliers,num_outliers,split))
            else:
                num_inliers = 0
                num_outliers = 0
                for file in files:
                    ply = read_ply_ls(os.path.join(self.data_root, file + '.ply'),["GT"])
                    gt = ply["GT"].squeeze()
                    num_inliers += gt[gt==0].shape[0]
                    num_outliers += gt[gt==1].shape[0]

                print("[COMPUTED] {:012d} INLIERS AND {:012d} OUTLIERS IN SPLIT {}".format(num_inliers,num_outliers,split))
                with open(num_classes_file, 'wb') as f:
                    pickle.dump((num_inliers,num_outliers), f)



        if self.DEBUG:
            self.train_clouds = [self.train_clouds[0],self.train_clouds[1]]
            self.val_clouds = [self.val_clouds[0],self.val_clouds[1]]
            self.test_clouds = [self.test_clouds[0],self.test_clouds[1]]

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
        cloud_ind = int(self.cloud_inds[idx + self.epoch * self.num_steps])
        point_ind = int(self.point_inds[idx + self.epoch * self.num_steps])
        noise = self.noise[idx + self.epoch * self.num_steps]


        if self.dataset_type=="PCN": # full
            cur_cloud_tree = self.clouds_trees[cloud_ind]
            cur_clouds_points_features = self.clouds_points_features[cloud_ind]
            cur_clouds_points_labels = self.clouds_points_labels[cloud_ind]
        else: # subsampled
            cur_cloud_tree = self.sub_cloud_trees[cloud_ind]
            cur_clouds_points_features = self.sub_clouds_points_features[cloud_ind]
            cur_clouds_points_labels = self.sub_clouds_points_labels[cloud_ind]

        points = np.array(cur_cloud_tree.data, copy=False)
        center_point = points[point_ind, :].reshape(1,3)#.reshape(1, -1)
        pick_point = center_point + noise.astype(center_point.dtype)


        # Indices of points in input region
        query_inds = cur_cloud_tree.query_radius(pick_point,r=self.in_radius,
                                                            return_distance=True,
                                                            sort_results=True)[0][0]


        # v = points[query_inds]
        # gt = self.clouds_points_labels[cloud_ind][query_inds]
        # write_ply("test_{:.2f}.ply".format(self.in_radius),[v,gt],["vertex","GT"])

        # Number collected
        cur_num_points = query_inds.shape[0]
        if self.num_points < cur_num_points:
            # choice = np.random.choice(cur_num_points, self.num_points)
            # input_inds = query_inds[choice]
            shuffle_choice = np.random.permutation(np.arange(self.num_points))
            input_inds = query_inds[:self.num_points][shuffle_choice]
            mask = torch.ones(self.num_points,).type(torch.int32)
        else:
            shuffle_choice = np.random.permutation(np.arange(cur_num_points))
            query_inds = query_inds[shuffle_choice]
            padding_choice = np.random.choice(cur_num_points, self.num_points - cur_num_points)
            input_inds = np.hstack([query_inds, query_inds[padding_choice]])
            mask = torch.zeros(self.num_points,).type(torch.int32)
            mask[:cur_num_points] = 1

        # v = points[input_inds]
        # gt = self.clouds_points_labels[cloud_ind][input_inds]
        # write_ply("test_{:.2f}.ply".format(self.in_radius),[v,gt],["vertex","GT"])

        original_points = points[input_inds]
        current_points = (original_points - pick_point)
        # current_points_height = original_points[:, 2:]
        # current_points_height = torch.from_numpy(current_points_height).type(torch.float32)

        current_features = cur_clouds_points_features[input_inds]
        #current_features = (current_features - self.color_mean) / self.color_std
        current_features = torch.from_numpy(current_features).type(torch.float32)

        current_features_drop = (torch.rand(1) > self.feature_drop).type(torch.float32)
        current_features = (current_features * current_features_drop).type(torch.float32)


        current_points_labels = torch.from_numpy(cur_clouds_points_labels[input_inds].squeeze()).contiguous().type(torch.int64)
        current_cloud_index = torch.from_numpy(np.array(cloud_ind)).type(torch.int64)

        if self.transforms is not None:
            current_points = self.transforms(current_points)

        # features = get_scene_seg_features(self.input_features_dim, current_points, current_features,
        #                                   current_points_height)

        features = get_scene_seg_features(self.input_features_dim, current_features)

        # if self.DEBUG:
        #     write_ply("test_E{:03d}_{:02d}.ply".format(self.epoch,cloud_ind),[current_points,mask,current_points_labels],["vertex","mask","GT"])

        return [current_points, mask, features,
                       current_points_labels, current_cloud_index, input_inds]

    def __len__(self):
        return self.num_steps

if __name__=="__main__":
    for diam_perc in [10.,20.,40.]:
        dset = OutlierSegmentationDataset(data_root=edf_root,input_features=[], katz_params=[], katz_type='std', in_radius=0.5*diam_perc*10./100., subsampling_parameter=0.04, num_points=15000,
                    num_steps=7, num_epochs=5, split='train', transforms=transforms,dataset_type="EDFS3f0",DEBUG=False)
