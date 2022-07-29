
from importlib.resources import path
from turtle import distance
from models import chamfer_distance
from data_utils import read_ply_ls
import os
import torch
import numpy as np

def get_cloud(cloud_name):
    cloud = read_ply_ls(cloud_name,["vertex"])["vertex"]
    return torch.from_numpy(cloud).unsqueeze(0).cuda()

def main():
    torch.cuda.set_device(1)

    path = 'cloud_points/denoised_clouds'
    f = open("Performance.txt","w")
    
    model_names = ["l1","full_cleaning_diverse_double_weight_test_5e3",
                   "full_cleaning_diverse_vanilla_test_5e3",
                   "full_cleaning_diverse_weighted_offset_test_5e3",
                   "full_cleaning_diverse_double_weight_fourier_test_5e3",
                   "full_cleaning_diverse_vanilla_fourier_test_5e3",
                   "full_cleaning_diverse_weighted_offset_fourier_test_5e3"]

    model_performances = []
    orginal_noise = None
    for model in model_names:
        
        path_model = os.path.join(path,model)
        path_clean_clouds = os.path.join(path_model,"clean")
        path_denoised_clouds = os.path.join(path_model,"denoised")
        path_noisy_clouds = os.path.join(path_model,"noisy")
        cloud_names = ["_".join(name.split("_")[:-1]) for name in os.listdir(path_clean_clouds)]
        
        print(f"Processing model {model}")

        measures = []
        
        tmp = []
        for cloud_name in cloud_names:
            
            clean_cloud = get_cloud(os.path.join(path_clean_clouds,cloud_name + "_clean.ply"))
            denoised_cloud = get_cloud(os.path.join(path_denoised_clouds,cloud_name + "_denoised.ply"))
            noisy_cloud = get_cloud(os.path.join(path_noisy_clouds,cloud_name + "_noisy.ply"))
            
            try:
                cd_noisy,_, distances_noisy = chamfer_distance(clean_cloud,noisy_cloud,batch_reduction="mean",
                                            point_reduction="mean",norm_type="L2",return_distances=True)
                cd_denoised,_, distances_denoised = chamfer_distance(clean_cloud,denoised_cloud,batch_reduction="mean",
                                            point_reduction="mean",norm_type="L2", return_distances=True)

                # Noise measure
                tmp += [cd_noisy.item()]
                noise_ratio = cd_denoised.item()
                # Outlier measure
                outlier_count_noisy = (torch.sqrt(distances_noisy)>0.05).sum().float()
                outlier_count_denoised =  (torch.sqrt(distances_denoised)>0.05).sum().float()
                
                print(f"{outlier_count_noisy}")

                if outlier_count_noisy.item():
                    outlier_ratio = outlier_count_denoised.item()/outlier_count_noisy.item()
                else:
                    outlier_ratio = 0

                measures.append([noise_ratio,outlier_ratio])
            except:
                print(f"{model}:{cloud_name}")
        
        if model == "l1":
            orginal_noise = np.mean(tmp)
        noise_ratio_global,outlier_ratio_global = np.mean(measures,axis=0)
        model_performances.append((model,noise_ratio_global/orginal_noise,outlier_ratio_global))
        


    for model,noise,outlier in model_performances:
        f.write(f"{model}: noise ratio {round(noise,2)} | outlier ratio {'{:.2E}'.format(outlier)}\n")
    f.close()
    



if __name__ == "__main__":
    main()