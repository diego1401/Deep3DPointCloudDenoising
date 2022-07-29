
from importlib.resources import path
from models import chamfer_distance
from data_utils import read_ply_ls
import os
import torch
import numpy as np

def read_xyz(filename):
    with open(filename,"r") as f:
        lines = f.readlines()
        array = []
        for line in lines:
            x,y,z = line.strip().split(" ")
            x,y,z = float(x),float(y),float(z)
            array += [[x,y,z]]
    return np.array(array,dtype=np.float32)


def main():
    torch.cuda.set_device(1)

    path = 'cloud_points/denoised_clouds'
    f = open("CD_table.txt","w")
    # model_names = list(os.listdir(path))
    model_names = ["l1_diverse_0.1","l1_diverse_0.5","l1_diverse_0.25","l1_diverse_1","l1_diverse_2","l1_diverse_test_1e3"]
    std_1e3 = []
    std_5e3 = []
    # model_1e3_error_map = {}
    # model_5e3_error_map = {}
    cloud_errors = {}
    for model in model_names:
        path_model = os.path.join(path,model)
        path_clean_clouds = os.path.join(path_model,"clean")
        path_denoised_clouds = os.path.join(path_model,"denoised")
        path_noisy_clouds = os.path.join(path_model,"noisy")
        
        mean_noisy = []
        mean_denoised = []
        
        if model in ["PCN","GAN_try1","GAN_try2"]:
            continue
        print(f"Processing model {model}")

        cloud_names = ["_".join(name.split("_")[:-1]) for name in os.listdir(path_clean_clouds)]
        for cloud_name in cloud_names:
            clean_cloud_name = os.path.join(path_clean_clouds,cloud_name + "_clean.ply")
            if model == "PCN":
                # denoised_cloud_name = os.path.join(path_denoised_clouds,cloud_name[17:] + ".xyz")
                # denoised_cloud = read_xyz(denoised_cloud_name)
                pass

            else:
                denoised_cloud_name = os.path.join(path_denoised_clouds,cloud_name + "_denoised.ply")
            noisy_cloud_name = os.path.join(path_noisy_clouds,cloud_name + "_noisy.ply")


           
            # try:
            #     clean_cloud = read_ply_ls(clean_cloud_name,["vertex","loss"])["vertex"]
            #     denoised_cloud = read_ply_ls(denoised_cloud_name,["vertex","loss"])["vertex"]
            #     noisy_cloud = read_ply_ls(noisy_cloud_name,["vertex","loss"])["vertex"]
            # except:
            clean_cloud = read_ply_ls(clean_cloud_name,["vertex"])["vertex"]
            denoised_cloud = read_ply_ls(denoised_cloud_name,["vertex"])["vertex"]
            noisy_cloud = read_ply_ls(noisy_cloud_name,["vertex"])["vertex"]

            

            clean_cloud = torch.from_numpy(clean_cloud).unsqueeze(0).cuda()
            denoised_cloud = torch.from_numpy(denoised_cloud).unsqueeze(0).cuda()
            noisy_cloud = torch.from_numpy(noisy_cloud).unsqueeze(0).cuda()
            
            cd_noisy,_ = chamfer_distance(clean_cloud,noisy_cloud,batch_reduction="mean",point_reduction="mean",norm_type="L2")
            cd_denoised,_ = chamfer_distance(clean_cloud,denoised_cloud,batch_reduction="mean",point_reduction="mean",norm_type="L2")

            # print(cd_denoised)

            mean_noisy += [cd_noisy.item()]
            mean_denoised += [cd_denoised.item()]

            real_name = cloud_name[len(model)+1:]
            print(model)
            # print(cloud_name)
            # print(cd_denoised.item()/cd_noisy.item())
            if "l1_diverse_" in cloud_name:
                cloud_name = cloud_name[len("l1_diverse_"):]
            if cloud_name not in cloud_errors:
                cloud_errors[cloud_name] = []
            cloud_errors[cloud_name].append(cd_denoised.item()/cd_noisy.item())

            # if "1e3" in model:
            #     if cloud_name in model_1e3_error_map:
            #         model_1e3_error_map[real_name][model] = cd_denoised.item()
            #     else:
            #         model_1e3_error_map[real_name] = {}
            # elif "5e3" in model or "PCN" in path_model:
                
            #     print()
            #     if cloud_name in model_5e3_error_map:
            #         model_5e3_error_map[real_name][model] = cd_denoised.item()
            #     else:
            #         model_5e3_error_map[real_name]  = {}

        mean_noisy = np.mean(mean_noisy)
        mean_denoised = np.mean(mean_denoised)

        # if "1e3" in model:
        #     noisy_1e3 = mean_noisy
        #     std_1e3.append((model,mean_denoised/mean_noisy))
        # elif "5e3" in model or "PCN" in path_model:
        noisy_5e3 = mean_noisy
        std_5e3.append((model,mean_denoised/mean_noisy))
        # else:
        #     # raise ValueError(f"Model name {model} does not have the correct format")
        #     print(f"Model name {model} does not have the correct format")

    std_1e3.append(("noisy",1))
    std_5e3.append(("noisy",1))
    std_1e3.sort(key=lambda tup: tup[1])  # sorts in place
    std_5e3.sort(key=lambda tup: tup[1])  # sorts in place

    # for k,v in cloud_errors.items():
    #     print(f"{k}: {v[0]/v[1]}")
    # for model,value in std_1e3:
    #     f.write(f"{model}: CD ratio {'{:.2f}'.format(value)}\n")
    # f.write("\n")
    for model,value in std_5e3:
        f.write(f"{model}: CD ratio {'{:.2f}'.format(value)}\n")

    f.write("\n")

    # for cloud_name in model_1e3_error_map.keys():
    #     f.write(f"{cloud_name}:: ")
    #     for model,value in model_1e3_error_map[cloud_name].items():
    #         f.write(f"| {model}: CD ratio {'{:.2f}'.format(value/noisy_1e3)} |")
    #     f.write("\n")

    # for cloud_name in model_5e3_error_map.keys():
    #     f.write(f"{cloud_name}:: ")
    #     for model,value in model_1e3_error_map[cloud_name].items():
    #         f.write(f"| {model}: CD ratio {'{:.2f}'.format(value/noisy_1e3)} |")
    #     f.write("\n")

    f.close()

    f = open("CD_table_absolute.txt","w")
    # for model,value in std_1e3:
    #     f.write(f"{model}: CD ratio {'{:.2E}'.format(value*noisy_1e3)}\n")
    # f.write("\n")
    for model,value in std_5e3:
        f.write(f"{model}: CD ratio {'{:.2E}'.format(value*noisy_5e3)}\n")

    f.write("\n")

    # for cloud_name in model_1e3_error_map.keys():
    #     f.write(f"{cloud_name}:: ")
    #     for model,value in model_1e3_error_map[cloud_name].items():
    #         f.write(f"| {model}: CD ratio {'{:.2f}'.format(value)} |")
    #     f.write("\n")

    # for cloud_name in model_5e3_error_map.keys():
    #     f.write(f"{cloud_name}:: ")
    #     for model,value in model_1e3_error_map[cloud_name].items():
    #         f.write(f"| {model}: CD ratio {'{:.2f}'.format(value)} |")
    #     f.write("\n")


    f.close()



if __name__ == "__main__":
    main()