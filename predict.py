import os, time, scipy.io, shutil
import numpy as np
import torch
import torch.nn as nn
import argparse
import cv2

from model.cbdnet import Network
from utils import read_img, chw_to_hwc, hwc_to_chw

from read_bmp import ReadBMPFile
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from SN import psnr

def predict():
    save_dir = './save_model/'  # 预训练模型保存路径

    model = Network()
    model.cuda()
    model = nn.DataParallel(model)

    model.eval()

    if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):    # 加载预训练模型
        # load existing model
        model_info = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'))
        model.load_state_dict(model_info['state_dict'])
    else:
        print('Error: no trained model detected!')
        exit(1)


    dir = "Dataset/1G_img/patches"

    for file in os.listdir(dir):
        if "src" in file:
            continue
        
        file_path = os.path.join(dir,  file)
        print(file_path)

        input_image = Image.open(file_path)
        input_image = np.array(input_image)
        input_image = input_image[:,:,np.newaxis]
        print(input_image.shape)

        input_var =  torch.from_numpy(hwc_to_chw(input_image)).unsqueeze(0).cuda()
        print("input: ", input_var.size())  # [1, 1, 32000, 32000]

        input_var = input_var.repeat(1, 3, 1, 1)
        print("input: ", input_var.size())  # [1, 3, 32000, 32000]
    
        with torch.no_grad():
            _, output = model(input_var)
    
        output_image = chw_to_hwc(output[0,...].cpu().numpy())
        output_image = np.uint8(np.round(np.clip(output_image, 0, 1) * 255.))[: ,: ,::-1]

        cv2.imwrite(os.path.join(dir, file.replace("noise", "denoise")), output_image)



    # 以下部分代码用于测试CBDNet在SIDD（部分）数据上的去噪效果
    """
    dir = 'Dataset/SIDD'
    Time = 0.0

    for d in os.listdir(dir):
        for file in os.listdir(os.path.join(dir, d)):

            if "GT" in file:
                print(os.path.join(dir, d, file))
            else:
                print(os.path.join(dir, d, file))
                input_image = read_img(os.path.join(dir, d, file))
                input_var =  torch.from_numpy(hwc_to_chw(input_image)).unsqueeze(0).cuda()
                print("input: ", input_var.size())  # [1, 3, 3000, 5328]
                
                start = time.time()
                lines = torch.split(input_var, 1000, dim=2) # 将原图张量拆分成定长的行

                Lines = []
                for line in lines:
                    print("line: ", line.size())
                    patches = torch.split(line, 1000, dim=3)# 每行再拆分成定长定宽的小块

                    Patches = []
                    for patch in patches:
                        print("patch: ", patch.size())
                        with torch.no_grad():
                            _, denoise_patch = model(patch) # 每小块分别运行去噪
                        Patches.append(denoise_patch)

                    Patches = torch.cat(Patches, dim=3)
                    print("Patches: ", Patches.size())
                    Lines.append(Patches)

                Lines = torch.cat(Lines, dim=2)
                print("Lines: ", Lines.size())

                output = Lines
                
                # with torch.no_grad():
                #     _, output = model(input_var)

                end = time.time()
                Time += (end-start)

                output_image = chw_to_hwc(output[0,...].cpu().numpy())
                output_image = np.uint8(np.round(np.clip(output_image, 0, 1) * 255.))[: ,: ,::-1]

                cv2.imwrite(os.path.join(dir, d, file.replace('NOISY', 'DENOISE')), output_image)
    
    num_pic = len(os.listdir(dir))
    print(num_pic)
    print("Total time: ", Time)
    print("avg time: ", Time/num_pic )
    """

    # dir = "expand_imgs"
    # Time = 0.0

    # for file in os.listdir(dir):
    #     print(os.path.join(dir, file))
    #     input_image = read_img(os.path.join(dir, file))
    #     input_var =  torch.from_numpy(hwc_to_chw(input_image)).unsqueeze(0).cuda()
    #     print("input: ", input_var.size())  # [1, 3, 3000, 5328]
        
    #     start = time.time()
    #     lines = torch.split(input_var, 3000, dim=2) # 将原图张量拆分成定长的行

    #     Lines = []
    #     for line in lines:
    #         print("line: ", line.size())
    #         patches = torch.split(line, 3000, dim=3)# 每行再拆分成定长定宽的小块

    #         Patches = []
    #         for patch in patches:
    #             print("patch: ", patch.size())
    #             with torch.no_grad():
    #                 _, denoise_patch = model(patch) # 每小块分别运行去噪
    #             Patches.append(denoise_patch)
    #             patch = patch.cpu()

    #         Patches = torch.cat(Patches, dim=3)
    #         print("Patches: ", Patches.size())
    #         Lines.append(Patches)
    #         Patches = Patches.cpu()
    #         torch.cuda.empty_cache()

    #     output = torch.cat(Lines, dim=2)
    #     print("output: ", output.size())

    #     end = time.time()
    #     Time += (end-start)
    #     print("Time: ", Time)

    #     output_image = chw_to_hwc(output[0,...].cpu().numpy())
    #     output_image = np.uint8(np.round(np.clip(output_image, 0, 1) * 255.))[: ,: ,::-1]

    #     Lines = Lines.cpu()
    #     output = output.cpu()
    #     torch.cuda.empty_cache()
    

    # num_pic = len(os.listdir(dir))
    # print(num_pic)
    # print("Total time: ", Time)
    # print("avg time: ", Time/num_pic )

# python predict.py CBSD68-dataset/noisy50 denoise


if __name__ == '__main__':
    # predict()

    dir = "Dataset/1G_img/patches"
    for file in os.listdir(dir):
        if not "src" in file:
            continue
        
        print(file)

        src_img = cv2.imread(os.path.join(dir, file), cv2.IMREAD_GRAYSCALE)
        print(src_img.shape)
        noise_img = cv2.imread(os.path.join(dir, file.replace("src", "noise")), cv2.IMREAD_GRAYSCALE)
        print(noise_img.shape)
        denoise_img = cv2.imread(os.path.join(dir, file.replace("src", "denoise")), cv2.IMREAD_GRAYSCALE)
        denoise_img = noise_img + denoise_img
        print(denoise_img.shape)

        PSNR1 = psnr(src_img, noise_img)
        PSNR2 = psnr(src_img, denoise_img)

        print("before denoising: ", PSNR1, "       After denoising: ", PSNR2)
    
