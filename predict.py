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


import random
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'

def load_model():
    save_dir = './save_model/'  # 预训练模型保存路径

    model = Network()
    model.cuda()
    model = nn.DataParallel(model)

    model.eval()

    if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
        # load existing model
        model_info = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'))
        model.load_state_dict(model_info['state_dict'])
    else:
        print('Error: no trained model detected!')
        exit(1)
    
    return model

def predict(test_dir):
    
    model = load_model()

    num = 0
    for file in os.listdir(test_dir):
        if "src" in file:   # 只读取测试目录下的noise图像
            continue

        print("num: ", num)
        num += 1
        
        file_path = os.path.join(test_dir,  file)
        print(file_path)

        input_image = read_img(file_path)
        input_image = np.mean(input_image, axis=2)[:,:,np.newaxis]
        input_var =  torch.from_numpy(hwc_to_chw(input_image)).unsqueeze(0).cuda()
        with torch.no_grad():
            _, output = model(input_var)
        

        src_image = read_img(file_path.replace("noise", "src"))
        src_image = np.mean(src_image, axis=2)[:,:,np.newaxis]
        src_var = torch.from_numpy(hwc_to_chw(src_image)).unsqueeze(0).cuda()

        mse = F.mse_loss(output, src_var)
        print("MSE: ", mse)
    
        output_image = chw_to_hwc(output[0,...].cpu().numpy())
        output_image = np.uint8(np.round(np.clip(output_image, 0, 1) * 255.))[: ,: ,::-1]

        cv2.imwrite(os.path.join(test_dir, file.replace("noise", "denoise")), output_image)


# 用于在CBDNEt输出图像上进一步去除黑色噪点
def denoise_black_points(test_dir):

    num = 0
    for file in os.listdir(test_dir):
        if not "denoise" in file:   # 只读取测试目录下的denoise(初步降噪)图像
            continue
        
        print("num: ", num)
        num += 1

        denoise_path = os.path.join(test_dir, file)
        denoise_img = Image.open(denoise_path)
        denoise_img = np.array(denoise_img)
        denoise_img = denoise_img[:,:,np.newaxis]

        median_img = cv2.medianBlur(denoise_img, 5) # 对初步降噪图像运行中值滤波
        _, mask = cv2.threshold(denoise_img, 80, 255, cv2.THRESH_BINARY_INV)    # 将初步降噪图像灰度阈值分割，得到掩模
        points = cv2.add(median_img, 0, mask=mask)  # 用掩模作用于中值滤波后图像，提取出噪点位置对应的滤波后像素
        _, denoise_img = cv2.threshold(denoise_img, 80, 0, cv2.THRESH_TOZERO)   # 将初步降噪图像中灰度低于阈值的点置零
        denoise_img = cv2.add(denoise_img, points, mask=None)   # 把掩模提取出的滤波后像素填充到对应位置上

        # _, mask = cv2.threshold(denoise_img, 170, 255, cv2.THRESH_BINARY)
        # points = cv2.add(median_img, 0, mask=mask)
        # _, denoise_img = cv2.threshold(denoise_img, 80, 0, cv2.THRESH_TOZERO_INV)
        # denoise_img = cv2.add(denoise_img, points, mask=None)

        cv2.imwrite(os.path.join(test_dir, file.replace("denoise", "denoise2")), denoise_img)


def cal_psnr(test_dir):
    sum_psnr1 = 0.0
    sum_psnr2 = 0.0
    sum_psnr3 = 0.0

    num = 0
    for file in os.listdir(test_dir):
        if not "src" in file:
            continue
        
        print("num: ", num)
        num += 1

        src_img = cv2.imread(os.path.join(test_dir, file), cv2.IMREAD_GRAYSCALE)
        # print(src_img.shape)
        noise_img = cv2.imread(os.path.join(test_dir, file.replace("src", "noise")), cv2.IMREAD_GRAYSCALE)
        # print(noise_img.shape)
        denoise_img = cv2.imread(os.path.join(test_dir, file.replace("src", "denoise")), cv2.IMREAD_GRAYSCALE)
        # print(denoise_img.shape)
        denoise2_img = cv2.imread(os.path.join(test_dir, file.replace("src", "denoise2")), cv2.IMREAD_GRAYSCALE)

        PSNR1 = psnr(src_img, noise_img)
        PSNR2 = psnr(src_img, denoise_img)
        PSNR3 = psnr(src_img, denoise2_img)

        print("before denoising: ", PSNR1, "       After denoising: ", PSNR2, "     After denoising blackpoints: ", PSNR3)

        sum_psnr1 += PSNR1
        sum_psnr2 += PSNR2
        sum_psnr3 += PSNR3
    
    print("avg PSNR before denoising: ", sum_psnr1 / num)
    print("avg PSNR after denoising: ", sum_psnr2 / num)
    print("avg PSNR after denoising black points: ", sum_psnr3 / num)


# 从高分辨率图像上随机裁剪出较大的patch对，测试去噪性能
def random_crop_test(patch_size, times): 
    
    model = load_model()
    
    noise_path = "Dataset/1G_img/100ms曝光.bmp"
    noise_image = Image.open(noise_path)
    noise_image = np.array(noise_image)
    noise_image = noise_image[:,:,np.newaxis]
    print(noise_image.shape)

    src_path = "Dataset/1G_img/500ms曝光.bmp"
    src_image = Image.open(src_path)
    src_image = np.array(src_image)
    src_image = src_image[:,:,np.newaxis]
    print(src_image.shape)


    sum_psnr1 = 0.0
    sum_psnr2 = 0.0
    for _ in range(times):
        x = random.randint(0, 3200*8)
        y = random.randint(0, 3200*8)
        print("x: ", x, "   y: ", y)

        noise_patch_ = noise_image[x:x+patch_size, y:y+patch_size]   # [patch_size, patch_size, 1]
        noise_patch = noise_patch_[:,:,::-1] / 255.0
        noise_patch = np.array(noise_patch).astype('float32')
        noise_var = torch.from_numpy(hwc_to_chw(noise_patch)).unsqueeze(0).cuda()

        with torch.no_grad():
            _, output = model(noise_var)

        output_patch = chw_to_hwc(output[0,...].cpu().numpy())
        output_patch = np.uint8(np.round(np.clip(output_patch, 0, 1) * 255.))[: ,: ,::-1]
        
        src_patch = src_image[x:x+patch_size, y:y+patch_size]
        
        PSNR1 = psnr(src_patch, noise_patch_)
        PSNR2 = psnr(src_patch, output_patch)
        sum_psnr1 += PSNR1
        sum_psnr2 += PSNR2
        print("before denoising: ", PSNR1, "       After denoising: ", PSNR2)

    print("avg PSNR before denoising: ", sum_psnr1 / times)
    print("avg PSNR after denoising: ", sum_psnr2 / times)

# 通过手动调节灰度+去除黑色噪点对1G特大图像进行处理
def denoise_1G_img():
    noise_path = "Dataset/1G_img/100ms曝光.bmp"
    noise_image = Image.open(noise_path)
    noise_image = np.array(noise_image)
    noise_image = noise_image[:,:,np.newaxis]
    print(noise_image.shape)

    src_path = "Dataset/1G_img/500ms曝光.bmp"
    src_image = Image.open(src_path)
    src_image = np.array(src_image)
    src_image = src_image[:,:,np.newaxis]
    print(src_image.shape)

    # denoise_img = noise_image + 32    # 手动调节灰度

    print("src mean: ", np.mean(src_image))
    print("noise mean: ", np.mean(noise_image))
    denoise_img = cv2.convertScaleAbs(noise_image, alpha=1.4, beta=0)   # 调节图像对比度
    print("denoise mean: ", np.mean(denoise_img))

    # 以下代码执行分块调节对比度
    """
    H, W, _ = noise_image.shape
    patch_size = 16000

    Y = H // patch_size
    X = W // patch_size
    print("X, Y: ", X, Y)

    row = []
    for y in range(Y):
        col = []
        for x in range(X):
            print("x: ", x, "   y:", y)
            noise_patch = noise_image[y*patch_size:min((y+1)*patch_size, H), x*patch_size:min((x+1)*patch_size, W), :]
            src_patch = src_image[y*patch_size:min((y+1)*patch_size, H), x*patch_size:min((x+1)*patch_size, W), :]

            avg_src = np.mean(src_patch)
            avg_noise = np.mean(noise_patch)
            patch = cv2.convertScaleAbs(noise_patch, alpha=avg_src/avg_noise, beta=0)

            col.append(patch)
        
        col = np.concatenate(col, axis=1)
        row.append(col)
    
    denoise_img = np.concatenate(row, axis=0)
    print("denoise_img: ", denoise_img.shape)
    """

    # median_img = cv2.medianBlur(denoise_img, 5)

    # 去除黑色噪点
    # _, mask = cv2.threshold(denoise_img, 80, 255, cv2.THRESH_BINARY_INV)
    # points = cv2.add(median_img, 0, mask=mask)
    # _, denoise_img = cv2.threshold(denoise_img, 80, 0, cv2.THRESH_TOZERO)
    # denoise_img = cv2.add(denoise_img, points, mask=None)

    # 去除白色噪点
    # _, mask = cv2.threshold(denoise_img, 120, 255, cv2.THRESH_BINARY)
    # points = cv2.add(median_img, 0, mask=mask)
    # _, denoise_img = cv2.threshold(denoise_img, 120, 0, cv2.THRESH_TOZERO_INV)
    # denoise_img = cv2.add(denoise_img, points, mask=None)

    _, mask = cv2.threshold(denoise_img, 100, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite("mask.bmp", cv2.resize(mask, (3200, 3200)))
    points = cv2.add(denoise_img, 0, mask=mask)
    points = cv2.convertScaleAbs(points, alpha=0.8, beta=0)
    cv2.imwrite("points.bmp", cv2.resize(points, (3200, 3200)))
    _, denoise_img = cv2.threshold(denoise_img, 100, 0, cv2.THRESH_TOZERO)
    denoise_img = cv2.add(denoise_img, points, mask=None)
    cv2.imwrite("denoise.bmp", cv2.resize(denoise_img, (3200, 3200)))

    denoise_img = denoise_img[:,:,np.newaxis]
    print("denoise_img: ", denoise_img.shape)

    PSNR1 = psnr(src_image, noise_image)
    PSNR2 = psnr(src_image, denoise_img)
    print("before denoising: ", PSNR1, "       After denoising: ", PSNR2)

    cv2.imwrite(os.path.join("Dataset/1G_img/", "crop_scaleAbs.bmp"), cv2.resize(denoise_img, (320,320)))

# 将1G特大分辨率图像分块，分别通过深度模型降噪，转成numpy数组再重新拼合
def crop_denoise_1G_img(noise_path, patch_size):

    model = load_model()

    noise_image = Image.open(noise_path)
    noise_image = np.array(noise_image)
    noise_image = noise_image[:,:,np.newaxis]
    noise_image = noise_image[:,:,::-1] / 255.0
    noise_image = np.array(noise_image).astype('float32')

    H, W, _ = noise_image.shape
    print(noise_image.shape)

    Y = H // patch_size
    X = W // patch_size
    print("X, Y: ", X, Y)

    row = []
    for y in range(Y):
        col = []
        for x in range(X):
            print("x: ", x, "   y:", y)
            patch = noise_image[y*patch_size:min((y+1)*patch_size, H), x*patch_size:min((x+1)*patch_size, W), :]
            print("patch: ", patch.shape)
            patch_var = torch.from_numpy(hwc_to_chw(patch)).unsqueeze(0).cuda()

            with torch.no_grad():
                _, output = model(patch_var)

            patch = chw_to_hwc(output[0,...].cpu().numpy())
            patch = np.uint8(np.round(np.clip(patch, 0, 1) * 255.))[: ,: ,::-1]

            col.append(patch)
        
        col = np.concatenate(col, axis=1)
        row.append(col)
    
    img = np.concatenate(row, axis=0)
    print("img: ", img.shape)

    cv2.imwrite("crop_denoise.bmp", img)
    

if __name__ == '__main__':
    test_dir = "Dataset/1G_img/patches_test/"
    # predict(test_dir)   # 调用CBDNet模型，对测试目录下的noise图像进行降噪，将所得denoise图像保存在同目录下
    # denoise_black_points(test_dir)
    # cal_psnr(test_dir)  # 对测试数据，计算降噪前后的平均psnr指标
    # random_crop_test(patch_size=2000, times=100)    # 在.246服务器上（1080ti，单卡11G显存），CBDNet单次能处理的图像尺寸上限约2000^2

    denoise_1G_img()
    # crop_denoise_1G_img("Dataset/1G_img/100ms曝光.bmp", patch_size=2000)

    # denoise_path = os.path.join(test_dir, "denoise_0_18.bmp")
    # denoise_img = Image.open(denoise_path)
    # denoise_img = np.array(denoise_img)
    # denoise_img = denoise_img[:,:,np.newaxis]

    # median_img = cv2.medianBlur(denoise_img, 5) # 对初步降噪图像运行中值滤波
    # _, mask = cv2.threshold(denoise_img, 100, 255, cv2.THRESH_BINARY_INV)    # 将初步降噪图像灰度阈值分割，得到掩模
    # points = cv2.add(median_img, 0, mask=mask)  # 用掩模作用于中值滤波后图像，提取出噪点位置对应的滤波后像素
    # _, denoise_img = cv2.threshold(denoise_img, 100, 0, cv2.THRESH_TOZERO)   # 将初步降噪图像中灰度低于阈值的点置零
    # denoise_img = cv2.add(denoise_img, points, mask=None)   # 把掩模提取出的滤波后像素填充到对应位置上

    # _, mask = cv2.threshold(denoise_img, 120, 255, cv2.THRESH_BINARY)
    # cv2.imwrite("mask.bmp", mask)
    # points = cv2.add(median_img, 0, mask=mask)
    # _, denoise_img = cv2.threshold(denoise_img, 120, 0, cv2.THRESH_TOZERO_INV)
    # denoise_img = cv2.add(denoise_img, points, mask=None)

    # # denoise_img = cv2.fastNlMeansDenoising(denoise_img, None, 3, 3, 7, 21)

    # cv2.imwrite("denoise_black_white.bmp", denoise_img)



    # src_light = cv2.imread("Dataset/1G_img/patches/src_25_25.bmp", cv2.IMREAD_GRAYSCALE)
    # src_dark = cv2.imread("Dataset/1G_img/patches/src_40_40.bmp", cv2.IMREAD_GRAYSCALE)
    # noise_light = cv2.imread("Dataset/1G_img/patches/noise_25_25.bmp", cv2.IMREAD_GRAYSCALE)
    # noise_dark = cv2.imread("Dataset/1G_img/patches/noise_40_40.bmp", cv2.IMREAD_GRAYSCALE)


    # print("src_light: ", np.mean(src_light))
    # print("src_dark: ", np.mean(src_dark))
    # print("noise_light: ", np.mean(noise_light))
    # print("noise_dark: ", np.mean(noise_dark))

# python predict.py CBSD68-dataset/noisy50 denoise