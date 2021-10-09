import cv2
import os
import numpy as np
from time import time

from SN import psnr


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def Denoise(img, mode, kernel_size, std, times):
    if mode == 'mean':
        dst_img = img
        for i in range(times):
            dst_img = cv2.blur(dst_img, (kernel_size, kernel_size))

    elif mode == 'median':
        dst_img = cv2.medianBlur(img, 5)
    
    elif mode == 'gauss':
        dst_img = img
        for i in range(times):
            dst_img = cv2.GaussianBlur(dst_img, (kernel_size, kernel_size), std)
    return dst_img

def AddNoise(src_img, mode):
    rows, cols, _ = src_img.shape

    if mode == 'random':
        img = src_img
        for i in range(1000000):
            print(i)
            x = np.random.randint(0, rows)
            y = np.random.randint(0, cols)
            img[x, y, :] = 255

    elif mode == 'gauss':
        noise = np.random.normal(loc=0, scale=0.01, size=src_img.shape) * 255
        noise = noise.astype(int)
        img = src_img + noise
        img = np.maximum(img, 0)
        img = np.minimum(img, 255)
        img = np.uint8(img)

    return img


# if __name__ == '__main__':
#     src_dir = 'src_imgs'

#     noise_imgs = 'noise_imgs'
#     if not os.path.exists(noise_imgs):
#         os.mkdir(noise_imgs)

#     # for file in os.listdir(src_dir):
#     #     file_path = os.path.join(src_dir, file)
#     #     src_img = cv2.imread(file_path)

#     #     print(src_img.shape)
#     #     noise_img = AddNoise(src_img, mode='gauss')
#     #     print(psnr(noise_img, src_img))

#     #     cv2.imwrite(os.path.join(noise_imgs, file), noise_img)

#     denoise_dir = 'denoise_imgs'
#     if not os.path.exists(denoise_dir):
#         os.mkdir(denoise_dir)

#     for file in os.listdir(noise_imgs):
#         file = '2.jpg'
#         start1 = time()
#         # src_img = cv2.imread(os.path.join(src_dir, file))
#         noise_img = cv2.imread(os.path.join(noise_imgs, file))
#         print('file: ', file)

#         # a = psnr(src_img, noise_img)
#         # print(a)

#         start2 = time()
#         denoise_img = Denoise(noise_img, 'gauss')
#         print("denoise time: ", time() - start2 )

#         # b = psnr(src_img, denoise_img)
#         # print(a-b)

#         cv2.imwrite(os.path.join(denoise_dir, file), denoise_img)
#         print("whole time: ", time() - start1 )

if __name__ == '__main__':

    origin_dir = 'CBSD68-dataset/original_png'

    noisy_dir = 'CBSD68-dataset/noisy50'

    for file in os.listdir(origin_dir):
        print(file)
        origin_img = cv2.imread(os.path.join(origin_dir, file))
        noisy_img = cv2.imread(os.path.join(noisy_dir, file))

        noise_img = AddNoise(origin_img, mode='gauss')

        std = np.std(noisy_img.astype(int) - origin_img.astype(int))
        print("std: ", std)

        denoise_img = Denoise(noisy_img, mode='gauss', kernel_size=5, std=0, times=1)

        PSNR1 = psnr(origin_img, noisy_img)
        PSNR2 = psnr(origin_img, denoise_img)
        print("before denoising: ", PSNR1, "       After denoising: ", PSNR2)

        # cv2.imwrite(file, noise_img)