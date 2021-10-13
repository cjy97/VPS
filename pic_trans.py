import cv2
import numpy as np
import os

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import random

# 随机裁剪出一一对应的3200*3200尺寸的patch对
def random_crop_pic(noise_image, src_image, i):
    x = random.randint(0, 3200*8)
    y = random.randint(0, 3200*8)
    print("x: ", x, "   y: ", y)

    noise_patch = noise_image[x:x+3200, y:y+3200]
    cv2.imwrite("Dataset/1G_img/patches/" + str(i) + "_noise.bmp", noise_patch)
    src_patch = src_image[x:x+3200, y:y+3200]
    cv2.imwrite("Dataset/1G_img/patches/" + str(i) + "_src.bmp", src_patch)

    # for d in os.listdir(dir):
    #     for file in os.listdir(os.path.join(dir, d)):
    #         path = os.path.join(dir, d, file)
    #         print(path)
    #         img = cv2.imread(path)
    #         img = img[0:2000, 0:2000, :]
    #         cv2.imwrite(path, img)

# 将特大图像划分成patch^2大小的patch
def fixed_crop_pic(image, patch_dir, patch_size, type):
    height, width, _ = image.shape
    x_num = height // patch_size
    y_num = width // patch_size
    print(x_num)
    print(y_num)

    num = 0
    for i in range(x_num):
        for j in range(y_num):
            print(num)
            num += 1
            patch = image[i*patch_size : (i+1)*patch_size, j*patch_size : (j+1)*patch_size]
            file_name = "{}_{}_{}.bmp".format(type, i, j)
            print(file_name)
            cv2.imwrite(os.path.join(patch_dir, file_name), patch)



def expand_pic(ori_dir, expand_dir, scale):
    if not os.path.exists(expand_dir):
        os.mkdir(expand_dir)

    for file in os.listdir(ori_dir):
        file_path = os.path.join(ori_dir, file)
        ori_img = cv2.imread(file_path)
        # ori_img = resize(ori_img, (3200, 3200))
        print("ori_img: ", ori_img.shape)

        h, w, _ = ori_img.shape

        img = ori_img.reshape((1, h, w, 3))
        img = np.repeat(img, repeats=scale, axis=0)
        img = img.reshape((1, h*scale, w, 3))
        img = np.repeat(img, repeats=scale, axis=1)
        img = img.reshape((h*scale, w*scale, 3))

        print("expand_img: ", img.shape)
        cv2.imwrite(os.path.join(expand_dir, file), img)

if __name__ == '__main__':
    # ori_dir = "ori_imgs"
    # expand_dir = "expand_imgs"
    # expand_pic(ori_dir, expand_dir, scale=8)
    
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

    # fixed_crop_pic(src_image, "Dataset/1G_img/patches", 320, "src")
    fixed_crop_pic(noise_image, "Dataset/1G_img/patches", 320, "noise")
    
