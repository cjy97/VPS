import cv2
import numpy as np
import os

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import random
import shutil

# 随机裁剪出一一对应的3200*3200尺寸的patch对
def random_crop_pic(noise_image, src_image, i):
    x = random.randint(0, 3200*8)
    y = random.randint(0, 3200*8)
    print("x: ", x, "   y: ", y)

    noise_patch = noise_image[x:x+3200, y:y+3200]
    cv2.imwrite("Dataset/1G_img/patches/" + str(i) + "_noise.bmp", noise_patch)
    src_patch = src_image[x:x+3200, y:y+3200]
    cv2.imwrite("Dataset/1G_img/patches/" + str(i) + "_src.bmp", src_patch)


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

# 从划分出的patches中随机抽样test_num作为测试集，移动到测试目录下
def sample_testdata(patch_dir, test_dir, test_num):
    src_file_list = [file for file in os.listdir(patch_dir) if "src" in file]

    print("num of src files: ", len(src_file_list))
    src_file_test_list = random.sample(src_file_list, test_num)
    print("num of src test files: ", len(src_file_test_list))
    for src_file in src_file_test_list:
        noise_file = src_file.replace("src", "noise")

        ori_path = os.path.join(patch_dir, src_file)
        dst_path = os.path.join(test_dir, src_file)
        shutil.move(ori_path, dst_path)

        ori_path = os.path.join(patch_dir, noise_file)
        dst_path = os.path.join(test_dir, noise_file)
        shutil.move(ori_path, dst_path)


# 利用小图像手动扩增出特大图像，以供模型测试（已弃用）
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
    
    noise_path = "Dataset/1G_img/500ms曝光.bmp"     # 以长曝光图像作为噪音图
    noise_image = Image.open(noise_path)
    noise_image = np.array(noise_image)
    noise_image = noise_image[:,:,np.newaxis]
    print(noise_image.shape)

    src_path = "Dataset/1G_img/100ms曝光.bmp"       # 以短曝光图像作为GT
    src_image = Image.open(src_path)
    src_image = np.array(src_image)
    src_image = src_image[:,:,np.newaxis]
    print(src_image.shape)

    # 将特大图像切分成一对一的小patch
    fixed_crop_pic(src_image, "Dataset/1G_img/patches", 320, "src")
    fixed_crop_pic(noise_image, "Dataset/1G_img/patches", 320, "noise")
    
    # 随机从patch划分出训练、测试集
    sample_testdata("Dataset/1G_img/patches", "Dataset/1G_img/patches_test", 1000)