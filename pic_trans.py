import cv2
import numpy as np
import os

def crop_pic(dir):
    for d in os.listdir(dir):
        for file in os.listdir(os.path.join(dir, d)):
            path = os.path.join(dir, d, file)
            print(path)
            img = cv2.imread(path)
            img = img[0:2000, 0:2000, :]
            cv2.imwrite(path, img)

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
    ori_dir = "ori_imgs"
    expand_dir = "expand_imgs"
    expand_pic(ori_dir, expand_dir, scale=8)
    