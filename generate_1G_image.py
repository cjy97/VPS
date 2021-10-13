import cv2
import os
import numpy as np

# 用于repeat扩充出高分辨率图像（暂时弃用）
if __name__ == '__main__':

    if not os.path.exists('src_imgs'):
        os.mkdir('src_imgs')

    ori_dir = 'origin_imgs'
    for file in os.listdir(ori_dir):
        file_path = os.path.join(ori_dir, file)
        ori_img = cv2.imread(file_path)
        ori_img = cv2.resize(ori_img, (320, 320))
        print("ori_img: ", ori_img.shape)

        src_img = ori_img.reshape(1, 320, 320, 3)
        src_img = np.repeat(src_img, repeats=100, axis=0)   #纵向扩增
        src_img = src_img.reshape((1, 32000, 320, 3))
        src_img = np.repeat(src_img, repeats=100, axis=1)   #横向扩增
        src_img = src_img.reshape((32000, 32000, 3))

        print("src_img: ", src_img.shape)
        # cv2.imshow("", src_img)
        # cv2.waitKey(0)

        cv2.imwrite(os.path.join('src_imgs', file), src_img)
        # break

