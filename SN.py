import cv2
import numpy as np
import math

def filter(img):
    dst_img = cv2.GaussianBlur(img, (5,5), 0)
    return dst_img


def snr(img1, img2):
    h, w, _ = img1.shape
    MSE = np.mean((img1 - img2) ** 2)
    # print("mse: ", MSE)
    if MSE < 1.0e-10:
        return 100

    S = np.mean(img1*img1)
    return 10 * math.log10(S / MSE)

def psnr(img1, img2):
    img1 = np.float64(img1)
    img2 = np.float64(img2)

    # h, w, _ = img1.shape
    MSE = np.mean((img1 - img2) ** 2)
    # print("mse: ", MSE)
    if MSE == 0:
        return 100

    PIXEL_MAX = 255.0
    return 10 * math.log10( (PIXEL_MAX*PIXEL_MAX) / MSE)

if __name__ == '__main__':
    src_img = cv2.imread('lenna.jpg')
    dst_img = filter(src_img)

    # cv2.imshow("", src_img)
    # cv2.waitKey(0)
    # cv2.imshow("", dst_img)
    # cv2.waitKey(0)

    print("SNR: ", snr(src_img, dst_img))
    print("PSNR: ", psnr(src_img, dst_img))
