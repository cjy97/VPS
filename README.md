# VPS
VPS项目相关代码（特大图像降噪部分）


# 降噪模型采用DnCNN网络
参考文献：[Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising](https://arxiv.org/pdf/1608.03981v1.pdf)
预训练模型可从[这里](https://github.com/cjy97/DnCNN-PyTorch)获取


运行以下指令，执行图像降噪：
    $ python predict.py

其中调用crop_denoise_1G_img(noise_path, src_path, patch_size, model_type)方法，将特大分辨率的图像分块，分别降噪后重新拼接。

- `noise_path`: 待处理的噪音图像。
- `src_path`: 无噪音的原图，作为GT供计算降噪前后PSNR指标。（若无，可将相关代码删去。）
- `patch_size`: 表示分块的尺寸，根据设备显存动态调整。
- `model_type`：使用的深度降噪网络类型，默认为‘DnCNN’。
