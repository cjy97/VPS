import os
import math
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
import cv2

from utils import read_img, chw_to_hwc, hwc_to_chw
from model.dncnn import DnCNN
from data.loader import My_Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser(description="DnCNN")
# parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=16, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=20, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
# parser.add_argument("--outf", type=str, default="logs", help='path of log files')
# parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
# parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
# parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
opt = parser.parse_args()

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant_(m.bias.data, 0.0)

class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def train_():
    save_dir = './save_model/'

    train_dataset = My_Dataset('./Dataset/1G_img/patches', 320) #Real('./data/SIDD_train/', 320, args.ps) + Syn('./data/Syn_train/', 100, args.ps)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    model = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    model.apply(weights_init_kaiming)
    model.cuda()
    model = nn.DataParallel(model)

    criterion = nn.MSELoss(size_average=False)
    criterion.cuda()

    optimizer = optim.Adam(model.parameters(), lr = opt.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs)

    print("len(train_loader): ", len(train_loader))
    for epoch in range(opt.epochs):
        print("epoch: ", epoch)
        # if epoch < opt.milestone:
        #     current_lr = opt.lr
        # else:
        #     current_lr = opt.lr / 10.
        
        for param_group in optimizer.param_groups:
            # param_group["lr"] = current_lr
            print('current learning rate %f' % param_group["lr"])

        losses = AverageMeter()
        model.train()
        for i, (noise_img, ori_img, sigma_img, flag) in enumerate(train_loader):
            noise_var = noise_img.cuda()
            ori_var = ori_img.cuda()

            gt = noise_var - ori_var
            # print("gt: ", gt.size())    # [bs, 1, 320, 320]
            out = model(noise_var)
            loss = criterion(out, gt) / (noise_var.size()[0] * 2)
            # print("loss: ", loss)
            losses.update(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # for predicted results
            # model.eval()
            # predict_var = torch.clamp(noise_var - model(noise_var), 0., 1.)
            # psnr_train = batch_PSNR(out_train, img_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f " %
                    (epoch+1, i+1, len(train_loader), loss.item()))
        
        scheduler.step()
        losses = losses.avg

        torch.save({
			'epoch': epoch + 1,
			'state_dict': model.state_dict(),
			'optimizer' : optimizer.state_dict()}, 
			os.path.join(save_dir, 'my_DnCNN.pth.tar'))

        print('Epoch [{0}]\t'
			'lr: {lr:.6f}\t'
			'Loss: {loss:.5f}'
			.format(
			epoch,
			lr=optimizer.param_groups[-1]['lr'],
			loss=losses))

def eval_(test_dir):
    save_dir = './save_model/'

    model = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    model.cuda()
    model = nn.DataParallel(model)
    model_info = torch.load(os.path.join(save_dir, 'my_DnCNN.pth.tar'))
    model.load_state_dict(model_info['state_dict'])

    model.eval()

    num = 0
    for file in os.listdir(test_dir):
        if not "noise_" in file:   # 只读取测试目录下的noise图像
            continue
    
        print("num: ", num)
        num += 1
        
        file_path = os.path.join(test_dir,  file)
        print(file_path)

        input_image = read_img(file_path)
        input_image = np.mean(input_image, axis=2)[:,:,np.newaxis]
        input_var =  torch.from_numpy(hwc_to_chw(input_image)).unsqueeze(0).cuda()

        with torch.no_grad():
            noise_var = model(input_var)

            output = torch.clamp(input_var - noise_var, 0., 1.)

            output_image = chw_to_hwc(output[0,...].cpu().numpy())
            output_image = np.uint8(np.round(np.clip(output_image, 0, 1) * 255.))[: ,: ,::-1]

            src_image = read_img(file_path.replace("noise", "src"))
            src_image = np.mean(src_image, axis=2)[:,:,np.newaxis]
            src_var = torch.from_numpy(hwc_to_chw(src_image)).unsqueeze(0).cuda()

            mse = F.mse_loss(output, src_var)
            print("MSE: ", mse)

            cv2.imwrite(os.path.join(test_dir, file.replace("noise", "denoise_dncnn")), output_image)
        
        # to continue

if __name__ == '__main__':
    train_()

    # test_dir = "Dataset/1G_img/patches_test/"
    # eval_(test_dir)