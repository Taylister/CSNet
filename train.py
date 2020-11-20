# Training script for the SRNet. Refer README for instructions.
# author: Niwhskal
# github : https://github.com/Niwhskal/SRNet

import numpy as np
import os
import torch
import torchvision.transforms
from utils import *
import cfg
from tqdm import tqdm
import torchvision.transforms.functional as F
from skimage.transform import resize
from skimage import io
from models import define_G, define_D, GANLoss, get_scheduler, update_learning_rate
from torchvision import models, transforms, datasets
from dataset import CSNet_dataset, Example_dataset, To_tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
        


def clip_grad(model):
    
    for h in model.parameters():
        h.data.clamp_(-0.01, 0.01)

def main():
    # ================================================
    # Preparation
    # ================================================
    if not torch.cuda.is_available():
        raise Exception('At least one gpu must be available.')
    
    gpu = torch.device('cuda:0')

    train_name = get_train_name()

    print('===> Loading datasets')
    train_data = CSNet_dataset(cfg, torp='train')
    train_data = DataLoader(dataset = train_data, batch_size = cfg.batch_size, shuffle = False, collate_fn = custom_collate,  pin_memory = True)
    
    # trfms = To_tensor()
    # example_data = Example_dataset(transform = trfms)    
    # example_loader = DataLoader(dataset = example_data, batch_size = 1, shuffle = False)


    print('===> Building models')
    net_g = define_G(3, 1, 64,'batch', False, 'normal', 0.02, gpu_id=gpu)
    net_d = define_D(3 + 1, 64, 'basic', gpu_id=gpu)

    criterionGAN = GANLoss().to(gpu)
    criterionL1 = nn.L1Loss().to(gpu)
    criterionMSE = nn.MSELoss().to(gpu)

    # setup optimizer
    optimizer_g = optim.Adam(net_g.parameters(), lr=cfg.lr, betas=(opt.beta1, 0.999))
    optimizer_d = optim.Adam(net_d.parameters(), lr=cfg.lr, betas=(opt.beta1, 0.999))
    net_g_scheduler = get_scheduler(optimizer_g, opt)
    net_d_scheduler = get_scheduler(optimizer_d, opt)

    # train
    trainer = iter(train_data)
    #example_iter = iter(example_loader)

    for epoch in range(1, cfg.niter + cfg.niter_decay + 1):
        for iteration, batch in enumerate(trainer):
            # forward
            input_src, mask_true = batch[0].to(gpu), batch[1].to(gpu)
            mask_false = net_g(input_src)

            ######################
            # (1) Update D network
            ######################

            optimizer_d.zero_grad()
            
            # train with fake
            fake_pair = torch.cat((input_src, mask_false), 1)
            pred_fake = net_d.forward(fake_pair.detach())
            loss_d_fake = criterionGAN(pred_fake, False)

            # train with real
            real_pair = torch.cat((input_src, mask_true), 1)
            pred_real = net_d.forward(real_pair)
            loss_d_real = criterionGAN(pred_real, True)
            
            # Combined D loss
            loss_d = (loss_d_fake + loss_d_real) * 0.5

            loss_d.backward()
        
            optimizer_d.step()

            ######################
            # (2) Update G network
            ######################

            optimizer_g.zero_grad()

            # First, G(A) should fake the discriminator
            fake_pair = torch.cat((input_src, mask_false), 1)
            pred_fake = net_d.forward(fake_pair)
            loss_g_gan = criterionGAN(pred_fake, True)

            # Second, G(A) = B
            loss_g_l1 = criterionL1(mask_false, mask_true) * opt.lamb
            
            loss_g = loss_g_gan + loss_g_l1
            
            loss_g.backward()

            optimizer_g.step()

            print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
                epoch, iteration, len(trainiter), loss_d.item(), loss_g.item()))

        update_learning_rate(net_g_scheduler, optimizer_g)
        update_learning_rate(net_d_scheduler, optimizer_d)

        # # test
        # avg_psnr = 0
        # for batch in testing_data_loader:
        #     input, target = batch[0].to(device), batch[1].to(device)

        #     prediction = net_g(input)
        #     mse = criterionMSE(prediction, target)
        #     psnr = 10 * log10(1 / mse.item())
        #     avg_psnr += psnr
        # print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))

        # #checkpoint
        # if epoch % 50 == 0:
        #     if not os.path.exists("checkpoint"):
        #         os.mkdir("checkpoint")
        #     if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
        #         os.mkdir(os.path.join("checkpoint", opt.dataset))
        #     net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, epoch)
        #     net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(opt.dataset, epoch)
        #     torch.save(net_g, net_g_model_out_path)
        #     torch.save(net_d, net_d_model_out_path)
        #     print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))

if __name__ == '__main__':
    main()