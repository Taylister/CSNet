import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms
from utils import *
import cfg
from tqdm import tqdm

from models import define_G, define_D, GANLoss, get_scheduler, update_learning_rate
from torchvision import models, transforms, datasets
import torchvision.transforms.functional as F
from dataset import CSNet_dataset, Example_dataset, To_tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def main():

    print('===> Loading datasets')
 
    trfms = To_tensor()
    example_data = Example_dataset(data_dir = cfg.predict_data_dir, transform = trfms)    
    example_loader = DataLoader(dataset = example_data, batch_size = 1, shuffle = False)


    print('===> Loading models')

    net_g = define_G(3, 1, 64,'batch', False, 'normal', 0.02, gpu_id="cpu")
    net_g.load_state_dict(torch.load(cfg.predict_ckpt_path, map_location='cpu'))
    
    savedir = os.path.join(cfg.predict_result_dir, "output")
         
    example_iter = iter(example_loader)
    net_g.eval()
    torch.set_grad_enabled(False)

    for ex_iter, batch in enumerate(example_iter):
         
        i_s = batch[0]
        name = str(batch[1][0])
                 
        o_mask = net_g(i_s)
        o_mask = o_mask.squeeze(0)
                 
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        o_mask = F.to_pil_image(o_mask)
        o_mask.save(os.path.join(savedir, name + '.png'))


if __name__ == "__main__":
    main()