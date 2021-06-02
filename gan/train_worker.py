import argparse
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

proj_root = '.'
sys.path.insert(0, proj_root)
data_root = os.path.join(proj_root, 'data')
model_root = os.path.join(proj_root, 'models')
bpe = os.path.join(proj_root, 'dalle/cub200_bpe.json')

from gan.data_loader import BirdsDataset
from gan.data_loader_flowers import FlowersDataset
from gan.networks import Generator
from gan.networks import Discriminator
from gan.networks import ImgEncoder
from segmentation.train import Unet

from gan.train import train_gan

from gan.train import train_gan
from dalle.dalle import DALLE
from dalle import VQGanVAE1024
s
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gans')

    parser.add_argument('--reuse_weights',   action='store_true',
                        default=False, help='continue from last checkpoint')
    parser.add_argument('--load_from_epoch', type=int,
                        default=0,  help='load from epoch')
    parser.add_argument('--batch_size', type=int,
                        default=16, metavar='N', help='batch size.')
    parser.add_argument('--model_name', type=str,      default=None)
    parser.add_argument('--dataset',    type=str,      default=None,
                        help='which dataset to use [birds or flowers]')
    parser.add_argument('--epoch_decay', type=float, default=100,
                        help='decay learning rate by half every epoch_decay')
    parser.add_argument('--maxepoch', type=int, default=600, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--g_lr', type=float, default=0.0002, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--d_lr', type=float, default=0.0002, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--save_freq', type=int, default=5, metavar='N',
                        help='how frequent to save the model')
    parser.add_argument('--verbose_per_iter', type=int, default=50,
                        help='print losses per iteration')
    parser.add_argument('--KL_COE', type=float, default=0.5, metavar='N', # TAGAN
                        help='kl divergency coefficient.')
    parser.add_argument('--CONSIST_COE', type=float, default=10, metavar='N',
                        help='Consistency module coefficient.')
    parser.add_argument('--unet_checkpoint', type=str, default='', 
                        help='Unet checkpoint')
    parser.add_argument('--emb_dim', type=int, default=128, metavar='N',
                        help='Text and segmentation embeddim dim.')
    parser.add_argument('--n_plots', type=int, default=8,
                        help='Number of images to plot on tensorboard')
    parser.add_argument('--scode_dim', type=int, default=1024,
                        help='Segmentation code dimention')
    parser.add_argument('--manipulate',   action='store_true',
                        default=False, help='Framework for image manipulation.')

    args = parser.parse_args()


if __name__ == '__main__':
    # NNs
    vae_klass = VQGanVAE1024
    vae = vae_klass()
    dalle_params = dict(
    num_text_tokens=7800,
    text_seq_len=256,
    dim=256,
    depth=8,
    heads=8,
    dim_head=64,
    reversible=0,
    attn_types=('full', 'axial_row', 'axial_col', 'conv_like'),
    )
    netG = DALLE(vae=vae, **dalle_params)
    #netG   = Generator(tcode_dim=512, scode_dim=args.scode_dim, emb_dim=args.emb_dim, hid_dim=128)
    netD   = Discriminator(256,num_text_token=7800,text_seq_len=256)
    netS   = Unet()

    netD  = netD.cuda()
    netG  = netG.cuda()
    netS  = netS.cuda()

    data_name = args.dataset
    datadir = os.path.join(data_root, data_name)
    
    print('> Loading training data ...')
    if args.dataset == 'birds':
        dataset = BirdsDataset(datadir,bpe,batch=args.batch_size, mode='train')
    elif args.dataset == 'flowers':
        dataset = FlowersDataset(datadir, mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # create model folder
    model_name = '{}_{}'.format(args.model_name, data_name)
    model_folder = os.path.join(model_root, model_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    print('> Model folder: %s' % model_folder)

    print('> Start training ...')
    print('>> Run tensorboard --logdir models/')
    train_gan(dataloader, model_folder, netG, netD, netS, args)