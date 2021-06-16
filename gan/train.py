import numpy as np
import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import time
from torchvision.utils import make_grid, save_image


from .proj_utils.torch_utils import set_lr, to_numpy, roll, to_binary
from .proj_utils.local_utils import save_images
from gan.networks import ImgEncoder

proj_root = '.'

def compute_g_loss(f_logit, f_logit_c, r_labels):
    criterion  = F.mse_loss
    r_g_loss   = criterion(f_logit, r_labels)
    f_g_loss_c = criterion(f_logit_c, r_labels)
    return r_g_loss + 10*f_g_loss_c

def compute_d_loss(r_logit, r_logit_c, w_logit_c, f_logit, r_labels, f_labels):
    criterion  = F.mse_loss
    r_d_loss   = criterion(r_logit,   r_labels)
    r_d_loss_c = criterion(r_logit_c, r_labels)
    w_d_loss_c = criterion(w_logit_c, f_labels)
    f_d_loss   = criterion(f_logit,   f_labels)
    return r_d_loss + 10*r_d_loss_c + 10*w_d_loss_c + f_d_loss

def get_kl_loss(mu, logvar):
    kld = mu.pow(2).add(logvar.mul(2).exp()).add(-1).mul(0.5).add(logvar.mul(-1))
    kl_loss = torch.mean(kld)
    return kl_loss

def shape_consistency_loss(f_seg, r_seg):
    consistency = F.mse_loss(f_seg, r_seg) # l1 bug?
    return consistency

def background_consistency_loss(f_bkgrds, bkgrds, f_segs, segs):
    crit_mask = ((f_segs < 1) & (segs < 1)).float().cuda()
    l1 = F.l1_loss(f_bkgrds, bkgrds, reduction='none')
    l1_mean = (crit_mask * l1).mean()
    return l1_mean

def idt_consistency_loss(f_images, r_images):
    consistency = F.l1_loss(f_images, r_images)
    return consistency

def obj_consistency_loss(img_images, txt_images, segs):
    l1 = F.l1_loss(img_images, txt_images, reduction='none')
    l1_mean = (segs * l1).mean()
    return l1_mean

def train_gan(dataloader, model_folder, netG, netS, netEs, args):
    """
    Parameters:
    ----------
    dataloader: 
        data loader. refers to fuel.dataset
    model_root: 
        the folder to save the models weights
    netG:
        Generator
    netS:
        Segmentation Network
    """

    g_lr = args.g_lr
    tot_epoch = args.maxepoch

    ''' configure optimizers '''
    paramsG = list(netG.parameters())+list(netEs.parameters())#+list(netEb.parameters())

    optimizerG = optim.Adam(paramsG, lr=g_lr, betas=(0.5, 0.999))

    ''' create tensorboard writer '''
    writer = SummaryWriter(model_folder)
    
    # --- load model from  checkpoint ---
    netS.load_state_dict(torch.load(args.unet_checkpoint))

    g_load = torch.load(args.dalle, map_location='cpu')
    gkeys = g_load['dalle']
    for k in list(gkeys):
        print(k)
        if k.split('.')[0]=='transformer':
            gkeys[k[:30]+'fn.'+k[30:]] = gkeys.pop(k)
    netG.load_state_dict(gkeys, strict=False)

    if args.reuse_weights:
        G_weightspath = os.path.join(
            model_folder, 'G_epoch{}.pth'.format(args.load_from_epoch))

        netG.load_state_dict(torch.load(G_weightspath))

        start_epoch = args.load_from_epoch + 1
        g_lr /= 2 ** (start_epoch // args.epoch_decay) 

    else:
        start_epoch = 1

    # --- Start training ---
    for epoch in range(start_epoch, tot_epoch + 1):
        start_timer = time.time()
        '''decay learning rate every epoch_decay epoches'''
        if epoch % args.epoch_decay == 0:
            g_lr = g_lr/2

            set_lr(optimizerG, g_lr)

        netG.train()
        netEs.train()
        netS.eval()

        for i, data in enumerate(dataloader):
            images, w_images, segs, txt_data, txt_len, raw, token = data

            # create labels
            r_labels = torch.FloatTensor(images.size(0)).fill_(1).cuda()
            f_labels = torch.FloatTensor(images.size(0)).fill_(0).cuda()

            it = epoch*len(dataloader) + i
            
            # to cuda
            images   = images.cuda()
            segs = segs.cuda()
            token = token.cuda()

            if args.manipulate:
                bimages = images # for text and seg mismatched backgrounds
                bsegs   = segs   # background segmentations
            else:
                bimages = roll(images, 2, dim=0) # for text and seg mismatched backgrounds
                segs    = roll(segs, 1, dim=0)   # for text mismatched segmentations

            segs_code = netEs(segs)

            f_images, img_loss, text_loss, smean = netG(token, bimages, segs_code)

            ''' UPDATE G '''
            optimizerG.zero_grad()

            f_segs = netS(f_images) # segmentation from Unet
            seg_consist_loss = shape_consistency_loss(f_segs, segs)
            bkg_consist_loss = background_consistency_loss(f_images, bimages, f_segs, segs)
            skl_loss = get_kl_loss(smean[0], smean[1]) # segmentation
            '''
            g_loss =  2   * text_loss \
                    + 10  * bkg_consist_loss \
                    + 10  * seg_consist_loss \
                    + 0.5 * skl_loss
            '''
            g_loss = (img_loss/4) \
                    + 2   * text_loss \
                    + 10   * bkg_consist_loss \
                    + 6   * seg_consist_loss \
                    + 0.5 * skl_loss

            #g_loss.requires_grad = True
            g_loss.backward()
            optimizerG.step()
            optimizerG.zero_grad()
                

            # --- visualize train samples----
            if it % args.verbose_per_iter == 0:
                writer.add_images('txt',         (images[:args.n_plots]+1)/2, it)
                writer.add_images('background', (images[:args.n_plots]+1)/2, it)
                writer.add_images('segmentation',   segs[:args.n_plots].repeat(1,3,1,1), it)
                writer.add_images('generated', (f_images[:args.n_plots]+1)/2, it)
                writer.add_scalar('g_lr', g_lr, it)
                writer.add_scalar('g_loss', to_numpy(g_loss).mean(), it)
                writer.add_scalar('img_loss', to_numpy(img_loss).mean(), it)
                writer.add_scalar('text_loss', to_numpy(text_loss).mean(), it)
                writer.add_scalar('seg_consist_loss', to_numpy(seg_consist_loss).mean(), it)
                writer.add_scalar('bkg_consist_loss', to_numpy(bkg_consist_loss).mean(), it)
                writer.add_scalar('segkl_loss', to_numpy(skl_loss).mean(), it)
                if args.manipulate:
                    writer.add_scalar('idt_consist_loss', to_numpy(idt_consist_loss).mean(), it)
            #print("passei porra")

        # --- save weights ---
        if epoch % args.save_freq == 0:

            netG  = netG.cpu()
            netEs = netEs.cpu()
            #netEb = netEb.cpu()

            torch.save(netG.state_dict(),  os.path.join(model_folder, 'G_epoch{}.pth'.format(epoch)))
            torch.save(netEs.state_dict(), os.path.join(model_folder, 'Es_epoch{}.pth'.format(epoch)))
            #torch.save(netEb.state_dict(), os.path.join(model_folder, 'Eb_epoch{}.pth'.format(epoch)))
            
            print('save weights at {}'.format(model_folder))
            netG  = netG.cuda()
            netEs = netEs.cuda()
            # netEb = netEb.cuda()
        
        vis_samples = [None for i in range(3)]
        vis_samples[0] = to_numpy(bimages)[0]
        # vis_samples[1] = to_numpy(w_images)[0]
        vis_samples[1] = to_numpy(segs)[0]
        vis_samples[2] = to_numpy(f_images)[0]
        print('Saving Image')
        save_images(vis_samples, save=True, save_path=os.path.join(proj_root , 'imgs2/E{}I{}.png'.format(epoch,i)), dim_ordering='th')
        save_image(f_images, os.path.join(proj_root , 'imgs/E{}I{}.png'.format(epoch,i)), normalize=True)

        end_timer = time.time() - start_timer
        print('epoch {}/{} finished [time = {}s] loss={} ...'.format(epoch, tot_epoch, end_timer, g_loss))

    writer.close()