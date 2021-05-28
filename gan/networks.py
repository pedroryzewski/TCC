import torch
import torch.nn as nn
from torch.nn import functional as F
import h5py
import numpy as np
import math
from gan.ViT_helper import trunc_normal_

#### TransGan

class matmul(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x1, x2):
        x = x1@x2
        return x

def count_matmul(m, x, y):
    num_mul = x[0].numel() * x[1].size(-1)
    # m.total_ops += torch.DoubleTensor([int(num_mul)])
    m.total_ops += torch.DoubleTensor([int(0)])

def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initialy created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=gelu, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def get_attn_mask(N, w):
    mask = torch.zeros(1, 1, N, N).cuda()
    for i in range(N):
        if i <= w:
            mask[:, :, i, 0:i+w+1] = 1
        elif N - i <= w:
            mask[:, :, i, i-w:N] = 1
        else:
            mask[:, :, i, i:i+w+1] = 1
            mask[:, :, i, i-w:i] = 1
    return mask

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., is_mask=0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mat = matmul()
        self.is_mask = is_mask
        self.remove_mask = False
        self.mask_4 = get_attn_mask(is_mask, 8)
        self.mask_5 = get_attn_mask(is_mask, 10)
        self.mask_6 = get_attn_mask(is_mask, 12)
        self.mask_7 = get_attn_mask(is_mask, 14)
        self.mask_8 = get_attn_mask(is_mask, 16)
        self.mask_10 = get_attn_mask(is_mask, 20)

    def forward(self, x, epoch):
        B, N, C = x.shape
        # print(self.qkv.in_features)
        # print("in ",x.shape)
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (self.mat(q, k.transpose(-2, -1))) * self.scale
        if self.is_mask:
            if epoch < 20:
                if epoch < 5:
                    mask = self.mask_4
                elif epoch < 10:
                    mask = self.mask_6
                elif epoch < 15:
                    mask = self.mask_8
                else:
                    mask = self.mask_10
                #attn = attn.to('cuda')
                attn = attn.masked_fill(mask.to(attn.get_device()) == 0, -1e9)
            else:
                pass
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = self.mat(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # print("out ",x.shape)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=gelu, norm_layer=nn.LayerNorm, is_mask=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, is_mask=is_mask)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, epoch):
        y = self.attn(x, epoch)
        y = self.norm1(y)
        x = x + self.drop_path(y)
        y = self.norm2(x)
        y = self.mlp(y)
        x = x + self.drop_path(y)
        return x

def pixel_upsample(x, H, W):
    B, N, C = x.size()
    assert N == H*W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(2)(x)
    B, C, H, W = x.size()
    x = x.view(-1, C, H*W)
    x = x.permute(0,2,1)
    return x, H, W

#### End TransGan

class CA(nn.Module):
    def __init__(self, noise_dim, emb_dim):
        super(CA, self).__init__()

        self.noise_dim = noise_dim
        self.emb_dim = emb_dim
        self.linear  = nn.Linear(noise_dim, emb_dim*2)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def sample_encoded_context(self, mean, logsigma, kl_loss=False, epsilon=None):
    
        if not isinstance(epsilon, torch.Tensor):
            epsilon = torch.cuda.FloatTensor(mean.size()).normal_()

        stddev  = logsigma.exp()
        
        return epsilon.mul(stddev).add_(mean)

    def forward(self, inputs, kl_loss=True, epsilon=None):
        '''
        inputs: (B, dim)
        return: mean (B, dim), logsigma (B, dim)
        '''
        out = self.relu(self.linear(inputs))
        mean = out[:, :self.emb_dim]
        log_sigma = out[:, self.emb_dim:]

        c = self.sample_encoded_context(mean, log_sigma, epsilon=epsilon)
        return c, mean, log_sigma

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        
        self.res_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
    
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(dim)
        )

    def forward(self, input):
        return self.res_block(input) + input

class VecToFeatMap(nn.Module):
    # used to project a sentence code into a set of feature maps
    def __init__(self, in_dim, bottom, embed_dim, activ=None):
        super(VecToFeatMap, self).__init__()
        
        out_dim = (bottom ** 2) * embed_dim
        
        self.bottom = bottom
        self.embed_dim = embed_dim
        self.tam = bottom ** 2

        self.out = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim)                
        )
    
    def forward(self, inputs):
        output = self.out(inputs)
        output = output.view(-1, self.tam, self.embed_dim)
        return output
#self.l1 = nn.Linear(args.latent_dim, (self.bottom_width ** 2) * self.embed_dim)
#x = self.l1(z).view(-1, self.bottom_width ** 2, self.embed_dim)
class ImgEncoder(nn.Module):
    def __init__(self, num_chan, out_dim):

        super(ImgEncoder, self).__init__()

        self.node = nn.Sequential(
            nn.Conv2d(num_chan, out_dim//16, kernel_size=3, padding=1, bias=False, stride=2),
            nn.LeakyReLU(0.2, True), # 32

            nn.Conv2d(out_dim//16, out_dim//8, kernel_size=3, padding=1, bias=False, stride=2),
            nn.BatchNorm2d(out_dim//8),
            nn.LeakyReLU(0.2, True), # 16

            nn.Conv2d(out_dim//8, out_dim//4, kernel_size=3, padding=1, bias=False, stride=2),
            nn.BatchNorm2d(out_dim//4),
            nn.LeakyReLU(0.2, True), # 8

            nn.Conv2d(out_dim//4, out_dim//2, kernel_size=3, padding=1, bias=False, stride=2),
            nn.BatchNorm2d(out_dim//2),
            nn.LeakyReLU(0.2, True), # 4

            nn.Conv2d(out_dim//2, out_dim, kernel_size=4, padding=0, bias=False, stride=1),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2, True) # 1
        )

    def forward(self, segs):
        return self.node(segs).squeeze(-1).squeeze(-1)

class Block_scale():
    def __init__(self, cur_dim, div, mask=0, num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., norm_layer=nn.LayerNorm):
        super(Block_scale, self).__init__()
        self.block1 = Block(
                        dim=cur_dim//div, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer)
        self.block2 = Block(
                        dim=cur_dim//div, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer, is_mask=0)
        self.block3 = Block(
                        dim=cur_dim//div, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer, is_mask=mask)

    def next(self, x, epoch):
        x = x.to('cuda')
        self.block1 = self.block1.to('cuda')
        y = self.block1(x,epoch)  
        #self.block1 = self.block1.to('cpu')

        y = y.to('cuda')        
        self.block2 = self.block2.to('cuda')
        y = self.block2(y,epoch)
        #self.block2 = self.block2.to('cpu')

        y = y.to('cuda')
        self.block3 = self.block3.to('cuda')
        y = self.block3(y,epoch)
        #self.block3 = self.block3.to('cpu')
        y = y.to('cpu')
        return y 

class Generator(nn.Module):
    def __init__(self, tcode_dim, scode_dim, emb_dim, hid_dim):
        """
        Parameters:
        ----------
        tcode_dim: int
            the dimension of sentence embedding
        scode_dim: int
            the dimension of segmentation embedding
        emb_dim : int
            the dimension of compressed sentence embedding.
        hid_dim: int
            used to control the number of feature maps.
        scode_dim : int
            the dimension of the segmentation embedding.
        """

        super(Generator, self).__init__()

        self.TCA = CA(tcode_dim, emb_dim)
        self.SCA = CA(scode_dim, emb_dim)
        self.BCA = CA(scode_dim, emb_dim)
        self.bottom_width = 8
        self.embed_dim = 1536
        self.vec_to_tensor = VecToFeatMap(384, self.bottom_width, self.embed_dim)
        cur_dim = self.embed_dim
        #self.l1 = nn.Linear(1024, cur_dim)
        #self.l1 = nn.Linear(1024, (self.bottom_width ** 2) * self.embed_dim)
        

        '''self.scale_4 = ResnetBlock(cur_dim)

        self.scale_8 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(cur_dim, cur_dim//2, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(cur_dim//2),
            nn.ReLU(True), 
            
            ResnetBlock(cur_dim//2),
        )

        self.scale_16 = nn.Sequential(
            ResnetBlock(cur_dim//2),
        )

        self.scale_32 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(cur_dim//2, cur_dim//4, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(cur_dim//4),
            nn.ReLU(True),
            
            ResnetBlock(cur_dim//4),
        )

        self.scale_64 = nn.Sequential(
            ResnetBlock(cur_dim//4),
        )'''
        num_heads = 4
        mlp_ratio=4.
        qkv_bias=False
        qk_scale=None
        drop_rate=0.
        attn_drop_rate=0.
        norm_layer=nn.LayerNorm

        self.block_4 = Block(
                    dim=cur_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer)
        self.block_8 = Block(
                    dim=cur_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer)
        self.block_16 = Block(
                    dim=cur_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer)
        self.block_32 = Block(
                    dim=cur_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer)
        self.block_64 = Block(
                    dim=cur_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer)
        
        self.scale_4 = Block_scale(cur_dim=cur_dim, div=4)
        #self.scale_8 = Block_scale(cur_dim=cur_dim, div=8)
        self.scale_16 = Block_scale(cur_dim=cur_dim, div=16)
        #self.scale_32 = Block_scale(cur_dim=cur_dim, div=32)
        self.scale_64 = Block_scale(cur_dim=cur_dim, div=64,mask=(64)**2)
        

        self.tensor_to_img_64 = nn.Sequential(
            #nn.ReflectionPad2d(1),
            #nn.Conv2d(cur_dim//4, 3, kernel_size=3, padding=0, bias=False),
            #nn.Tanh()
            nn.Conv2d(self.embed_dim//64, 3, 1, 1, 0)
        )

        self.apply(weights_init)

        self.txt_encoder_f = nn.GRUCell(300, 512)
        self.txt_encoder_b = nn.GRUCell(300, 512)

    def forward(self, epoch, txt_data=None, txt_len=None, seg_cond=None, bkg_cond=None, z_list=None,
        shape_noise=False, background_noise=False, vs=False):

        out = []
        if not z_list:
            txt_data = txt_data.permute(1,0,2)

            hi_f = torch.zeros(txt_data.size(1), 512, device=txt_data.device)
            hi_b = torch.zeros(txt_data.size(1), 512, device=txt_data.device)
            h_f = []
            h_b = []
            mask = []
            for i in range(txt_data.size(0)):
                mask_i = (txt_data.size(0) - 1 - i < txt_len).float().unsqueeze(1).cuda()
                mask.append(mask_i)
                hi_f = self.txt_encoder_f(txt_data[i], hi_f)
                h_f.append(hi_f)
                hi_b = mask_i * self.txt_encoder_b(txt_data[-i - 1], hi_b) + (1 - mask_i) * hi_b
                h_b.append(hi_b)
            mask = torch.stack(mask[::-1])
            h_f = torch.stack(h_f) * mask
            h_b = torch.stack(h_b[::-1])
            h = (h_f + h_b) / 2
            txt_cond = h.sum(0) / mask.sum(0)

            z_t, tmean, tlogsigma = self.TCA(txt_cond)
            z_s, smean, slogsigma = self.SCA(seg_cond)
            z_b, bmean, blogsigma = self.BCA(bkg_cond)


            z_list = [z_t, z_s, z_b]
            
            out.append((tmean, tlogsigma))
            out.append((smean, slogsigma))
            out.append((bmean, blogsigma))

        z_t, z_s, z_b = z_list

        if shape_noise:
            z_s = torch.cuda.FloatTensor(z_s.size()).normal_()
        if background_noise:
            z_b = torch.cuda.FloatTensor(z_b.size()).normal_()

        z_list = [z_t, z_s, z_b]

        z = torch.cat(z_list, dim=1)

        x = self.vec_to_tensor(z)
        #x = x.to('cpu')
        #epoch = epoch.to('cuda')
        #x = self.l1(batata).view(-1, self.bottom_width ** 2, self.embed_dim)
        '''
        x_4  = self.scale_4(x)
        x_8  = F.interpolate(x_4, scale_factor=2, mode='nearest')
        x_8  = self.scale_8(x_8)
        x_16 = F.interpolate(x_8, scale_factor=2, mode='nearest')
        x_16 = self.scale_16(x_16)
        x_32 = F.interpolate(x_16, scale_factor=2, mode='nearest')
        x_32 = self.scale_32(x_32)
        x_64 = F.interpolate(x_32, scale_factor=2, mode='nearest')
        x_64 = self.scale_64(x_64)
        '''
        H, W = self.bottom_width, self.bottom_width
        
        x = x.to('cuda')
        #self.block_4 = self.block_4.to('cpu')
        #self.block_8 = self.block_8.to('cpu')
        #self.block_16 = self.block_16.to('cpu')
        #self.block_32 = self.block_32.to('cpu')
        #self.block_64 = self.block_64.to('cpu')
    
        x = self.block_4(x,epoch)
        x = self.block_8(x,epoch)
        x = self.block_16(x,epoch)
        x = self.block_32(x,epoch)
        x = self.block_64(x,epoch)

        x, H, W = pixel_upsample(x, H, W)
        x_4 = self.scale_4.next(x,epoch)

        x_4, H, W = pixel_upsample(x_4, H, W)
        x_16 = self.scale_16.next(x_4,epoch)

        x_16, H, W = pixel_upsample(x_16, H, W)
        x_64 = self.scale_64.next(x_16,epoch)
         
        #img_64 = self.tensor_to_img_64(x_64)
        x_64 = x_64.to('cuda')
        img_64 = self.tensor_to_img_64(x_64.permute(0, 2, 1).view(-1, self.embed_dim//64, H, W))

        out.append(img_64)
        out.append(z_list)
        # print('ok')
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.eps = 1e-7

        self.encoder_1 = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.encoder_3 = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.GAP_1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.GAP_2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.GAP_3 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # text feature
        self.txt_encoder_f = nn.GRUCell(300, 512)
        self.txt_encoder_b = nn.GRUCell(300, 512)

        self.gen_filter = nn.ModuleList([
            nn.Linear(512, 256 + 1),
            nn.Linear(512, 512 + 1),
            nn.Linear(512, 512 + 1)
        ])
        self.gen_weight = nn.Sequential(
            nn.Linear(512, 3),
            nn.Softmax(-1)
        )

        self.classifier = nn.Conv2d(512, 1, 4)

        self.apply(init_weights)

    def forward(self, img, txt_data, txt_len, negative=False):
        txt_data = txt_data.permute(1,0,2)

        img_feat_1 = self.encoder_1(img)
        img_feat_2 = self.encoder_2(img_feat_1)
        img_feat_3 = self.encoder_3(img_feat_2)
        img_feats = [self.GAP_1(img_feat_1), self.GAP_2(img_feat_2), self.GAP_3(img_feat_3)]
        D = self.classifier(img_feat_3).squeeze()

        # text attention
        u, m, mask = self._encode_txt(txt_data, txt_len)
        att_txt = (u * m.unsqueeze(0)).sum(-1)
        att_txt_exp = att_txt.exp() * mask.squeeze(-1)
        att_txt = (att_txt_exp / att_txt_exp.sum(0, keepdim=True))

        weight = self.gen_weight(u).permute(2, 1, 0)

        sim = 0
        for i in range(3):
            img_feat = img_feats[i]
            W_cond = self.gen_filter[i](u).permute(1, 0, 2)
            W_cond, b_cond = W_cond[:, :, :-1], W_cond[:, :, -1].unsqueeze(-1)
            img_feat = img_feat.mean(-1).mean(-1).unsqueeze(-1)

            sim += torch.sigmoid(torch.bmm(W_cond, img_feat) + b_cond).squeeze(-1) * weight[i]

        sim = torch.clamp(sim + self.eps, max=1).t().pow(att_txt).prod(0)

        return D, sim

    def _encode_txt(self, txt, txt_len):
        hi_f = torch.zeros(txt.size(1), 512, device=txt.device)
        hi_b = torch.zeros(txt.size(1), 512, device=txt.device)
        h_f = []
        h_b = []
        mask = []
        for i in range(txt.size(0)):
            mask_i = (txt.size(0) - 1 - i < txt_len).float().unsqueeze(1).cuda()
            mask.append(mask_i)
            hi_f = self.txt_encoder_f(txt[i], hi_f)
            h_f.append(hi_f)
            hi_b = mask_i * self.txt_encoder_b(txt[-i - 1], hi_b) + (1 - mask_i) * hi_b
            h_b.append(hi_b)
        mask = torch.stack(mask[::-1])
        h_f = torch.stack(h_f) * mask
        h_b = torch.stack(h_b[::-1])
        u = (h_f + h_b) / 2
        m = u.sum(0) / mask.sum(0)
        return u, m, mask

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if m.weight.requires_grad:
            m.weight.data.normal_(std=0.02)
        if m.bias is not None and m.bias.requires_grad:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d) and m.affine:
        if m.weight.requires_grad:
            m.weight.data.normal_(1, 0.02)
        if m.bias.requires_grad:
            m.bias.data.fill_(0)

def weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1: 
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)     
