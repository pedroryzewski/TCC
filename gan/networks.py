import torch
import torch.nn as nn
from torch.nn import functional as F
from dalle.dalle import DALLE
from dalle.dalle import VQGanVAE1024
from dalle.transformer import Transformer

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
    def __init__(self, in_dim, row, col, channel, activ=None):
        super(VecToFeatMap, self).__init__()
        
        out_dim = row*col*channel
        
        self.row = row
        self.col = col
        self.channel = channel

        self.out = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim)                
        )

    def forward(self, inputs):
        output = self.out(inputs)
        output = output.view(-1, self.channel, self.row, self.col)
        return output


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

class Discriminator(nn.Module):
    def __init__(self,dim,num_text_token=10000,text_seq_len=80):
        super(Discriminator, self).__init__()
        self.eps = 1e-7

        self.num_text_token = num_text_token
        self.text_seq_len = text_seq_len
       
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
            nn.MaxPool2d(2),
            nn.Conv2d(512, 512, 4, 2, padding=1, bias=False),
            nn.MaxPool2d(2),
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

        self.text_emb = nn.Embedding(8056, dim)

        self.text_pos_emb = nn.Embedding(text_seq_len + 1, dim)

        self.classifier = nn.Conv2d(512, 1, 4)
        
        self.txt_encoder_f = nn.GRUCell(256, 256)
        self.txt_encoder_b = nn.GRUCell(256, 256)

        self.gen_filter = nn.ModuleList([
            nn.Linear(256, 256 + 1),
            nn.Linear(256, 512 + 1),
            nn.Linear(256, 512 + 1)
        ])
        self.gen_weight = nn.Sequential(
            nn.Linear(256, 3),
            nn.Softmax(-1)
        )

        self.apply(init_weights)

    def forward(self, img, text, txt_len, negative=False):
        #txt_data = txt_data.permute(1,0,2)
        #print(img)
        img_feat_1 = self.encoder_1(img)
        #print(img_feat_1.shape)
        img_feat_2 = self.encoder_2(img_feat_1)
        #print(img_feat_2.shape)
        img_feat_3 = self.encoder_3(img_feat_2)
        #print(img_feat_3.shape)
        img_feats = [self.GAP_1(img_feat_1), self.GAP_2(img_feat_2), self.GAP_3(img_feat_3)]
      
        D = self.classifier(img_feat_3).squeeze()

        num_text_tokens = self.num_text_token
        text_seq_len = self.text_seq_len
        
        text = text[:, :text_seq_len]
        text_range = torch.arange(text_seq_len, device = text.device) + (num_text_tokens - text_seq_len)
        text = torch.where(text == 0, text_range, text)

        tokens = self.text_emb(text)
        tokens += self.text_pos_emb(torch.arange(text.shape[1], device = text.device))
        
        # text attention [8, 80, 256]
        tokens = tokens.permute(1,0,2)
        #print(tokens.shape, tokens.type())
        #print(tokens)
        u, m, mask = self._encode_txt(tokens, txt_len)
        
        att_txt = (u * m.unsqueeze(0)).sum(-1)
        for batata in att_txt[1]:
            #for bat in batata:
            #for ba in bat:
            print(batata)

        att_txt_exp = att_txt.exp() * mask.squeeze(-1)
        att_txt = (att_txt_exp / att_txt_exp.sum(0, keepdim=True))

        weight = self.gen_weight(u).permute(2, 1, 0)

        sim = 0
        for i in range(3):
            img_feat = img_feats[i]
            W_cond = self.gen_filter[i](u).permute(1, 0, 2)
            W_cond, b_cond = W_cond[:, :, :-1], W_cond[:, :, -1].unsqueeze(-1)
            img_feat = img_feat.mean(-1).mean(-1).unsqueeze(-1)
            #print("img feat",img_feat.shape)
            #print('w_cond',W_cond.shape)
            #print('b_cond',b_cond.shape)

            x = torch.bmm(W_cond, img_feat)
            x = x + b_cond
            sim += torch.sigmoid(x).squeeze(-1) * weight[i]
        
        sim = torch.clamp(sim + self.eps, max=1).t().pow(att_txt).prod(0)

        #print(D.shape)
        #print('sim2',sim)
        return D, sim

    def _encode_txt(self, txt, txt_len):
        hi_f = torch.zeros(txt.size(1), 256, device=txt.device)
        hi_b = torch.zeros(txt.size(1), 256, device=txt.device)
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
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1: 
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)        