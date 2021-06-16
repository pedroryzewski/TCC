from math import log2, sqrt
import torch
from torch import nn, einsum
import torch.nn.functional as F

from axial_positional_embedding import AxialPositionalEmbedding
from einops import rearrange

from dalle import distributed_utils
from dalle.vae import OpenAIDiscreteVAE
from dalle.vae import VQGanVAE1024
from dalle.transformer import Transformer

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def always(val):
    def inner(*args, **kwargs):
        return val
    return inner

def is_empty(t):
    return t.nelement() == 0

def masked_mean(t, mask, dim = 1):
    t = t.masked_fill(~mask[:, :, None], 0.)
    return t.sum(dim = 1) / mask.sum(dim = 1)[..., None]

def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# sampling helpers

def top_k(logits, thres = 0.5, full = False):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    if full == False:
        probs.scatter_(1, ind, val)
    else:
        probs.scatter_(2, ind, val)
    return probs

# discrete vae class

class ResBlock(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x

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
        mean = out[:, :, :self.emb_dim]
        log_sigma = out[:, :, self.emb_dim:]

        c = self.sample_encoded_context(mean, log_sigma, epsilon=epsilon)
        c = c.permute(0,2,1)
        return c, mean, log_sigma

class DALLE(nn.Module):
    def __init__(
        self,
        *,
        dim,
        vae,
        num_text_tokens = 10000,
        text_seq_len = 256,
        depth,
        heads = 8,
        dim_head = 64,
        reversible = False,
        attn_dropout = 0.,
        ff_dropout = 0,
        sparse_attn = False,
        attn_types = None,
        loss_img_weight = 7,
        batch=16,
    ):
        super().__init__()

        image_size = vae.image_size
        num_image_tokens = vae.num_tokens
        image_fmap_size = (vae.image_size // (2 ** vae.num_layers))
        image_seq_len = image_fmap_size ** 2

        num_text_tokens = num_text_tokens + text_seq_len  # reserve unique padding tokens for each position (text seq len)

        self.batch = batch
        self.text_emb = nn.Embedding(num_text_tokens, dim)
        self.image_emb = nn.Embedding(num_image_tokens, dim)

        self.text_pos_emb = nn.Embedding(text_seq_len + 1, dim) # +1 for <bos>
        self.image_pos_emb = AxialPositionalEmbedding(dim, axial_shape = (image_fmap_size, image_fmap_size))

        self.num_text_tokens = num_text_tokens # for offsetting logits index and calculating cross entropy loss
        self.num_image_tokens = num_image_tokens

        self.text_seq_len = text_seq_len
        self.image_seq_len = image_seq_len

        seq_len = text_seq_len + image_seq_len + 256
        total_tokens = num_text_tokens + num_image_tokens
        self.total_tokens = total_tokens
        self.total_seq_len = seq_len

        self.SCA = CA(169, 256)
        self.vae = vae
        set_requires_grad(self.vae, False) # freeze VAE from being trained

        self.transformer = Transformer(
            dim = dim,
            causal = True,
            seq_len = seq_len,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            reversible = reversible,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            attn_types = attn_types,
            image_fmap_size = image_fmap_size,
            sparse_attn = sparse_attn
        )

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.total_tokens),
        )
        print('seq_len', seq_len)
        print('total_tokens', total_tokens)
        print('text_seq_len', text_seq_len)
        print('num_text_tokens', num_text_tokens)

        seq_range = torch.arange(seq_len)
        logits_range = torch.arange(total_tokens)

        seq_range = rearrange(seq_range, 'n -> () n ()')
        logits_range = rearrange(logits_range, 'd -> () () d')

        logits_mask = (
            ((seq_range >= text_seq_len) & (logits_range < num_text_tokens)) |
            ((seq_range < text_seq_len) & (logits_range >= num_text_tokens))
        )

        self.register_buffer('logits_mask', logits_mask, persistent=False)
        self.loss_img_weight = loss_img_weight
        
    def process(
        self,
        text,
        image = None,
        segs = None,
        mask = None,
        return_loss = False
    ):
        
        device, total_seq_len = text.device, self.total_seq_len

        text_range = torch.arange(self.text_seq_len, device = device) + (self.num_text_tokens - self.text_seq_len)
        text = torch.where(text == 0, text_range, text)

        # add <bos>
        text = F.pad(text, (1, 0), value = 0)
        #tokens = text
        tokens = self.text_emb(text)
        tokens += self.text_pos_emb(torch.arange(text.shape[1], device = device))# shape devia ter [1]
        tokens = torch.cat((tokens,segs), dim=1)
        seq_len = tokens.shape[1]
        #print("imgem cru: ",image.shape)
        if exists(image) and not is_empty(image):
            is_raw_image = len(image.shape) == 4

            if is_raw_image:
                image_size = self.vae.image_size
                assert tuple(image.shape[1:]) == (3, image_size, image_size), f'invalid image of dimensions {image.shape} passed in during training'

                image = self.vae.get_codebook_indices(image)

            image_len = image.shape[1]
            #print(image.shape)
            image_emb = self.image_emb(image)
            #print(image_emb.shape)
            image_emb += self.image_pos_emb(image_emb)
            #print(image_emb.shape)
            
            #print("imagem editada: ",image_emb.shape)
            #print(image_emb)
            
            tokens = torch.cat((tokens, image_emb), dim = 1)

            seq_len += image_len
        #print("imagem + texto: ", tokens.shape)
        # when training, if the length exceeds the total text + image length
        # remove the last token, since it needs not to be trained

        if tokens.shape[1] > total_seq_len:
            seq_len -= 1
            tokens = tokens[:, :-1]
        #print("imagem+texto antes de entra na transformer", tokens.shape)
        out = self.transformer(tokens)
        logits = self.to_logits(out)

        # mask logits to make sure text predicts text (except last token), and image predicts image

        print('logits shape? ',logits.shape)
        logits_mask = self.logits_mask[:, :seq_len]
        print('logits mask shape? ', logits_mask.shape)
        print('logits mask? ', logits_mask)
        max_neg_value = -torch.finfo(logits.dtype).max
        print(logits)
        logits.masked_fill_(logits_mask, max_neg_value)
        print(logits)
        offsetted_image = image + self.num_text_tokens
        labels = torch.cat((text[:, 1:], offsetted_image), dim = 1)

        nlogits = rearrange(logits, 'b n c -> b c n')
        #print(nlogits.shape)
        #print(labels.shape)
        loss_text = F.cross_entropy(nlogits[:, :, :self.text_seq_len], labels[:, :self.text_seq_len])
        loss_img = F.cross_entropy(nlogits[:, :, -self.image_seq_len:], labels[:, self.text_seq_len:])

        #loss = (loss_text + self.loss_img_weight * loss_img) / (self.loss_img_weight + 1)
        return logits, loss_img, loss_text
        
    def forward(
        self,
        text,
        img,
        segs,
        clip = None,
        mask = None,
        filter_thres = 0.5,
        temperature = 1.,
        num_init_img_tokens = None
    ):
        vae, text_seq_len, image_seq_len, num_text_tokens = self.vae, self.text_seq_len, self.image_seq_len, self.num_text_tokens
        total_len = text_seq_len + image_seq_len

        text = text[:, :text_seq_len] # make sure text is within bounds
        out = text
        
        #print(segs.shape)
        segs = segs.view(self.batch,256, -1)   # TODO: substituir primeiro parametro por batch
        #print(segs.shape)
        z_s, smean, slogsigma = self.SCA(segs) # make sure text is within bounds
        smean_var = (smean, slogsigma)
        
        #### Full 
        if exists(img):
            image_size = vae.image_size
            assert img.shape[1] == 3 and img.shape[2] == image_size and img.shape[3] == image_size, f'input image must have the correct image size {image_size}'

            indices = vae.get_codebook_indices(img)
            indices = indices[:, :256]
            out = torch.cat((out, indices), dim = -1)

        text, image = out[:, :text_seq_len], out[:, text_seq_len:]
        logits, loss_img, loss_text = self.process(text, image, z_s, mask = mask)
            
        #filtered_logits = top_k(logits, thres = filter_thres, full=True)   ## TODO erro aqui porra
        probs = F.softmax(logits / temperature, dim = -1)
        #for i, p in enumerate(probs):
        #    if i == 0:
        #        sample = torch.multinomial(p, 1)
        #    else:
        #        if i == 1:
        #            sample = torch.stack((sample, torch.multinomial(p, 1)),dim = 0)
        #        else:
        #            sample = torch.cat((sample, torch.multinomial(p, 1).unsqueeze(0)),dim = 0)
        #sample -= num_text_tokens# if is_image else 0) # offset sampled token if it is an image token, since logit space is composed of text and then image tokens
        #print("out:",out.shape)
        #print("sample:",sample.shape)
        #out = torch.cat((out, sample.squeeze()), dim=1)
        #### Full   
            

        #print('$$$$$$$$$$$$$$$$$$$$FOIIIIII$$$$$$$$$$$$$$$$$$$$$$$$$$$$4')
        #print('$$$$$$$$$$$$$$$$$$$$FOIIIIII$$$$$$$$$$$$$$$$$$$$$$$$$$$$4')

        img_seq = probs[:, -image_seq_len:, -1024:]
        #print("img seq ", img_seq.shape)

        images = vae.decode(img_seq)

        #if exists(clip):
        #    scores = clip(text_seq, images, return_loss = False)
        #    return images, scores

        return images, loss_img, loss_text, smean_var
