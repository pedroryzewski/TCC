import numpy as np
import pickle
import random
import os
import scipy.misc as misc
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from tokenizers import Tokenizer

from .proj_utils.torch_utils import to_binary

class BirdsDataset(Dataset):
    def __init__(self, workdir, bpe, batch, mode='train'):

        self.workdir = workdir
        self.mode = mode

        self.imsize =256

        self.image_filename = '/304images.pickle'
        self.segs_filename = '/304segmentations.pickle'

        pickle_path = os.path.join(self.workdir, self.mode)

        # images
        with open(pickle_path + self.image_filename, 'rb') as f:
            images = pickle.load(f)
            self.images = np.array(images)

        # segmentations
        with open(pickle_path + self.segs_filename, 'rb') as f:
            segmentations = pickle.load(f, encoding='bytes')
            self.segmentations = np.array(segmentations)

        # names
        with open(pickle_path + '/filenames.pickle', 'rb') as f:
            self.filenames = pickle.load(f)
        #print(len(self.filenames),' ', len(self.images)) 
        self.size = (len(self.filenames)//batch)*batch
        self.saveIDs = np.arange(self.size)

        # caption vectors
        caption_root = os.path.join(self.workdir, 'cub_icml')
        self.caption_vecs = self.load_caption_vecs(caption_root, self.filenames)

        # class ids
        with open(pickle_path + '/class_info.pickle', 'rb') as f:
            class_id = pickle.load(f, encoding="bytes")
            self.class_id = np.array(class_id)
        
        self.tokenizer = Tokenizer.from_file(bpe)
        
    def tokenize_text(self, txt):
        token_list = []
        sot_token = self.tokenizer.encode("<|startoftext|>").ids[0]
        eot_token = self.tokenizer.encode("<|endoftext|>").ids[0]
        #for txt in input_text:
        codes = [0] * 256
        text_token = self.tokenizer.encode(txt).ids
        tokens = [sot_token] + text_token + [eot_token]
        codes[:len(tokens)] = tokens
        caption_token = torch.LongTensor(codes).cuda()
        #token_list.append(caption_token)
        text = caption_token #torch.stack(caption_token)
        return text

    def load_caption_vecs(self, caption_root, filenames):
        word_vecs = torch.FloatTensor(self.size, 10, 50, 300)
        len_descs = torch.LongTensor(self.size, 10)
        for i, filename in enumerate(filenames):
            if(i < self.size):
                #print(i,'   ',self.size)
                data = torch.load(os.path.join(caption_root + '_vec', filename + '.pth'))
                word_vecs[i] = data['word_vec']
                len_descs[i] = torch.LongTensor(data['len_desc'])
        return (word_vecs, len_descs)

    def read_captions(self, filename, randid):
        cap_path = '{}/captions/{}.txt'.format(self.workdir, filename)
        with open(cap_path, "r") as f:
            captions = f.read().split('\n')
        return captions[randid]

    def transforms(self, image):
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(256),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ])
            return transform(image)
        elif self.mode == 'test':
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop(256),
                transforms.ToTensor()
            ])
            return transform(image)


    def __getitem__(self, index):
        w_index = np.random.randint(0, self.size)

        image   = self.images[index]
        seg     = self.segmentations[index]
        w_image = self.images[w_index, :, :, :]

        if self.mode == 'test':
            seed = 0
        else:
            seed = np.random.randint(2147483647)

        random.seed(seed)
        image = self.transforms(image)
        random.seed(seed)
        seg = self.transforms(seg)
        w_image = self.transforms(w_image)

        image   = image * 2 - 1
        seg     = to_binary(seg)
        w_image = w_image * 2 - 1

        filename = self.filenames[index]

        if self.mode == 'test':
            randid = 0 # choose instance caption
        else:
            randid = np.random.randint(0, 10) # choose instance caption

        word_vec = self.caption_vecs[0][index][randid]
        len_desc = self.caption_vecs[1][index][randid]
        raw_captions = self.read_captions(filename, randid)
        token = self.tokenize_text(raw_captions)
        data = [image, w_image, seg, word_vec, len_desc, raw_captions,token]

        if self.mode == 'test':
            data.append(self.saveIDs[index])
            data.append(self.class_id[index])

        return data

    def __len__(self):
        return self.size
