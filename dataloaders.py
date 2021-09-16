import sys
import argparse
import json
import os
import threading
import multiprocessing
from os.path import exists, join, split, dirname

import time

import numpy as np
import shutil

import sys
from PIL import Image
import PIL
import torch
import torch.utils.data
from torch import nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable,Function
import torchvision.transforms as ttransforms
import data_transforms as transforms
import dataset,cv2
import sys
import random

from torch.utils.data import DataLoader, Dataset
import numpy as np
import io
from PIL import Image


class McDataset_seg(Dataset):
    def __init__(self, data_dir, phase, transforms, list_dir=None,
                 out_name=False, out_size=False, binary=False, repeat=1,args = None, aug = None, t2 = None, train_flag = False):
        self.data_dir = data_dir
        self.out_name = out_name
        self.phase = phase
        self.transforms = transforms
        self.aug = aug
        self.t2 = t2
        self.out_size = out_size
		
        self.image_list = []
        self.label_list = []
        image_list = []
        data_list = self.phase.split(',')
        self.train_flag = train_flag 
        self.dc = {}
        
        for i in range(0,len(data_list),2):
            image_path = join(self.data_dir, data_list[i])
            img_p = [line.strip() for line in open(image_path, 'r')] * int(data_list[i+1])
            
            
            if 'finegrain' in image_path:
                if 'finegrain' not in self.dc.keys():
                    self.dc['finegrain'] = {}
                self.dc['finegrain'][image_path] = img_p
                image_list += img_p * repeat
                
            elif 'pseudo' in image_path:
                if 'pseudo' not in self.dc.keys():
                    self.dc['pseudo'] = {}
                self.dc['pseudo'][image_path] = img_p
                
            elif 'neg' in image_path:
                if 'neg' not in self.dc.keys():
                    self.dc['neg'] = {}
                self.dc['neg'][image_path] = img_p
            
        self.image_list = image_list
        self.initialized = False
        self.num = len(image_list)
        print(self.num, self.dc.keys())
        for k in self.dc.keys():
            print(k)
            for k2 in self.dc[k].keys():
                print(k2)
        

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
	
        rdx = random.randint(0, 100)
        pseudo_flag = False
        neg_flag = False
        if rdx < 40 and 'neg' in self.dc.keys():
            source = random.choice(list(self.dc['neg'].keys()))
            line = random.choice(self.dc['neg'][source])
            im_path,label_path = line.split(',')
            neg_flag = True
        elif rdx > 40 and rdx < 50 and 'pseudo' in self.dc.keys():
            source = random.choice(list(self.dc['pseudo'].keys()))
            line = random.choice(self.dc['pseudo'][source])
            im_path,label_path = line.split(',')
            pseudo_flag = True
        else:
            source = random.choice(list(self.dc['finegrain'].keys()))
            line = random.choice(self.dc['finegrain'][source])
            im_path,label_path = line.split(',')
            
	
        z = np.zeros((1024,1024,3),dtype = np.uint8)
        try:
            img = Image.open(im_path).convert('RGB')
        except:
            img = Image.fromarray(z)
            
        w,h = img.size
		
        data = [img]
        try:
            label_map = Image.open(label_path).convert('L')
        except:
            label_map = Image.fromarray(z).convert('L')
        label_map = np.array(label_map).astype(np.float32)
        
        if pseudo_flag == True:
            thres1 = 255 * 0.01
            V = 4.0
            label_map = label_map * V
            label_map[label_map>255] = 255
            label_map[label_map<thres1] = 0
        elif neg_flag == True:
            label_map = np.zeros((h,w))
        
        label_map = label_map.astype(np.uint8)
        label_map = Image.fromarray(label_map)
        
        data.append(label_map)
        
        if self.train_flag ==  True and self.aug is not None:
            data = list(self.t2(*data))
            augmented = self.aug(image=np.array(data[0]), mask=np.array(data[1]))
            data[0] = Image.fromarray(augmented['image'].astype(np.uint8))
            data[1] = Image.fromarray(augmented['mask'].astype(np.uint8))

        data = list(self.transforms(*data))
        data[1] = data[1].float()/255.0
        if self.out_name:
            data.append(im_path)
        if self.out_size:
            data.append(torch.from_numpy(np.array(image.size, dtype=int)))
        ret = tuple(data)
        return ret 
	
