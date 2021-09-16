# given one model 
import data_transforms as transforms
import torch
import torch.utils.data
import torch.nn as nn
import cv2,os
import numpy as np
import datetime,pickle
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
from PIL import Image
import sys
import matplotlib.pyplot as plt
import PIL.Image as Image
from skimage import measure
import cv2
import random
import dla_up_bn as dla_up
import time
import openslide

def ostu(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #ret1, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ret1, th1 =  cv2.threshold(gray,204,255,cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    th1 = cv2.morphologyEx(th1.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    th1[th1>0] = 1.0
    th1[img[:,:,0]==0] = 0
    return th1




class loader(torch.utils.data.Dataset):
    def __init__(self, wsi):
        self.wsi = wsi
        self.read_patchsize = 1024
        #oganize data list--------------------------------------------------------------------
        mw,mh = wsi.level_dimensions[0]
        sw,sh = wsi.level_dimensions[4]
        self.patchlist = []

        thumb_im = wsi.read_region((0,0),4,(sw,sh))
        
        #get foreground msk here--------------------------------------------------------------------
        thumb_msk = ostu(np.array(thumb_im))
        check_ps = round(self.read_patchsize * sw / mw)

        # overlap patch or not --------------------------------------------------------------------
        for x in range(0,sw - check_ps ,check_ps *  5//6): 
            for y in range(0,sh - check_ps ,check_ps * 5//6):
                x1 = x 
                x2 = x + check_ps
                y1 = y
                y2 = y + check_ps
                tmp_msk = thumb_msk[y1:y2,x1:x2]
                if tmp_msk.mean()>0:
                    # add more type here --------------------------------------------------------------------
                    xx1 = int(x1 * mw / sw)
                    yy1 = int(y1 * mw / sw)
                    ww  = self.read_patchsize
                    hh  = self.read_patchsize

                    self.patchlist.append([xx1,yy1,0,ww,hh]) 

        print(len(self.patchlist),check_ps,sw,sh,self.patchlist[0])   
        

    def __getitem__(self, index):
        x,y,lv,w,h = self.patchlist[index]
        try:
            im = self.wsi.read_region((x,y),lv,(w,h)).convert('RGB')
        except:
            im = Image.fromarray(np.zeros((h,w,3),dtype = np.uint8))
        im_tensor = torch.from_numpy(np.array(im).transpose(2,0,1)).float().div(255)
        data_final = [im_tensor,x,y,lv,w,h]
        return tuple(data_final)

    def __len__(self):
        return len(self.patchlist)

def allslide_seg(model_list ,wsi_path, cache_dir):
    
    wsi = openslide.OpenSlide(wsi_path)
    #oganize the loader--------------------------------------------------------------------
    dataloader = loader(wsi)
    inference_loader = torch.utils.data.DataLoader(
            dataloader,
            batch_size=16, shuffle=False, num_workers=4,
            pin_memory=True
        )
    infer_t = 0

    for iter, (image,x,y,lv,w,h) in enumerate(inference_loader):

        if iter % 10 == 0:
            print(str(datetime.datetime.now()),iter,len(inference_loader))
        image_var = image.to(device)
        B,C,H,W = image_var.size()
        pred_list = []
        feature_list = []
        #print(image_var)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            for i in range(0,len(model_list)):
                model = model_list[i]
                prob_map = model(image_var).sigmoid().squeeze(1) 
                B,H,W = prob_map.size()
                pred_list.append(prob_map)
            
        torch.cuda.synchronize()
        infer_t += time.time() - t0
        # seg ensemble -----------------------------------------------------------------------------    
        t0 = time.time()		
        preds = torch.stack(pred_list,1) 
        pesudo  = preds.mean(1).cpu().data.numpy()
        t0 = time.time()
        # batch ele extract contour ----------------------------------------------------------
        for i in range(B):
            v = float(pesudo[i,...].max())
            if v < 0.4 :
                continue
            wsi_name = wsi_path.split('/')[-1]
            sp = cache_dir + '%s_%d_%d_%d_%d_%d.png'%(wsi_name,int(x[i]),int(y[i]),int(lv[i]),int(w[i]),int(h[i]))
            ddir = '/'.join(sp.split('/')[:-1])	
            try:
                os.makedirs(ddir)
            except:
                pass
            msk = pesudo[i,...]
            msk = (msk*255).astype(np.uint8)
            im_np = np.array(image[i,:,:,:]).transpose(1,2,0)*255
            im_np = im_np.astype(np.uint8)
            im    = Image.fromarray(im_np)
            im.save(sp)
            sp = sp.replace('.png','_msk.png')
            im = Image.fromarray(msk)
            im.save(sp)

if __name__ == '__main__':

    ddir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
    model_path_list = sys.argv[3].split(',')
    
    device = torch.device('cuda')
    model_list = []
    for i in range(0,len(model_path_list)):
        seg_path = model_path_list[i]
        print(seg_path)
        model = dla_up.__dict__.get('dla34up_bn')(1,None,down_ratio=2).eval()
        model = torch.nn.DataParallel(model).to(device)
        checkpoint = torch.load(seg_path)
        model.load_state_dict(checkpoint['state_dict'])
        model_list.append(model)


    wsi_list = []
    wsi_index = sys.argv[4].split(',')

    flg = -1
    kfb_ids = []
    for idx_path in wsi_index:
        idx_path = ddir +'/data/wsi_and_anno/' + idx_path
        lines = [line.strip('\n') for line in open(idx_path,'r')]
        lines_group = []
        for line in lines:
            lines_group.append([line,idx_path.split('.')[0].split('/')[-1]])
        wsi_list += lines_group
     
    sp0 = int(sys.argv[1])
    lll = int(sys.argv[2])
    
    midx = sp0
    madx = min( sp0 + lll , len(wsi_list) ) 
    o_l  = len(wsi_list)
    
    if madx < len(wsi_list):
        wsi_list_cut = wsi_list[midx : madx]
        print(wsi_list_cut,len(wsi_list_cut),len(wsi_list))
        cnt = 0
        for wsi_path, idx_name in wsi_list_cut:
            sss = sys.argv[5]
            wsi_name = wsi_path.split('/')[-1].split('.')[0]
            cache_dir = ddir +'/data/patches/' + sss + '/%s/'%(idx_name) + wsi_name + '/'
            allslide_seg(model_list ,wsi_path, cache_dir)