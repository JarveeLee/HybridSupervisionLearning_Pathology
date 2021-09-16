import os,cv2
import numpy as np
from multiprocessing import Pool
	
def deal_one_wsi(wsi_p):
    print(wsi_p)
    ratio = 32
    coords = []
    for im_p in os.listdir(wsi_p):
        if im_p.endswith('.png'):
            x,y = im_p.split('_')[-5:-3]
            x = int(x)
            y = int(y)
            coords.append([x,y])
        
    coords = np.array(coords)
    xmin = coords[:,0].min()
    ymin = coords[:,1].min()
    w = coords[:,0].max() - xmin + 2048
    h = coords[:,1].max() - ymin + 2048
    
    mp = np.zeros((h//ratio,2 * w//ratio,3))
    ls = os.listdir(wsi_p)
    for i,im_p in enumerate(ls):
        if i % 100 == 0:
            print(i,len(ls),wsi_p)
        if im_p.endswith('_msk.png'):
            x,y,lv,d = im_p.split('_')[4:8]
        else:
            x,y,lv,d = im_p.split('_')[4:8] 
        x = int(x)-xmin
        y = int(y)-ymin
        d = int(d)
        #print(wsi_p+'/'+im_p)
        im = cv2.imread(wsi_p+'/'+im_p)
        if im is None:
            print(wsi_p+'/'+im_p)
            continue

        im = cv2.resize(im,(d//ratio,d//ratio))  
        #print(i,len(os.listdir(wsi_p)),y//ratio,(y+d)//ratio,x//ratio,(x+d)//ratio,im.shape,mp[y//ratio:(y+d)//ratio,x//ratio:(x+d)//ratio].shape,h//32,w//32)
        if im_p.endswith('_msk.png'):
            mp[y//ratio:(y+d)//ratio,w//ratio + x//ratio:w//ratio + (x+d)//ratio,:] = im
        else:
            mp[y//ratio:(y+d)//ratio,x//ratio:(x+d)//ratio,:] = im
    sp = wsi_p+'.png'    
    cv2.imwrite(sp,mp)
	

if __name__ == '__main__':

    p = Pool(8) 
    root_dirs = [
        'data/patches/round3_exp0/train_pos_3_wsi/',
        'data/patches/round3_exp0/train_pos_2_wsi/',
        'data/patches/round3_exp0/train_pos_1_wsi/',
        'data/patches/round3_exp0/train_neg_wsi/',
        'data/patches/round3_exp0/test_wsi/',
    ]
        
    for root_dir in root_dirs:
        for fn in sorted(os.listdir(root_dir)):
            #print(fn)
            if fn.endswith('.png'):
                continue
            wsi_p = root_dir + fn
            if os.path.exists(wsi_p+'.png'):
                print('skip ',wsi_p)
                continue							
            p.apply_async(deal_one_wsi,args = (wsi_p,))
            #deal_one_wsi(wsi_p)
    p.close()
    p.join()