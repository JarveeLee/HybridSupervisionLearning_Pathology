import os
import numpy as np
import cv2
import openslide
from multiprocessing import Pool

def ostu(img):
    gray = img
    #ret1, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ret1, th1 =  cv2.threshold(gray,204,255,cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    th1 = cv2.morphologyEx(th1.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    th1[th1>0] = 1.0
    th1[img==0] = 0
    return th1

def xml2Array(xml_path):
    seg_ls = []
    rec_ls = []
    with open(xml_path, 'r') as reader:
        content = reader.read()
    pre_start = 0

    gts = []
    while 1:
        ori_start = content[pre_start: ].find('<Annotation ')
        ori_end = content[pre_start: ].find('</Annotation>')

        if ori_start == -1:
            break
        sub_content = content[
            pre_start + ori_start: pre_start + ori_end
        ]
        gt = 0
        for ele in ['PartOfGroup="_0"','PartOfGroup="_1"','PartOfGroup="Tumor"','PartOfGroup="metastases"']:
            fd = sub_content.find(ele)
            if fd !=-1:
                gt = 255
                break
        #print(gt)
        gts.append(gt)
        gt_flag = False
        rec_flag = False
        seg = []
        points = sub_content.strip().split('/>')

        '''
        s = sub_content.find('Confidence="')
        e = sub_content[s + len('Confidence="'): ].find('"')
        confidence = float(
            sub_content[s + len('Confidence="'): s + len('Confidence="') + e]
        )
        start = content[pre_start: ].find('<Coordinates>')
        end = content[pre_start: ].find('</Coordinates>')
        sub_content = content[
            pre_start + start + len('<Coordinate>'): pre_start + end]
        points = sub_content.strip().split('/>')

        if len(points) != 5:
            print(sub_content)
            raise Exception('Error')
        '''
        for i in range(0 , len(points) - 1):
            #print(i,points,len(points),end-start - 1)
            if 'X="' in points[i] and 'Y="' in points[i]:
                s = points[i].find('X="')
                e = points[i][s + 3: ].find('"')
                x = int(float(points[i][s + 3: s + 3 + e]))
                s = points[i].find('Y="')
                e = points[i][s + 3: ].find('"')
                y = int(float(points[i][s + 3: s + 3 + e]))
                if rec_flag:
                    rec.append([x,y])
                else:
                    seg.append([x,y,gt])
        if rec_flag:
            rec_ls.append(rec)
        else:
            seg_ls.append(np.array(seg))        
        pre_start += ori_end + len('</Coordinates>')
    return seg_ls,gts

def iou(box1,box2):
    score = 0
    inter = 0
    x1 = max(box1[0],box2[0])
    y1 = max(box1[1],box2[1])
    x2 = min(box1[2],box2[2])
    y2 = min(box1[3],box2[3])
    w = x2 - x1
    h = y2 - y1
    if w < 0 or h < 0:
        return 0
    return 1

    
def crop_one_tif(seg_arr,tif_path,op_dir,id):
    try:
        os.makedirs(op_dir+'POS/') 
        os.makedirs(op_dir+'NEG/')
    except:
        print(op_dir+id+'/')
        
    print(tif_path)
    tif = openslide.OpenSlide(tif_path)
    fns = []
    
    w,h = tif.level_dimensions[0]
    anno_regions = []
    for ele in seg_arr:
        arr = ele
        xmin = arr[:,0].min()
        xmax = arr[:,0].max()
        ymin = arr[:,1].min()
        ymax = arr[:,1].max()
        anno_regions.append([xmin,ymin,xmax,ymax])
        
    mw,mh = tif.level_dimensions[0]
    sw,sh = tif.level_dimensions[4]

    thumb_im = tif.read_region((0,0),4,(sw,sh)).convert('L')
    #thumb_im.save('tmp.png')
    thumb_msk = ostu(np.array(thumb_im))
    #cv2.imwrite('tmp2.png',thumb_msk*255)
    ratio = sw / mw
    check_ps = round(1024 * ratio)
        
    for x in range(0,w,1024):
        for y in range(0,h,1024):
            ok = 1
            box1 = [x,y,x+1024,y+1024]
            for ele in anno_regions:
                box2 = ele
                if iou(box1,box2) > 0:
                    ok = 0
                    break
            if ok == 0:
                continue
                
            xtiny = int(x * ratio)
            ytiny = int(y * ratio)
            value = thumb_msk[ytiny:ytiny+check_ps,xtiny:xtiny+check_ps].mean()
            if value == 0:
                continue
            #print(value)
            im = tif.read_region((x,y),0,(1024,1024)).convert('RGB')
            sp = op_dir + 'NEG/' + id + '_%d_%d_%d_%d.png'%(x,y,1024,1024)
        
            msk = np.zeros((1024,1024))
            im.save(sp)
            fns.append(sp)
            sp2 = op_dir+ 'NEG/'  + id + '_%d_%d_%d_%d_msk.png'%(x,y,1024,1024)
            cv2.imwrite(sp2,msk)
                    
                    
                
    
    for ele in seg_arr:
        arr = ele
        xmin = arr[:,0].min()
        xmax = arr[:,0].max()
        ymin = arr[:,1].min()
        ymax = arr[:,1].max()
		

			
        w = xmax - xmin
        h = ymax - ymin
        if h == 0 or w == 0:
            continue

        elif h>1024 and w > 1024:
		
            for dx in range(0,w,768):
                for dy in range(0,h,768):
                    im   = tif.read_region(((xmin+dx)*1,(ymin+dy)*1),0,(1024,1024)).convert('RGB')
                    sp = op_dir + 'POS/'  + id + '_%d_%d_%d_%d.png'%(xmin+dx,ymin+dy,1024,1024)
        
		
                    #arr_tmp = arr - np.array([xmin+dx,ymin+dy])
					
                    arr_tmp = seg_arr.copy()
                    for i in range(0,len(arr_tmp)):
                        arr_tmp[i] = arr_tmp[i] - np.array([xmin+dx,ymin+dy,0])
                        arr_tmp[i][arr_tmp[i]<0] = 0
						
                    #print(arr_tmp)
                    msk = np.zeros((1024,1024))

                    for i in range(0,len(arr_tmp)):
                        t = arr_tmp[i] 
                        msk = cv2.fillPoly(msk,[t[:,:2]],int(t[0,2]))
		
                    #if msk.mean()<0.01:
                    #    continue
		
                    sp2 = op_dir + 'POS/' + id + '_%d_%d_%d_%d_msk.png'%(xmin+dx,ymin+dy,1024,1024)
                    cv2.imwrite(sp2,msk)
                    im.save(sp)
                    fns.append(sp)
                    
		
		
        else:
		
		
            if xmax-xmin < 1024:
                cx = (xmin+xmax)//2
                xmin = cx - 512
                xmax = cx + 512
				
            if ymax-ymin<1024:
                cy = (ymin+ymax)//2
                ymin = cy - 512
                ymax = cy + 512
		
            im   = tif.read_region((xmin*1,ymin*1),0,(xmax-xmin,ymax-ymin)).convert('RGB')
            sp = op_dir  + 'POS/' + id + '_%d_%d_%d_%d.png'%(xmin,ymin,xmax-xmin,ymax-ymin)
        
		
            arr_tmp = seg_arr.copy()

            for i in range(0,len(arr_tmp)):
                arr_tmp[i] = arr_tmp[i] - np.array([xmin,ymin,0])
                arr_tmp[i][arr_tmp[i]<0] = 0
            
            msk = np.zeros((ymax-ymin,xmax-xmin))
            for i in range(0,len(arr_tmp)):
                t = arr_tmp[i] 
                msk = cv2.fillPoly(msk,[t[:,:2]],int(t[0,2]))
                #msk = cv2.fillPoly(msk,arr_tmp,255)
		
            #if msk.mean()<0.01:
            #    continue
		
            sp2 = op_dir  + 'POS/' + id + '_%d_%d_%d_%d_msk.png'%(xmin,ymin,xmax-xmin,ymax-ymin)

            cv2.imwrite(sp2,msk)
            im.save(sp)
            fns.append(sp)
    return fns
	
if __name__ == '__main__':

    p = Pool(8) 
    
    xml_dir = 'data/wsi_and_anno/17xmls/'
    save_dir = 'data/patches/tiles17_class3/'
    idx_f = open('data/wsi_and_anno/train_pos_3_wsi.txt').readlines()
    
    for i,fn in enumerate(idx_f):
        fn = fn.strip()
        id = fn.split('/')[-1][:-4]
        xml_pa = xml_dir + id + '.xml'
        print(i,id,xml_pa,fn,os.path.exists(fn),save_dir+id+'/')
        #wsi = AllSlide.AllSlide(fn)
        #print(wsi.properties['tiff.XResolution'],wsi.properties['tiff.ResolutionUnit'])
        if os.path.exists(xml_pa) == False:
            continue
        #ls = sorted(os.listdir('/'.join(fn.split('/')[:-1])))[0]
        #print(
        #if i <= 159:
        #    continue
        #try:
        seg_arr,gts = xml2Array(xml_pa)
        #if 0 in gts:
        #    print(gts)
        #if os.path.exists(save_dir+id+'/'):
        try:
            os.makedirs(save_dir+id+'/POS/') 
            os.makedirs(save_dir+id+'/NEG/')
        except:
            print(save_dir+id+'/')
        if 0 in gts:
            print(gts)
        #crop_one_tif(seg_arr,fn,save_dir+id+'/',id)
        p.apply_async(crop_one_tif,args=(seg_arr,fn,save_dir+id+'/',id,))
        #except:
        #    print('fail')
        
    
    xml_dir = 'data/wsi_and_anno/17xmls/'
    save_dir = 'data/patches/tiles17_class2/'
    idx_f = open('data/wsi_and_anno/train_pos_2_wsi.txt').readlines()
    
    for i,fn in enumerate(idx_f):
        fn = fn.strip()
        id = fn.split('/')[-1][:-4]
        xml_pa = xml_dir + id + '.xml'
        print(i,id,xml_pa,fn,os.path.exists(fn),save_dir+id+'/')
        #wsi = AllSlide.AllSlide(fn)
        #print(wsi.properties['tiff.XResolution'],wsi.properties['tiff.ResolutionUnit'])
        if os.path.exists(xml_pa) == False:
            continue
        #ls = sorted(os.listdir('/'.join(fn.split('/')[:-1])))[0]
        #print(
        #if i <= 159:
        #    continue
        #try:
        seg_arr,gts = xml2Array(xml_pa)
        #if 0 in gts:
        #    print(gts)
        #if os.path.exists(save_dir+id+'/'):
        try:
            os.makedirs(save_dir+id+'/') 
        except:
            print(save_dir+id+'/')
        if 0 in gts:
            print(gts)
        #crop_one_tif(seg_arr,fn,save_dir+id+'/',id)
        p.apply_async(crop_one_tif,args=(seg_arr,fn,save_dir+id+'/',id,))
        #except:
        #    print('fail')
        
        
        
    xml_dir = 'data/wsi_and_anno/17xmls/'
    save_dir = 'data/patches/tiles17_class1/'
    idx_f = open('data/wsi_and_anno/train_pos_1_wsi.txt').readlines()
    
    for i,fn in enumerate(idx_f):
        fn = fn.strip()
        id = fn.split('/')[-1][:-4]
        xml_pa = xml_dir + id + '.xml'
        print(i,id,xml_pa,fn,os.path.exists(fn),save_dir+id+'/')
        #wsi = AllSlide.AllSlide(fn)
        #print(wsi.properties['tiff.XResolution'],wsi.properties['tiff.ResolutionUnit'])
        if os.path.exists(xml_pa) == False:
            continue
        #ls = sorted(os.listdir('/'.join(fn.split('/')[:-1])))[0]
        #print(
        #if i <= 159:
        #    continue
        #try:
        seg_arr,gts = xml2Array(xml_pa)
        #if 0 in gts:
        #    print(gts)
        #if os.path.exists(save_dir+id+'/'):
        try:
            os.makedirs(save_dir+id+'/') 
        except:
            print(save_dir+id+'/')
        if 0 in gts:
            print(gts)
        #crop_one_tif(seg_arr,fn,save_dir+id+'/',id)
        p.apply_async(crop_one_tif,args=(seg_arr,fn,save_dir+id+'/',id,))
        #except:
        #    print('fail')
        
        
    xml_dir = 'data/wsi_and_anno/16xmls/'
    save_dir = 'data/patches/tiles16/'
    idx_f = open('data/wsi_and_anno/pos16wsi.txt').readlines()

    for i,fn in enumerate(idx_f):
        fn = fn.strip()
        id = fn.split('/')[-1][:-4]
        xml_pa = xml_dir + id + '.xml'
        print(i,id,xml_pa,fn,os.path.exists(fn),save_dir+id+'/')
        #wsi = AllSlide.AllSlide(fn)
        #print(wsi.properties['tiff.XResolution'],wsi.properties['tiff.ResolutionUnit'])
        if os.path.exists(xml_pa) == False:
            continue
        #ls = sorted(os.listdir('/'.join(fn.split('/')[:-1])))[0]
        #print(
        #if i <= 159:
        #    continue
        #try:
        seg_arr,gts = xml2Array(xml_pa)
        #if 0 in gts:
        #    print(gts)
        #if os.path.exists(save_dir+id+'/'):
        try:
            os.makedirs(save_dir+id+'/') 
        except:
            print(save_dir+id+'/')
        if 0 in gts:
            print(gts)
        #crop_one_tif(seg_arr,fn,save_dir+id+'/',id)
        p.apply_async(crop_one_tif,args=(seg_arr,fn,save_dir+id+'/',id,))
        #except:
        #    print('fail')
        
    p.close()
    p.join()   
    #f = open('data/train_pos_anno16.txt','w')
    #f = open('data/train_pos_anno17_class1.txt','w')
    #for fn in sorted(os.listdir(save_dir)):
    #    if 'jpg' in fn:
    #        f.write('/mnt/lustre/lijiahui/pathadox_mixsupervision_camelyon/' + save_dir + fn+'\n')
    