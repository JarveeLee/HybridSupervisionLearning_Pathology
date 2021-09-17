import os
import numpy as np
import sklearn.metrics
import sklearn
from sklearn.ensemble import RandomForestClassifier
import cv2
import numpy as np
import scipy.ndimage as ndimage
from skimage import measure
import random

def feature_extract(heatmap):

    fs = []
    # 1. max valule of heatmap
    feature_max_value = np.max(heatmap)
    fs.append(feature_max_value)
    # 2.
    #   <1> threshold 0.5
    
    for i in [0.9]:
        f_tp = _feature_extract_thre(heatmap, threshold=int(i*255)) 
        fs += f_tp
    
    
    return np.array(fs)



def _feature_extract_thre(heatmap, threshold):
    bin_heatmap = heatmap.copy()
    bin_heatmap[bin_heatmap>threshold] = 255
    bin_heatmap[bin_heatmap<=threshold] = 0
    all_area = np.sum(bin_heatmap) // 255
    label_heatmap =  measure.label(bin_heatmap, connectivity=2)
    # feature
    area = 0
    extent = eccentricity = major_axis_length = mean_intensity = solidity = perimeter = -1.
    for region in measure.regionprops(label_heatmap):
        if region.label == 0:
            continue
        if region.area > area:
            area = region.area
            extent = region.extent
            eccentricity = region.eccentricity
            major_axis_length = region.major_axis_length
            mean_intensity = np.mean(heatmap[label_heatmap == region.label])
            solidity = region.solidity
            perimeter = region.perimeter

    #assert area != -1
    #return all_area, area, extent, eccentricity, major_axis_length, mean_intensity, solidity, perimeter
    return major_axis_length,perimeter,area

def node2patient(a):
    while(len(a)<5):
        a.append(0)
    #print(len(a))
    a = np.array(a)
    if a.sum() == 0:
        return 0
    elif a.max() == 1:
        return 1
    elif a.max() == 2:
        return 2
    elif a.max() == 3 and (a>=2).sum() <=3 and (a>=2).sum() >= 1:
        return 3
    else:
        return 4

    
def gather_submission(ypred2, name_list):

    tif = {}
    tif[0] = 'negative'
    tif[1] = 'itc'
    tif[2] = 'micro'
    tif[3] = 'macro'    
    
    sas = {}
    sas[0]='pN0'
    sas[1]='pN0(i+)'
    sas[2]='pN1mi'
    sas[3]='pN1'
    sas[4]='pN2'
    
    dc = {}
    for i in range(0,len(ypred2)):
        p = ypred2[i]
        n = name_list[i]
        dc[n] = p
    f = open('submission.csv','w')
    f.write('patient,stage\n')
    

    for i in range(100,200):
        a = []
        for j in range(0,5):
            ln = 'patient_%3d_node_%d.tif'%(i,j)
            if ln in dc.keys():
                a.append(dc[ln])
            else:
                dc[ln] = 0
                a.append(dc[ln])
            
        cls = node2patient(a)
        cls = sas[cls]
        ln = 'patient_%3d.zip,%s\n'%(i,cls)
        f.write(ln)
        
        for j in range(0,5):
            ln = 'patient_%3d_node_%d.tif'%(i,j)
            f.write('%s,%s\n'%(ln,tif[dc[ln]]))
            

    

if __name__ == '__main__':

    ddir_parent = 'data/patches/round3_exp0/'
    classes = ['train_neg_wsi','train_pos_1_wsi','train_pos_2_wsi','train_pos_3_wsi']
    lls = []
    name_list = []
    f_list = []
    gt_list = []
    for c,ele in enumerate(classes):
        ddir = ddir_parent + ele
        for fn in os.listdir(ddir):
            if fn.endswith('.png'):
                img_path = ddir + '/' + fn
                #print(img_path)
                feature_path = img_path.replace('.png','.txt')
                gt_this = c
                if os.path.exists(feature_path):
                    feature = np.loadtxt(feature_path)
                else:
                    msk = cv2.imread(img_path,0)
                    h,w = msk.shape
                    feature = feature_extract(msk[:,w//2:])
                    np.savetxt(feature_path,feature)
                
                name_list.append(fn.replace('.png','.tif'))
                f_list.append(feature)
                gt_list.append(gt_this)
                
    
    clfs = []
    indexes = list(range(0,len(f_list)))
    random.shuffle(indexes)
    div = len(f_list) // 5
    indexes_fold = []
    for i in range(0,5):
        if i != 4:
            index_this  = indexes[ i * div : (i+1) * div ]
        else:
            index_this  = indexes[ i * div :  ]
        indexes_fold.append(np.array(index_this).astype(np.int32))
        
    f_list = np.array(f_list)
    gt_list = np.array(gt_list)
    ys = []
    ypreds = []
    for i in range(0,5):
        clf = RandomForestClassifier()
        Xtrain = []
        Ytrain = []
        Xtest = []
        Ytest = []
        for j in range(0,5):
            index_this = indexes_fold[j]
            if j == i:
                Xtest.extend(f_list[index_this])
                Ytest.extend(gt_list[index_this])
            else:
                Xtrain.extend(f_list[index_this])
                Ytrain.extend(gt_list[index_this])
                
               
        Xtrain = np.array(Xtrain)
        Ytrain = np.array(Ytrain) 
        Xtest = np.array(Xtest)
        Ytest = np.array(Ytest) 
        clf.fit(Xtrain, Ytrain)
        ypred = clf.predict(Xtest)
        ypreds.extend(list(ypred))
        ys.extend(list(Ytest))
        clfs.append(clf)
        
    kappa = sklearn.metrics.cohen_kappa_score(y1=ypreds, y2=ys, weights='quadratic')
    print('kappa of train 5fold in wsi level, not patient level ', kappa)

    
    
    
    ddir_parent = 'data/patches/round3_exp0/'
    classes = ['test_wsi']
    lls = []
    name_list = []
    f_list = []
    gt_list = []
    for c,ele in enumerate(classes):
        ddir = ddir_parent + ele
        for fn in os.listdir(ddir):
            if fn.endswith('.png'):
                img_path = ddir + '/' + fn
                #print(img_path)
                feature_path = img_path.replace('.png','.txt')
                gt_this = c
                if os.path.exists(feature_path):
                    feature = np.loadtxt(feature_path)
                else:
                    msk = cv2.imread(img_path,0)
                    h,w = msk.shape
                    feature = feature_extract(msk[:,w//2:])
                    np.savetxt(feature_path,feature)
                
                name_list.append(fn.replace('.png','.tif'))
                f_list.append(feature)
                gt_list.append(gt_this)
                
    
    Xtest = np.array(f_list)
    ypreds = []
    for i in range(0,5):
        clf = clfs[i]
        ypred_this = clf.predict_proba(Xtest)
        ypreds.append(ypred_this)
    ypreds = np.stack(ypreds,2).mean(2).argmax(1)
    
    gather_submission(ypreds, name_list)