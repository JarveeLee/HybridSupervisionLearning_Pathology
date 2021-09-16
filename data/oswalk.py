import os


k = 'round2_exp0'
ddir = '/'.join(os.path.abspath(__file__).split('/')[:-1])+'/patches/%s'%(k) 
for root, dirs, files in os.walk(ddir, topdown=False):
    for name in files:
        if name.endswith('_msk.png'):
            continue
        pa = os.path.join(root, name)
        if 'neg' in pa:
            f_n = pa.split('/')[-3]+'_%s.txt'%(k) 
        elif 'pos' in pa:
            f_n = pa.split('/')[-3]+'_%s_pseudo.txt'%(k) 
        elif 'POS' in pa:
            f_n = pa.split('/')[-4]+'_POS_finegrain.txt'
        elif 'NEG' in pa:
            f_n = pa.split('/')[-4]+'_NEG_finegrain.txt'
        f = open(f_n,'a')
        f.write(pa+','+pa.replace('.png','_msk.png'+'\n'))