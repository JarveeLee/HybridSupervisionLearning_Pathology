from multiprocessing import Pool
import os, time, random
def long_time_task_step4(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    order = 'srun --mpi=pmi2 -p MIA -n1 --gres=gpu:4 --job-name Camelyon_train --ntasks-per-node=1 python -u Stage1_M_step_retrain.py train \
	--exp round1_exp0 --out-dir output/ --random-scale 0.5 --random-rotate 360 --random-color \
	--train-repeat 1 --save-freq 5 --epochs 3 --batch-size 32 --classes 1 \
	--workers 4 --resume output/Camelyon_dla34up_bn_init_exp0/Camelyon_dla34up_bn_latest.pth.tar \
	--lr 1e-3 \
	--trainfile tiles16_POS_finegrain.txt,1,tiles16_NEG_finegrain.txt,1,tiles17_class1_NEG_finegrain.txt,1,tiles17_class1_POS_finegrain.txt,1,tiles17_class2_NEG_finegrain.txt,1,tiles17_class2_POS_finegrain.txt,1,tiles17_class3_NEG_finegrain.txt,1,tiles17_class3_POS_finegrain.txt,1,neg16wsi_init_exp0.txt,1,pos16wsi_init_exp0_pseudo.txt,1,train_neg_wsi_init_exp0.txt,1,train_pos_1_wsi_init_exp0_pseudo.txt,1,train_pos_2_wsi_init_exp0_pseudo.txt,1,train_pos_3_wsi_init_exp0_pseudo.txt,1 \
	--optim adam \
	--valfile tiles16_POS_finegrain.txt,1 '
    print(order)
    os.system(order)
    
if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    long_time_task_step4(0)
