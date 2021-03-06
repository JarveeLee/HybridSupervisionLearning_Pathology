from multiprocessing import Pool
import os, time, random
def long_time_task_step4(start_point, step_length):
    ddir = '/'.join(os.path.abspath(__file__).split('/')[:-2])+'/output/'
    model_path = ddir + 'Camelyon_dla34up_bn_init_exp0/Camelyon_dla34up_bn_latest.pth.tar'
    
    
    print('Run task %s (%s)...' % (start_point, os.getpid()))
    order = 'srun --mpi=pmi2 -p MIA --job-name Camelyon_infer -n1 --gres=gpu:1 --ntasks-per-node=1 python -u Stage1_E_step_generate_pseudo_label.py \
	%d %d %s train_pos_1_wsi.txt,train_pos_2_wsi.txt,train_pos_3_wsi.txt,train_neg_wsi.txt,pos16wsi.txt,neg16wsi.txt \
    init_exp0 '%(start_point, step_length,model_path,)
    print(order)
    os.system(order)
    
    # start_point length model_pathN, index_file.txts, round_name
    
if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    step_length = 8
    p = Pool(32)
    for i in range(0,899,step_length):
        #long_time_task_step4(0)
        p.apply_async(long_time_task_step4,args = (i,step_length, ))
    p.close()
    p.join()
