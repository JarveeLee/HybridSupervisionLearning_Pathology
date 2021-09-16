import argparse
import json
import os
print('import multiprocessing')
import multiprocessing
print('imported multiprocessing')
from os.path import exists, join, split, dirname

import time

import numpy as np
import shutil

import sys
from PIL import Image
print('import torch')
import torch
print('imported torch')
import torch.utils.data
from torch import nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable,Function
import torchvision.transforms as ttransforms
import data_transforms as transforms
import dataset,cv2
import sys

import PIL
PIL.Image.MAX_IMAGE_PIXELS = 1e50
from dataloaders import McDataset_seg as SegList
import datetime
print('full import')

from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,    
    CenterCrop,    
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,    
    RandomGamma,
    IAASharpen,
    HueSaturationValue,
    RGBShift,    
)

PALLETE = np.asarray([
    [0, 0, 0],
    [255, 255, 255]], dtype=np.uint8)

mean = [0.6743682882352942, 0.4711362002614377, 0.6204416456209151]
std  = [0.21792378060808987, 0.23067648118994308, 0.18450216349991813]
def channel_fill(img):
    new_img = np.stack([img,img,img],2)
    return new_img

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(
        description='DLA Segmentation and Boundary Prediction')
    parser.add_argument('cmd', choices=['train', 'test'])
    parser.add_argument('--data', default='Camelyon')
    parser.add_argument('--exp', required=True, help='The name of experiment')
    parser.add_argument('--data-dir', default='/'.join(os.path.abspath(__file__).split('/')[:-1])+'/data' ) 
    parser.add_argument('--out-dir', default='output/')
    parser.add_argument('--classes', default=1, type=int)
    parser.add_argument('--crop-size', default=512+256, type=int)
    parser.add_argument('--scale-size', default=2048, type=int)
    parser.add_argument('--step', type=int, default=200)
    parser.add_argument('--arch', type=str, default='dla34up_bn')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--loss', default='l1', type=str)
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--evaluate', dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='- seed (default: 1)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained-base', default='imagenet',
                        help='use pre-trained model')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--down', default=2, type=int, choices=[2, 4, 8, 16],
                        help='Downsampling ratio of IDA network output, which '
                             'is then upsampled to the original resolution '
                             'with bilinear interpolation.')
    parser.add_argument('--lr-mode', default='poly')
    parser.add_argument('--random-scale', default=0.5, type=float)
    parser.add_argument('--random-rotate', default=10, type=int)
    parser.add_argument('--random-color', action='store_true', default=False)
    parser.add_argument('--save-freq', default=5, type=int)
    parser.add_argument('--ms', action='store_true', default=False)
    parser.add_argument('--edge-weight', type=int, default=-1)
    parser.add_argument('--train-repeat', type=int, default=100)
	
    parser.add_argument('--trainfile', default='train', type = str)
    parser.add_argument('--valfile', default='val', type = str)
    parser.add_argument('--optim', default='sgd',type = str)
    parser.add_argument('--gtdir', default='data/',type = str)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--datatype', default='png',type = str)
    parser.add_argument('--spehard', action='store_true', default=False)   
    parser.add_argument('--hard', action='store_true', default=False)
    parser.add_argument('--geno', default=None,type = str)
    parser.add_argument('--clsratio', default=1.0, type=float)
    
    args = parser.parse_args()
    #args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = torch.device("cuda")



    args.out_dir = args.out_dir + args.data + '_' + args.arch + '_' + args.exp + '/'
    time_stamp = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    args.log_file = args.out_dir + args.cmd + '_' + time_stamp + '.log'
    if not exists(args.out_dir): 
        os.makedirs(args.out_dir)
    if not exists(args.out_dir+'/train_viz/'):
        os.makedirs(args.out_dir+'/train_viz/')

    assert args.data_dir is not None
    assert args.classes > 0

    print(' '.join(sys.argv))
    print(args)
    with open(args.log_file, 'w') as fp:
        fp.write(str(args) + '\n\n')

    return args

args = parse_args()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# def accuracy(output, target):
#     """Computes the precision@k for the specified values of k"""
#     # batch_size = target.size(0) * target.size(1) * target.size(2)
#     _, pred = output.max(1)
#     pred = pred.view(1, -1)
#     target = target.view(1, -1)
#     correct = pred.eq(target)
#     correct = correct[target != 255]
#     correct = correct.view(-1)
#     score = correct.float().sum(0).mul(100.0 / correct.size(0))
#     return score.item()


def train(args, train_loader, val_loader, model, criterion, optimizer, start_epoch, end_epoch , print_freq=50 , val_freq = 1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    hist = np.zeros((2, 2))
    # switch to train mode
    model.train()
    best_prec1 = 0.0
    loader_length = int(len(train_loader) / (end_epoch - start_epoch))

    end = time.time()
    for i, (input, target,names) in enumerate(train_loader):
	
        if i % loader_length == 0:
            epoch_this = int(start_epoch + i//loader_length)
            lg = str(datetime.datetime.now()) + ' Epoch: [%d/%d] %d \n'%(epoch_this,end_epoch,loader_length)
            print(lg)
            with open(args.log_file, 'a') as fp:
                fp.write(lg)

        # measure data loading time
        data_time.update(time.time() - end)

        # pdb.set_trace()

        if type(criterion) in [torch.nn.modules.loss.L1Loss,
                               torch.nn.modules.loss.MSELoss]:
            target = target.float()

        input = input.to(args.device)
        target = target.to(args.device).unsqueeze(1).float()

        output = model(input)
        loss_seg = criterion(output, target)
        loss = loss_seg

        conf, pred = torch.max(output, 1)
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        losses.update(loss.item(), input.size(0))
        #print('after hist ',pred.size(),output.size(),target.size(),target.max(),target.min())
        if i % print_freq == 0 or i == len(train_loader) -1 :
            sss = 'Epoch: [{0}][{1}/{2}] Time {batch_time.avg:.3f} Data {data_time.avg:.3f} Loss {loss.avg:.4f} '.format( epoch_this, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses)
            sss = str(datetime.datetime.now()) + ' ' + sss  
            print(sss)

            with open(args.log_file, 'a') as fp:
                fp.write(sss+'\n')

        if i % 200 == 0 or i == len(train_loader) -1 :
            os.system('rm -rf ' + args.out_dir+'/train_viz/')
            os.system('mkdir '+ args.out_dir+'/train_viz/')
            output = output.sigmoid()
            for j in range(0,input.size()[0]):
                img   = input[j,:,:,:].data.cpu().numpy().transpose(1,2,0)*255
                gt    = target[j,0,:,:].data.cpu().numpy()*255
                #print(np.unique(gt),np.unique(target[j,1,:,:].data.cpu().numpy()),target[j,1,:,:].max(),target[j,0,:,:].max())
                segmp = output[j,0,:,:].data.cpu().numpy()*255
                viz   = np.concatenate([img,channel_fill(gt),channel_fill(segmp)],1)
                viz   = Image.fromarray(viz.astype(np.uint8))
                path  = args.out_dir+'train_viz/%d.jpg'%(j)
                viz.save(path)

        if epoch_this % val_freq == 0 and i % loader_length == 0 and i > 0:
            #prec1 = validate(args, val_loader, model, criterion)
            prec1 = 0
            best_prec1 = max(prec1, best_prec1)

            print('===> Prec1 vs Best: %.2f vs %.2f' % (prec1, best_prec1))
            with open(os.path.join(args.log_file), 'a') as fp:
                fp.write('\n===> Prec1 vs Best: %.2f vs %.2f\n' %(prec1, best_prec1))

            checkpoint_path = args.out_dir + args.data + '_' + args.arch + '_latest.pth.tar'
            torch.save({
                'epoch': epoch_this,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, checkpoint_path)

            history_path = args.out_dir + args.data + '_' + args.arch + '_{:03d}.pth.tar'.format(epoch_this )
            shutil.copyfile(checkpoint_path, history_path)
            losses = AverageMeter()
            hist = np.zeros((2, 2))
        


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'checkpoint_best.pth.tar')

                         


def train_seg(args):

    if args.arch == 'resnet34':
        import resnet 
        model = resnet.Res_DNet(classes = args.classes)
    elif args.arch == 'mobilenet':
        import light.model.mobilenetv3_seg as mbs
        model = mbs.get_mobilenet_v3_small_seg_xiaobiaoben(classes = 2)
		
    elif args.arch == 'unet':
        import unet
        model = unet.UNet1024(co = args.classes)
		
    elif args.arch == 'dense34':
        import densenet
        model = densenet.densenet34_seg(num_classes = args.classes)
		
    elif args.arch == 'deeplab':
        import deeplab
        model = deeplab.resnet50(num_classes = args.classes)
		
    else:
        import dla_up_bn as dla_up
        pretrained_base = args.pretrained_base
        model = dla_up.__dict__.get(args.arch)(args.classes, pretrained_base, down_ratio=args.down)

    model = torch.nn.DataParallel(model).to(args.device)

    print(' '.join(sys.argv))

    for k, v in args.__dict__.items():
        print(k, ':', v)

    criterion = torch.nn.BCEWithLogitsLoss()
    criterion.to(args.device)
    data_dir = args.data_dir
    #info = dataset.load_dataset_info('info.json')
    #normalize = transforms.Normalize(mean=info.mean, std=info.std)
    #t = [transforms.Resize(args.scale_size)]
    t = []
	
    '''
    t.append(transforms.RandomCrop(args.crop_size))
    if args.random_rotate > 0:
        t.append(transforms.RandomRotate(args.random_rotate))
    if args.random_scale > 0:
        # t.append(transforms.RandomScale(args.random_scale))
        t.append(transforms.RandomScale(
            [1 - args.random_scale, 1 + args.random_scale]))
    t.append(transforms.RandomCrop(args.crop_size))
    if args.random_color:
        t.append(transforms.RandomJitter(0.4, 0.4, 0.4))
              transforms.RandomHorizontalFlip(),
              transforms.RandomVerticalFlip()
    
    '''
    t.extend([transforms.RandomJitter(0.4, 0.4, 0.4),transforms.ToTensor()])
	
    t2 = transforms.Compose([transforms.RandomCrop(1024)])
    aug = Compose([RandomSizedCrop(min_max_height=(args.crop_size*(1-args.random_scale),  1024 ), height=args.crop_size, width=args.crop_size, p=1.0),  
    VerticalFlip(p=0.5),              
    RandomRotate90(p=0.5),HorizontalFlip(p=0.5),
    RandomBrightnessContrast(p=0.8),IAASharpen(p=0.8),HueSaturationValue(p=0.5),RGBShift(p=0.5)])

    p_to_optim = []
    for p in model.parameters():
        if p.requires_grad:
            p_to_optim.append(p)
    if args.optim == 'adam':
        print('adam')
        optimizer = torch.optim.Adam(p_to_optim,lr=args.lr)
    else:
        optimizer = torch.optim.SGD(p_to_optim,args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    cudnn.benchmark = True
    best_prec1 = 0
    start_epoch = 0
    #args.resume = 'output/Camelyon_dla34up_bn_init_exp0/Camelyon_dla34up_bn_latest.pth.tar'
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            #start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            dc = checkpoint['state_dict']
            #del dc['module.fc.0.weight']
            #del dc['module.fc.0.bias']
            model.load_state_dict(dc)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    epochs = args.epochs - start_epoch

    if args.test == False:
        train_loader = torch.utils.data.DataLoader(
            SegList(data_dir, args.trainfile, transforms.Compose(t),
                    binary=(args.classes == 2), repeat=epochs * args.train_repeat,args = args, aug = aug, t2 = t2, train_flag = True,out_name = True),
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
            pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            SegList(data_dir, args.valfile, transforms.Compose([
                transforms.RandomCrop(args.crop_size),
                transforms.ToTensor(),
            ]), binary=(args.classes == 2),args = args),
            batch_size=1, shuffle=False, num_workers=args.workers,
            pin_memory=True
        )
    else:
        test_loader = torch.utils.data.DataLoader(
            SegList(data_dir, args.valfile, transforms.Compose([
                transforms.ToTensor(),
            ]), out_name = True, binary=(args.classes == 2),args = args),
            batch_size=1, shuffle=False, num_workers=args.workers,
            pin_memory=True
        )

    train(args, train_loader, val_loader, model, criterion, optimizer ,start_epoch, args.epochs )



def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    v = np.bincount(n * label[k].astype(int) + pred[k], minlength=n ** 2)
    #print(pred.shape,label.shape,n,v.shape)
    return v.reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def main():
    torch.set_num_threads(multiprocessing.cpu_count())
    args = parse_args()
    
    if args.cmd == 'train':
        train_seg(args)
    elif args.cmd == 'test':
        test_seg(args)


if __name__ == '__main__':
    main()
