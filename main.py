import argparse
import os
import time
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.autograd import Variable


import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils

from models import *
from splitted_cifar100 import CIFAR100
from tensorboardX import SummaryWriter
from datetime import datetime
import logutil

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('-ct', '--cifar-type', default='100', type=int, metavar='CT', help='10 for cifar10,100 for cifar100 (default: 10)')
parser.add_argument('--jobs_dir', default='jobs/', type=str,  help='set if you want to use exp time')
parser.add_argument('--exp_name', default=None, type=str,  help='set if you want to use exp name')

args = parser.parse_args()

best_prec = 0

# writer = SummaryWriter(log_dir='log', comment='resNeXt_cifar100')
current_time    = datetime.now()
exp_time        = current_time.strftime('%Y-%m-%d_%Hh%Mm')
exp_name        = '_' + args.exp_name if args.exp_name is not None else ''
jobs_dir        = os.path.join( args.jobs_dir, exp_time + exp_name )
args.exp_time   = exp_time
if not os.path.exists(jobs_dir):    os.makedirs(jobs_dir)

logger          = logutil.getLogger()
logutil.set_output_file( os.path.join(jobs_dir, 'log_%s.txt' % exp_time) )
logutil.logging_run_info( vars(args) )

# if not os.path.exists('runs'):    os.makedirs('runs')
# writer          = SummaryWriter(log_dir= os.path.join('runs', exp_time + exp_name))
if not os.path.exists( os.path.join(jobs_dir, 'tensorboardX') ):    os.makedirs( os.path.join(jobs_dir, 'tensorboardX') )
writer          = SummaryWriter(log_dir= os.path.join(jobs_dir, 'tensorboardX'))

def print(msg):
    logger.info(msg)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    global best_prec    
    use_gpu = torch.cuda.is_available()

    # Model building
    print('=> Building model...')
    if use_gpu:
        # model can be set to anyone that I have defined in models folder
        # note the model should match to the cifar type !

        # model = resnet20_cifar()
        # model = resnet32_cifar()
        # model = resnet44_cifar()
        # model = resnet110_cifar()
        # model = preact_resnet110_cifar()
        # model = resnet164_cifar(num_classes=100)
        # model = resnet1001_cifar(num_classes=100)
        # model = preact_resnet164_cifar(num_classes=100)
        # model = preact_resnet1001_cifar(num_classes=100)

        # model = wide_resnet_cifar(depth=20, width=10, num_classes=100)

        model = resneXt_cifar(depth=29, cardinality=2, baseWidth=64, num_classes=100)
        
        print('Number of trainable parameters in the model: {:,d}'.format(
            count_parameters(model)) )
        # model = densenet_BC_cifar(depth=190, k=40, num_classes=100)

        # mkdir a new folder to store the checkpoint and best model
        # if not os.path.exists('result'):
        #     os.makedirs('result')
        # fdir = 'result/resnext_cifar100'
        # fdir = 'result/wide_resnet_20_10_cifar100'
        # if not os.path.exists(fdir):
        #     os.makedirs(fdir)

        # # adjust the lr according to the model type
        # if isinstance(model, (ResNet_Cifar, PreAct_ResNet_Cifar)):
        #     model_type = 1
        # elif isinstance(model, Wide_ResNet_Cifar):
        #     model_type = 2
        # elif isinstance(model, (ResNeXt_Cifar, DenseNet_Cifar)):
        #     model_type = 3
        # else:
        #     print('model type unrecognized...')
        #     return
        # model_type = 3
        
        # Updating for pytorch >= 0.4, https://github.com/lanpa/tensorboard-pytorch/pull/106
        dummy = Variable( torch.randn( (args.batch_size, 1, 32, 32) ).cuda() )
        dummy2 = Variable( torch.randn( (args.batch_size, 1, 32, 32) ).cuda() )
        writer.add_graph( model.cuda(), (dummy, dummy2, ), {'R', 'B'})

        model = nn.DataParallel(model).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.SGD(model.parameters(), args.lr, 
            momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
        cudnn.benchmark = True
    else:
        print('Cuda is not available!')
        return

    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint "{}"'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec = checkpoint['best_prec']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading and preprocessing
    # CIFAR10
    if args.cifar_type == 10:
        print('=> loading cifar10 data...')
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', 
            train=True, 
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        test_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)
    # CIFAR100
    else:
        print('=> loading cifar100 data...')
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            
        # train_dataset = torchvision.datasets.CIFAR100(
        train_dataset = CIFAR100(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),                
                transforms.ToTensor(),
                normalize,
            ]))
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
                
        # test_dataset = torchvision.datasets.CIFAR100(
        test_dataset = CIFAR100(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([                
                transforms.ToTensor(),
                normalize,
            ]))
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=50, shuffle=False, num_workers=2)

    if args.evaluate:
        validate(testloader, model, criterion)
        return

    # model type 3
    # WRN
    # milestones = [ 60, 120, 160 ]  
    # optim_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)

    milestones = [ 150, 225 ]    
    optim_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    print('Milestones for LR schedulring: {}'.format(milestones))


    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch, model_type)
        optim_scheduler.step()

        # train for one epoch
        prec_, losses_ = train(trainloader, model, criterion, optimizer, epoch, optim_scheduler)        

        # evaluate on test set
        prec, losses = validate(testloader, model, criterion)

        writer.add_scalars('top1-prec', {'train': prec_, 'val': prec}, epoch )
        writer.add_scalars('loss', {'train': losses_, 'val': losses}, epoch )
        writer.add_scalar('lr', optim_scheduler.get_lr()[0], epoch )

        # remember best precision and save checkpoint
        is_best = prec > best_prec
        best_prec = max(prec,best_prec)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optimizer': optimizer.state_dict(),
        }, is_best, jobs_dir)


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


def train(trainloader, model, criterion, optimizer, epoch, optim_scheduler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    end = time.time()
    # for i, (input, target) in enumerate(trainloader):
    for i, (B, G, R, target) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        # input, target = input.cuda(), target.cuda()
        B, G, R, target = B.cuda(), G.cuda(), R.cuda(), target.cuda()
        # input_var = Variable(input)
        # target_var = Variable(target)        

        # compute output
        # output = model(input)

        # output = model(R)
        output = model(R, B)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec = accuracy(output.data, target)[0]
        losses.update(loss.item(), R.size(0))
        top1.update(prec.item(), R.size(0))

        # losses.update(loss.data[0], input.size(0))
        # losses.update(loss.item(), input.size(0))
        # top1.update(prec[0], input.size(0))
        # top1.update(prec.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            for name, param in model.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), i)

            writer.add_scalar('loss/train', loss.item(), i+(epoch-1)*len(trainloader) )            

            print('Epoch: [{:3d}][{:3d}/{:3d}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:2.4f} ({loss.avg:2.4f})\t'
                  'Prec {top1.val:3.2f}% ({top1.avg:3.2f}%)\t'
                  'LR {learning_rate:.4f}'.format(
                   epoch, i, len(trainloader), learning_rate=optim_scheduler.get_lr()[0],
                   batch_time=batch_time, data_time=data_time, loss=losses, top1=top1))

        if i % args.print_freq*10 == 0:
            x = vutils.make_grid(R.cpu()[:4,...], normalize=True, scale_each=True)
            writer.add_image('R/train', x, i)            
            x = vutils.make_grid(G.cpu()[:4,...], normalize=True, scale_each=True)
            writer.add_image('G/train', x, i)            
            x = vutils.make_grid(B.cpu()[:4,...], normalize=True, scale_each=True)
            writer.add_image('B/train', x, i)

    return top1.avg, losses.avg


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    with torch.no_grad():
        # for i, (input, target) in enumerate(val_loader):        
        for i, (B, G, R, target) in enumerate(val_loader):        
            # input, target = input.cuda(), target.cuda()
            B, G, R, target = B.cuda(), G.cuda(), R.cuda(), target.cuda()

            # input_var = Variable(input, volatile=True)
            # target_var = Variable(target, volatile=True)

            # compute output
            # output = model(input)
            # output = model(R)
            output = model(R, B)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec = accuracy(output.data, target)[0]
            losses.update(loss.item(), R.size(0))
            top1.update(prec.item(), R.size(0))

            # losses.update(loss.data[0], input.size(0))
            # top1.update(prec[0], input.size(0))
            # losses.update(loss.item(), input.size(0))
            # top1.update(prec.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{:3d}/{:3d}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:2.4f} ({loss.avg:2.4f})\t'
                      'Prec {top1.val:3.2f}% ({top1.avg:3.2f}%)'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1))

            if i % args.print_freq*10 == 0:
                x = vutils.make_grid(R.cpu()[:4,...], normalize=True, scale_each=True)
                writer.add_image('R/val', x, i)            
                x = vutils.make_grid(G.cpu()[:4,...], normalize=True, scale_each=True)
                writer.add_image('G/val', x, i)            
                x = vutils.make_grid(B.cpu()[:4,...], normalize=True, scale_each=True)
                writer.add_image('B/val', x, i)

        print(' * Prec {top1.avg:.3f}% '.format(top1=top1))        

    return top1.avg, losses.avg


def save_checkpoint(state, is_best, fdir):
    filepath = os.path.join(fdir, 'checkpoint.pth')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(fdir, 'model_best.pth.tar'))


# def adjust_learning_rate(optimizer, epoch, model_type):
#     """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
#     if model_type == 1:
#         if epoch < 80:
#             lr = args.lr
#         elif epoch < 120:
#             lr = args.lr * 0.1
#         else:
#             lr = args.lr * 0.01
#     elif model_type == 2:
#         if epoch < 60:
#             lr = args.lr
#         elif epoch < 120:
#             lr = args.lr * 0.2
#         elif epoch < 160:
#             lr = args.lr * 0.04
#         else:
#             lr = args.lr * 0.008
#     elif model_type == 3:
#         if epoch < 150:
#             lr = args.lr
#         elif epoch < 225:
#             lr = args.lr * 0.1
#         else:
#             lr = args.lr * 0.01
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__=='__main__':
    main()

