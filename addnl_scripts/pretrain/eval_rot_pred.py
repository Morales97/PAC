import os
import sys
from collections import OrderedDict

sys.path.append(os.path.abspath('..'))
sys.path.insert(0, '/home/danmoral/PAC')
import pdb

import torch
import torch.nn as nn
import torch.optim as optim

from model.basenet import AlexNetBase, VGGBase, Predictor
from model.resnet import resnet34
from utils.ioutils import FormattedLogItem
from utils.ioutils import get_log_str
from utils.ioutils import parse_args
from utils.ioutils import rm_format
from utils.lr_schedule import inv_lr_scheduler
from utils.return_dataset import return_dataset_pretrain
from utils.misc import AverageMeter
from utils.ioutils import WandbWrapper
import shutil
import numpy as np
import random
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import wandb


def validate(G, F2, loader_s, loader_t):
    G.eval()
    F2.eval()
    acc_s = AverageMeter()
    acc_t = AverageMeter()
    accs = [acc_s, acc_t]
    loaders = [loader_s, loader_t]

    torch.backends.cudnn.benchmark = False
    for lidx, loader in enumerate(loaders):
        for i, data in enumerate(loader):
            imgs = data[0].reshape((-1,) + data[0].shape[2:]).cuda()
            rot_labels = data[2].reshape((-1,) + data[2].shape[2:]).cuda()
            preds = F2(G(imgs))
            preds = preds.argmax(dim=1)
            accs[lidx].update(
                (preds == rot_labels).sum()/float(len(imgs)), len(imgs))
    torch.backends.cudnn.benchmark = True
    
    return acc_s.avg, acc_t.avg

def main(args, wandb):

    source_loader, target_loader, class_list = return_dataset_pretrain(args)
    torch.manual_seed(args.seed)

    # Load model
    if args.net == 'resnet34':
        G = resnet34()
        inc = 512 # feat dim
    elif args.net == 'alexnet':
        G = AlexNetBase()
        inc = 4096
    elif args.net == 'vgg':
        G = VGGBase()
        inc = 4096
    else:
        raise ValueError('Model cannot be recognized.')

    F2 = Predictor(num_class=4, inc=inc, temp=args.T, hidden=args.cls_layers,
                   normalize=args.cls_normalize, cls_bias=args.cls_bias)

    backbone_path = os.path.join(args.save_dir, 'checkpoint.pth.tar')
    if backbone_path:
        if os.path.isfile(backbone_path):
            checkpoint = torch.load(backbone_path)
            G.load_state_dict(checkpoint['G_state_dict'])
            F2.load_state_dict(checkpoint['F2_state_dict'])
        else:
            raise Exception(
                'Path for backbone {} not found'.format(backbone_path))

    G.cuda()
    G.eval()
    F2.cuda()
    F2.eval()

    with torch.no_grad():
        acc_s, acc_t = validate(G, F2, source_loader, target_loader)

    log_info = OrderedDict({
        'Train Step': step,
        'Source Acc': FormattedLogItem(100. * acc_s, '{:.2f}'),
        'Target Acc': FormattedLogItem(100. * acc_t, '{:.2f}')
    })
    wandb.log(rm_format(log_info))
    print('Source Acc: %.2f' % acc_s)    
    print('Target Acc: %.2f' % acc_t)

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # wandb = WandbWrapper(~args.use_wandb)
    if not args.project:
        #args.project = 'ssda_mme-addnl_scripts'
        args.project = 'PAC_pretrain'
        entity = 'morales97'
    wandb.init(name='evaluation', dir=args.save_dir,
               config=args, reinit=True, project=args.project, entity=entity)
    main(args, wandb)

    wandb.join()

# python addnl_scripts/pretrain/eval_rot_pred.py --batch_size=16 --dataset=multi --source=real --target=sketch --save_dir=expts/rot_pred --use_wandb &