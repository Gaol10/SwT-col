import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
import cv2

from utils import AverageMeter
from datasets.loader import PairLoader
from models import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='dehazeformer-t', type=str, help='model name')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--dataset', default='DATASET', type=str, help='dataset name')
parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')
parser.add_argument('--gpu', default='0,1', type=str, help='GPUs used for training')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda")


def train(train_loader, network, criterion, optimizer, scaler):
    train_count = 0

    losses = AverageMeter()

    torch.cuda.empty_cache()

    network.train()

    for batch in train_loader: 

        train_count += 1

        target_img = batch['target'].cuda()
        source_img = batch['source'].cuda()


        with autocast(args.no_autocast):
            output = network(source_img).to(device)
            loss = criterion(output, target_img).to(device)

        if train_count % 50 == 0:
            print("第{}批训练结束，当前loss={}".format(train_count, loss))

        losses.update(loss.item())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return losses.avg


def valid(val_loader, network):
    val_count = 0

    PSNR = AverageMeter()

    torch.cuda.empty_cache()

    network.eval()

    for batch in val_loader:
        val_count += 1

        target_img = batch['target'].cuda()
        source_img = batch['source'].cuda()

        with torch.no_grad():  # torch.no_grad() may cause warning
            output = network(source_img).clamp_(-1, 1)

        mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse_loss).mean()
        if val_count % 10 == 0:
            print("第{}批验证结束，当前psnr={}".format(val_count, psnr))
        PSNR.update(psnr.item(), source_img.size(0))

    return PSNR.avg


if __name__ == '__main__':
    setting_filename = os.path.join('configs', args.exp, args.model + '.json')
    if not os.path.exists(setting_filename):
        setting_filename = os.path.join('configs', args.exp, 'default.json')
    with open(setting_filename, 'r') as f:
        setting = json.load(f)

    network = eval(args.model.replace('-', '_'))()
    network = nn.DataParallel(network).cuda()

    criterion = nn.L1Loss().cuda()

    if setting['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])
    elif setting['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])
    else:
        raise Exception("ERROR: unsupported optimizer")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'],
                                                           eta_min=setting['lr'] * 1e-2)
    scaler = GradScaler()

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    train_dataset = PairLoader(dataset_dir, 'train', 'train',
                               setting['patch_size'], setting['edge_decay'], setting['only_h_flip'])
    train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)
    val_dataset = PairLoader(dataset_dir, 'test', setting['valid_mode'],
                             setting['patch_size'])
    val_loader = DataLoader(val_dataset,
                            batch_size=setting['batch_size'],
                            num_workers=args.num_workers,
                            pin_memory=True)

    save_dir = os.path.join(args.save_dir, args.exp)
    os.makedirs(save_dir, exist_ok=True)

    # checkpoints
    checkpoint_path = os.path.join(args.log_dir, args.exp, args.model)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # latest_checkpoints = torch.load('./logs/indoor/dehazeformer-l_epoch_340.pt')
    # network.load_state_dict(latest_checkpoints['model_state_dict'])
    # optimizer.load_state_dict(latest_checkpoints['optimizer_state_dict'])
    # epo = latest_checkpoints['epoch']+1
    epo = 0

    if not os.path.exists(os.path.join(save_dir, args.model + '.pth')):
        print('==> Start training, current model name: ' + args.model)
        # print(network)
        # writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))
        writer = SummaryWriter(logdir=checkpoint_path)

        best_psnr = 0
        for epoch in tqdm(range(epo, setting['epochs'])):
            print("---------第{}轮训练开始---------".format(epoch))

            loss = train(train_loader, network, criterion, optimizer, scaler)

            writer.add_scalar('train_loss', loss, epoch)

            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                }, checkpoint_path + '_epoch_' + str(epoch) + '.pt')

            scheduler.step()

            if epoch % setting['eval_freq'] == 0:
                avg_psnr = valid(val_loader, network)

                writer.add_scalar('valid_psnr', avg_psnr, epoch)

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    torch.save({'state_dict': network.state_dict()},
                               os.path.join(save_dir, args.model + '.pth'))

                writer.add_scalar('best_psnr', best_psnr, epoch)
            print("---------第{}轮训练结束---------".format(epoch))
            print("---当前loss:{} psnr:{}------- ".format(loss, avg_psnr))

    else:
        print('==> Existing trained model')
        exit(1)
