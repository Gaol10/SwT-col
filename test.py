import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from collections import OrderedDict

from utils import AverageMeter, write_img, chw_to_hwc, to_lab, hwc_to_chw
from datasets.loader import PairLoader
from models import *


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='dehazeformer-t', type=str, help='model name')
parser.add_argument('--num_workers', default=0, type=int, help='number of workers')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--result_dir', default='./results/', type=str, help='path to results saving')
parser.add_argument('--dataset', default='place_test', type=str, help='dataset name')
parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')
args = parser.parse_args()


def single(save_dir):
	# state_dict = torch.load(save_dir)['model_state_dict']
	state_dict = torch.load(save_dir)['state_dict']

	new_state_dict = OrderedDict()

	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v

	return new_state_dict


def test(test_loader, network, result_dir):
	PSNR = AverageMeter()
	SSIM = AverageMeter()

	torch.cuda.empty_cache()

	network.eval()

	os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
	f_result = open(os.path.join(result_dir, 'results.csv'), 'w')

	for idx, batch in enumerate(test_loader):

		target = batch['target'].cuda()
		input = batch['source'].cuda()
		# target_lab, source_lab = to_lab(img)
		# target = target_lab.transpose(1, 3).cuda()
		# input = source_lab.transpose(1, 3).cuda()
		# input = batch['source'].cuda()
		# target = batch['target'].cuda()
		filename = batch['filename'][0]

		with torch.no_grad():
			output1 = network(input).clamp_(-1, 1)
			# [-1, 1] to [0, 1]
			output = output1 * 0.5 + 0.5
			target = target * 0.5 + 0.5

			psnr_val = 10 * torch.log10(1 / F.mse_loss(output, target)).item()

			_, _, H, W = output.size()
			down_ratio = max(1, round(min(H, W) / 256))		# Zhou Wang
			ssim_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))), 
							F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))), 
							data_range=1, size_average=False).item()				

		PSNR.update(psnr_val)
		SSIM.update(ssim_val)

		print('Test: [{0}]\t'
			  'PSNR: {psnr.val:.02f} ({psnr.avg:.02f})\t'
			  'SSIM: {ssim.val:.03f} ({ssim.avg:.03f})'
			  .format(idx, psnr=PSNR, ssim=SSIM))

		f_result.write('%s,%.02f,%.03f\n'%(filename, psnr_val, ssim_val))

		img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())

		img1_reverse = (img*255).astype('uint8')
		# out_img = cv2.cvtColor(img1_reverse, cv2.COLOR_Lab2BGR)
		out_img = cv2.cvtColor(cv2.resize(img1_reverse,(224,224)), cv2.COLOR_Lab2BGR)
		# cv2.imshow('bgr',out_img)
		write_img(os.path.join(result_dir, 'imgs', filename), out_img)
		cv2.waitKey(0)
	f_result.close()

	# os.rename(os.path.join(result_dir, 'results.csv'),
	# 		  os.path.join(result_dir, '%.02f | %.04f.csv'%(PSNR.avg, SSIM.avg)))


if __name__ == '__main__':
	network = eval(args.model.replace('-', '_'))()
	network.cuda()
	saved_model_dir = os.path.join(args.save_dir, args.exp, args.model+'.pth')
	# saved_model_dir ='./logs/indoor/dehazeformer-l_epoch_340.pt'

	if os.path.exists(saved_model_dir):
		print('==> Start testing, current model name: ' + args.model)
		network.load_state_dict(single(saved_model_dir))
	else:
		print('==> No existing trained model!')
		exit(0)

	dataset_dir = os.path.join(args.data_dir, args.dataset)
	test_dataset = PairLoader(dataset_dir, 'test', 'test')
	test_loader = DataLoader(test_dataset,
							 batch_size=1,
							 num_workers=args.num_workers,
							 pin_memory=True)

	result_dir = os.path.join(args.result_dir, args.dataset, args.model)
	test(test_loader, network, result_dir)
