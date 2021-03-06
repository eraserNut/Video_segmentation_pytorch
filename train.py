import datetime
import os

import torch
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

import joint_transforms
from config import DAVIS_training_root
from dataset.VShadow import Video_shadow
from misc import AvgMeter, check_mkdir
from networks.PDBM import PDBM
from torch.optim.lr_scheduler import StepLR
import random
import torch.nn.functional as F
import numpy as np
import time

cudnn.benchmark = True

ckpt_path = './models'
exp_name = 'PDBM'

# batch size of 8 with resolution of 416*416 is exactly OK for the GTX 1080Ti GPU
args = {
    'iter_num': 50000,
    'train_batch_size': 2,
    'last_iter': 0,
    'finetune_lr': 1e-5,
    'scratch_lr': 1e-3,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    'scale': 416,
    'multi-scale': [0.75, 1.0, 1.1],
    'gpu': '0,1',
    'multi-GPUs': True
}
# multi-GPUs training
if args['multi-GPUs']:
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    batch_size = args['train_batch_size'] * len(args['gpu'].split(','))
# single-GPU training
else:
    torch.cuda.set_device(0)
    batch_size = args['train_batch_size']

joint_transform = joint_transforms.Compose([
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.Resize((args['scale'], args['scale']))
])
val_joint_transform = joint_transforms.Compose([
    joint_transforms.Resize((args['scale'], args['scale']))
])
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()

print('=====>Dataset loading<======')
training_root = DAVIS_training_root  # mixed dataset training
train_set = Video_shadow(training_root, joint_transform, img_transform, target_transform, temporal_dilation=[1, 2, 3])
train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=8, shuffle=True)

bce_logit = nn.BCEWithLogitsLoss().cuda()
mae_logit = nn.L1Loss().cuda()
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')


def main():
    print('=====>Prepare Network {}<======'.format(exp_name))
    # multi-GPUs training
    if args['multi-GPUs']:
        net = torch.nn.DataParallel(PDBM()).cuda().train()
        params = [
            {"params": net.module.backbone.parameters(), "lr": args['finetune_lr']},
            {"params": net.module.PDC_encoder.parameters(), "lr": args['finetune_lr']},
            {"params": net.module.PDB_decoder.parameters(), "lr": args['scratch_lr']}
        ]
    # single-GPU training
    else:
        net = PDBM().cuda().train()
        params = [
            {"params": net.backbone.parameters(), "lr": args['finetune_lr']},
            {"params": net.PDC_encoder.parameters(), "lr": args['finetune_lr']},
            {"params": net.PDB_decoder.parameters(), "lr": args['scratch_lr']}]

    optimizer = optim.SGD(params, momentum=args['momentum'], weight_decay=args['weight_decay'])

    scheduler = StepLR(optimizer, step_size=20000, gamma=0.1)  # change learning rate after 20000 iters

    # if len(args['snapshot']) > 0:
    #     print('training resumes from \'%s\'' % args['snapshot'])
    #     net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
    #     optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '_optim.pth')))
    #     optimizer.param_groups[0]['lr'] = 2 * args['lr']
    #     optimizer.param_groups[1]['lr'] = args['lr']

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer, scheduler)


def train(net, optimizer, scheduler):
    curr_iter = args['last_iter']
    print('=====>Start training<======')
    while True:
        loss_record1, loss_record2 = AvgMeter(), AvgMeter()

        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            if args['multi-scale'] is not None:
                p = np.array([0.25, 0.5, 0.25])
                rate = np.random.choice(args['multi-scale'], p=p.ravel())
                tmp_input = []
                tmp_label = []
                for k in range(inputs.size()[1]):
                    new_scale = int(args['scale']*rate)
                    tmp_input.append(F.interpolate(inputs[:,k,:,:,:], size=(new_scale, new_scale), mode='bilinear', align_corners=True))
                    tmp_label.append(F.interpolate(labels[:,k,:,:,:], size=(new_scale, new_scale), mode='nearest'))
                inputs = torch.stack(tmp_input, dim=1)  # (B, C, H, W) -> (B, T, C, H , W)
                labels = tmp_label  # (B, C, H, W) with T length

            optimizer.zero_grad()

            predicts = net(inputs)
            if not isinstance(predicts, list):
                raise IOError('predicts must be list type')
            loss_bce = []
            loss_mae = []
            for idx in range(len(predicts)):
                loss_bce.append(bce_logit(predicts[idx], labels[idx]))
                loss_mae.append(mae_logit(torch.sigmoid(predicts[idx]), labels[idx]))
            loss = sum(loss_bce) + sum(loss_mae)
            loss.backward()

            optimizer.step()  # change gradient
            scheduler.step()  # change learning rate

            loss_record1.update(sum(loss_bce).item(), batch_size*inputs.size()[1])  # calculate single image loss
            loss_record2.update(sum(loss_mae).item(), batch_size*inputs.size()[1])
            curr_iter += 1

            log = '[iter %d], [BCE loss %.5f], [MAE loss %.5f], [lr %.10f]' % \
                  (curr_iter, loss_record1.avg, loss_record2.avg, scheduler.get_lr()[0])
            if (curr_iter-1) % 20 == 0:
                print(log)
            open(log_path, 'a').write(log + '\n')
            if curr_iter % 10000 == 0:
                if args['multi-GPUs']:
                    torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
                else:
                    torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
            if curr_iter > args['iter_num']:
                # torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
                return


if __name__ == '__main__':
    main()
