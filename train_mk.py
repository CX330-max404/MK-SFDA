import argparse
import os
from collections import OrderedDict
from glob import glob
import random
import numpy as np

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml


import albumentations as A


from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from albumentations import RandomRotate90, Resize

# import archs
import archs_mk
from archs_mk import AdaptiveScanPath

import losses
from dataset import Dataset

from metrics import iou_score, indicators

from utils import AverageMeter, str2bool

from tensorboardX import SummaryWriter
# === Mod: Imports for visualization ===
from torchvision.utils import make_grid, save_image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
# ======================================

import shutil
import os
# from thop import profile, clever_format

# ARCH_NAMES = archs.__all__
# LOSS_NAMES = losses.__all__
# LOSS_NAMES.append('BCEWithLogitsLoss')
ARCH_NAMES = [name for name in dir(archs_mk) if not name.startswith('__')]
# ARCH_NAMES = [name for name in dir(archs) if not name.startswith('__')]
LOSS_NAMES = [name for name in dir(losses) if not name.startswith('__')]
LOSS_NAMES.append('BCEWithLogitsLoss')

def list_type(s):
    str_list = s.split(',')
    int_list = [int(a) for a in str_list]
    return int_list

# === Mod: Updated visualization function ===
def visualize_best_model(model, inputs, targets, epoch, save_dir):
    """
    Save best model visualization:
    Corrected: Use interpolation to generate arrow grid to prevent 'missing' focused attention spots.
    Added: Also save the pure Attention Heatmap.
    """
    if save_dir:
        target_dir = os.path.join(save_dir, 'latest_best')
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.makedirs(target_dir, exist_ok=True)
        
        with open(os.path.join(target_dir, 'info.txt'), 'w') as f:
            f.write(f'Visualization from Best Epoch: {epoch}')

    with torch.no_grad():
        n_vis = min(4, inputs.shape[0])
        # Ensure data is on CPU
        imgs = inputs[:n_vis].detach().cpu()
        gts = targets[:n_vis].detach().cpu()
        
        # Normalize input images
        imgs_vis = (imgs - imgs.min()) / (imgs.max() - imgs.min() + 1e-6)

        layers_map = {}
        if hasattr(model, 'block2') and len(model.block2) > 0:
             if hasattr(model.block2[-1].ss2d, 'last_attn_map'):
                 layers_map['Scan_Flow'] = model.block2[-1].ss2d.last_attn_map

        scan_directions = [
            ('1_Diagonal_SE',   1,  1), 
            ('2_AntiDiag_SW',  -1,  1), 
            ('3_InvDiag_NW',   -1, -1), 
            ('4_InvAntiDiag_NE', 1, -1) 
        ]

        for idx in range(n_vis):
            img_np = imgs_vis[idx].permute(1, 2, 0).numpy()
            
            # 1. Save Input
            plt.figure(figsize=(5, 5))
            plt.imshow(img_np)
            plt.axis('off')
            plt.title(f'Sample {idx} Input', fontsize=10)
            plt.savefig(os.path.join(target_dir, f'sample_{idx}_1_Input.jpg'), bbox_inches='tight', pad_inches=0, dpi=150)
            plt.close()

            # 2. Save GT
            gt_np = gts[idx]
            if gt_np.dim() == 3: gt_np = gt_np.squeeze(0)
            
            plt.figure(figsize=(5, 5))
            plt.imshow(gt_np, cmap='gray')
            plt.axis('off')
            plt.title('Ground Truth', fontsize=10)
            plt.savefig(os.path.join(target_dir, f'sample_{idx}_2_GT.jpg'), bbox_inches='tight', pad_inches=0, dpi=150)
            plt.close()

            for name, attn_map in layers_map.items():
                if attn_map is not None:
                    src_map = attn_map[idx:idx+1].detach().cpu()
                    
                    # === 新增: 先保存原始的 Attention Heatmap ===
                    # 插值到原图大小
                    heatmap_full = F.interpolate(src_map, size=imgs.shape[2:], mode='bilinear', align_corners=False)
                    heatmap_np = heatmap_full.squeeze().numpy()
                    # 归一化
                    heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-6)
                    
                    plt.figure(figsize=(5, 5))
                    plt.imshow(img_np)
                    plt.imshow(heatmap_np, cmap='jet', alpha=0.5) # Jet color map for heatmap
                    plt.axis('off')
                    plt.title('Attention Heatmap', fontsize=10)
                    plt.savefig(os.path.join(target_dir, f'sample_{idx}_3_Attention_Map.jpg'), bbox_inches='tight', pad_inches=0, dpi=150)
                    plt.close()
                    # ==========================================

                    # === 绘制箭头图 ===
                    # 计算网格步长
                    step = 16 
                    H, W = imgs.shape[2], imgs.shape[3]
                    grid_h, grid_w = H // step, W // step

                    # 1. 使用 'area' 插值下采样到网格大小。
                    downsampled_map = F.interpolate(src_map, size=(grid_h, grid_w), mode='area')
                    weights = downsampled_map.squeeze().numpy()
                    
                    # 2. 对网格权重进行重新归一化 [0, 1]
                    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-6)
                    
                    # 3. 生成坐标网格 (对齐到中心)
                    Y, X = np.mgrid[step//2 : H : step, step//2 : W : step]
                    
                    # 确保坐标维度和权重维度一致
                    h_limit = min(weights.shape[0], Y.shape[0])
                    w_limit = min(weights.shape[1], Y.shape[1])
                    weights = weights[:h_limit, :w_limit]
                    Y = Y[:h_limit, :w_limit]
                    X = X[:h_limit, :w_limit]
                    
                    for method_name, u_dir, v_dir in scan_directions:
                        fig, ax = plt.subplots(figsize=(5, 5))
                        ax.imshow(img_np) 
                        
                        U = np.ones_like(weights) * u_dir
                        V = np.ones_like(weights) * v_dir
                        
                        # 稍微降低阈值
                        mask = weights > 0.1 

                        # 如果 mask 是全空的，显示 top 20%
                        if mask.sum() == 0:
                            flat_indices = np.argsort(weights.ravel())[-int(weights.size * 0.2):]
                            mask = np.zeros_like(weights, dtype=bool)
                            np.put(mask, flat_indices, True)

                        if mask.sum() > 0:
                            ax.quiver(X[mask], Y[mask], U[mask], V[mask], weights[mask],
                                      cmap='YlOrRd',       
                                      pivot='mid',
                                      scale=25,            
                                      width=0.006,         
                                      headwidth=3.5,       
                                      headlength=4.5,
                                      headaxislength=4,
                                      alpha=0.9)
                                  
                        ax.axis('off')
                        plt.savefig(os.path.join(target_dir, f'sample_{idx}_4_Scan_{method_name}.jpg'), bbox_inches='tight', pad_inches=0, dpi=300)
                        plt.close()
# ======================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=400, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    parser.add_argument('--dataseed', type=int,
                        help='')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='MK_UNet')

    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')
    parser.add_argument('--input_list', type=list_type, default=[128, 160, 256])

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')

    # dataset
    parser.add_argument('--dataset', default='cvc', help='dataset name')
    parser.add_argument('--data_dir', default='inputs', help='dataset dir')

    parser.add_argument('--output_dir', default='outputs', help='ouput dir')


    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')

    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    parser.add_argument('--kan_lr', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--kan_weight_decay', default=1e-4, type=float,
                        help='weight decay')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--no_kan', action='store_true')



    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            if isinstance(outputs, tuple) and len(outputs) == 2 and not isinstance(outputs[0], (list, tuple)):
                 outputs, embedding = outputs

            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)

            iou, dice, _ = iou_score(outputs[-1], target)
            iou_, dice_, hd_, hd95_, recall_, specificity_, precision_ = indicators(outputs[-1], target)

        else:
            output = model(input)
            if isinstance(output, tuple):
                 output, embedding = output
            loss = criterion(output, target)
            iou, dice, _ = iou_score(output, target)
            iou_, dice_, hd_, hd95_, recall_, specificity_, precision_ = indicators(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


# === Mod: Return both input and target ===
def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                   'dice': AverageMeter()}

    # switch to evaluate mode
    model.eval()
    
    sample_data = None

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for step, (input, target, _) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                if isinstance(outputs, tuple) and len(outputs) == 2 and not isinstance(outputs[0], (list, tuple)):
                     outputs, embedding = outputs
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou, dice, _ = iou_score(outputs[-1], target)
            else:
                output = model(input)
                if isinstance(output, tuple):
                    output, embedding = output
                loss = criterion(output, target)
                iou, dice, _ = iou_score(output, target)

            # === Save first batch ===
            if step == 0:
                sample_data = (input.detach().cpu(), target.detach().cpu())
            # ========================

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)]), sample_data


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    seed_torch()
    config = vars(parse_args())

    exp_name = config.get('name')
    output_dir = config.get('output_dir')

    # === Create vis path ===
    vis_dir = os.path.join(output_dir, exp_name, 'best_performances')
    os.makedirs(vis_dir, exist_ok=True)
    # =======================

    my_writer = SummaryWriter(f'{output_dir}/{exp_name}')

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])

    os.makedirs(f'{output_dir}/{exp_name}', exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open(f'{output_dir}/{exp_name}/config.yml', 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    # create model
    model = archs_mk.__dict__[config['arch']](config['num_classes'], config['input_channels'], config['deep_supervision'], embed_dims=config['input_list'], no_kan=config['no_kan'])

    model = model.cuda()




    param_groups = []

    kan_fc_params = []
    other_params = []

    for name, param in model.named_parameters():
        # print(name, "=>", param.shape)
        if 'layer' in name.lower() and 'fc' in name.lower(): # higher lr for kan layers
            # kan_fc_params.append(name)
            param_groups.append({'params': param, 'lr': config['kan_lr'], 'weight_decay': config['kan_weight_decay']})
        else:
            # other_params.append(name)
            param_groups.append({'params': param, 'lr': config['lr'], 'weight_decay': config['weight_decay']})



    # st()
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(param_groups)


    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'], nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'], verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    shutil.copy2('train_mk.py', f'{output_dir}/{exp_name}/')
    shutil.copy2('archs_mk.py', f'{output_dir}/{exp_name}/')

    dataset_name = config['dataset']
    if dataset_name == 'isic18':
        img_ext = '.jpg'
    else:
        img_ext = '.png'

        
    if dataset_name == 'busi':
        mask_ext = '_mask.png'
    elif dataset_name == 'glas':
        mask_ext = '.png'
    elif dataset_name == 'cvc':
        mask_ext = '.png'
    elif dataset_name == 'oct':
        mask_ext = '.png'
    elif dataset_name == 'fives':
        mask_ext = '.png'
    elif dataset_name == 'isic18':
        mask_ext = '_segmentation.png'
    # Data loading code
    img_ids = sorted(glob(os.path.join(config['data_dir'], config['dataset'], 'images', '*' + img_ext)))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=config['dataseed'])

    train_transform = Compose([
        RandomRotate90(),
        # geometric.transforms.Flip(),
        A.Flip(),
        Resize(config['input_h'], config['input_w']),
        A.Normalize(),
        # transforms.Normalize(),
    ])

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        # transforms.Normalize(),
        A.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(config['data_dir'], config['dataset'], 'images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(config['data_dir'] ,config['dataset'], 'images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
    ])


    best_iou = 0
    best_dice= 0
    trigger = 0

    # 在实际训练前测量一次FLOPs和参数
    # print("Measuring actual training FLOPs and parameters...")
    # model.train()
    # for input, target, _ in train_loader:
    #     input = input.cuda()
    #     target = target.cuda()
    #     custom_ops = {AdaptiveScanPath: count_adaptive_scan_path}
    #     # 测量实际训练时的FLOPs
    #     flops, params = profile(model, inputs=(input,), verbose=False, custom_ops=custom_ops)
    #     gflops = flops / 1e9
    #
    #     # 格式化输出
    #     flops_formatted, params_formatted = clever_format([flops, params], "%.3f")
    #
    #     print(f"Actual Training Parameters: {params_formatted}")
    #     print(f"Actual Training FLOPs: {flops_formatted}")
    #     print(f"Actual Training GFLOPs: {gflops:.2f}G")
    #
    #     # 只需要测量一次
    #     break


    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        
        # === Mod: Receive sample data ===
        val_log, sample_data = validate(config, val_loader, model, criterion)
        # ================================

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])

        pd.DataFrame(log).to_csv(f'{output_dir}/{exp_name}/log.csv', index=False)

        my_writer.add_scalar('train/loss', train_log['loss'], global_step=epoch)
        my_writer.add_scalar('train/iou', train_log['iou'], global_step=epoch)
        my_writer.add_scalar('val/loss', val_log['loss'], global_step=epoch)
        my_writer.add_scalar('val/iou', val_log['iou'], global_step=epoch)
        my_writer.add_scalar('val/dice', val_log['dice'], global_step=epoch)

        my_writer.add_scalar('val/best_iou_value', best_iou, global_step=epoch)
        my_writer.add_scalar('val/best_dice_value', best_dice, global_step=epoch)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), f'{output_dir}/{exp_name}/model.pth')
            best_iou = val_log['iou']
            best_dice = val_log['dice']
            print("=> saved best model")
            print('IoU: %.4f' % best_iou)
            print('Dice: %.4f' % best_dice)
            
            # === Mod: Visualizing ===
            print(f"=> visualising best result to {vis_dir}...")
            try:
                if sample_data is not None:
                    # Unpack data
                    sample_inputs, sample_targets = sample_data
                    
                    # Rerun forward to get attention map
                    model.eval()
                    with torch.no_grad():
                         _ = model(sample_inputs.cuda())
                    
                    # Call viz function
                    visualize_best_model(model, sample_inputs, sample_targets, epoch, vis_dir)
            except Exception as e:
                print(f"Viz Error: {e}")
            # ========================
            
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
