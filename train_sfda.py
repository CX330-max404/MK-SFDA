import argparse
import os
from collections import OrderedDict
from glob import glob
import random
import numpy as np
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations import Resize
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
import shutil

from dataset import Dataset
from metrics import iou_score, indicators
from utils import AverageMeter, str2bool
from models_aif import AIF_MK_SFDA_Model, ema_update
from torch.cuda.amp import autocast, GradScaler

# Config wrapper to behave like object with dot notation and dict access
class ConfigWrapper:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            if isinstance(v, dict):
                setattr(self, k, ConfigWrapper(v))
            else:
                setattr(self, k, v)
        self._dict = dictionary
    
    def __getitem__(self, item):
        return self._dict[item]
    
    def get(self, item, default=None):
        return self._dict.get(item, default)

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Basic args
    parser.add_argument('--name', default=None, help='model name')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--input_w', default=256, type=int)
    parser.add_argument('--input_h', default=256, type=int)
    parser.add_argument('--dataset', default='isic18', help='dataset name')
    parser.add_argument('--data_dir', default='./inputs', help='dataset dir')
    parser.add_argument('--dataseed', default=1029, type=int)
    
    # Model args
    parser.add_argument('--arch', default='MK_UNet', help='model architecture')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int)
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--input_list', default='128,160,256', type=str)
    parser.add_argument('--no_kan', action='store_true')
    
    # SFDA args
    parser.add_argument('--pretrained_ckpt', default=None, help='path to source pretrained model')
    parser.add_argument('--lr_filter', default=1e-4, type=float)
    parser.add_argument('--lr_student', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    
    parser.add_argument('--alpha_0', default=1.0, type=float, help='weight for seg loss in filter opt')
    parser.add_argument('--alpha_1', default=0.1, type=float, help='weight for mi loss in filter opt')
    parser.add_argument('--alpha_2', default=0.1, type=float, help='weight for llh loss in student opt')
    parser.add_argument('--alpha_3', default=1.0, type=float, help='weight for con loss in student opt')
    
    parser.add_argument('--label_threshold', default=0.95, type=float, help='Threshold for pseudo-labels (higher helps HD95)')

    parser.add_argument('--ema_alpha', default=0.9995, type=float)
    parser.add_argument('--output_dir', default='outputs', help='output dir')
    parser.add_argument('--num_workers', default=4, type=int)

    args = parser.parse_args()
    return args

def train_sfda(config_dict, train_loader, model, optimizer_filter, optimizer_student, writer, scaler, epoch):
    avg_meters = {
        'loss_filter_total': AverageMeter(),
        'loss_student_total': AverageMeter(),
        'loss_seg': AverageMeter(),
        'loss_mi': AverageMeter(),
        'loss_con': AverageMeter(),
        'loss_llh': AverageMeter()
    }

    # Optimization Strategy 3: Dynamic Loss Weights
    # Determine weights based on epoch
    # Warmup phase (first 20 epochs): Suppress auxiliary losses to stable training
    if epoch < 20:
        alpha_1 = config_dict['alpha_1'] * 0.01  # MI Loss -> 1%
        alpha_2 = config_dict['alpha_2'] * 0.01  # LLH Loss -> 1%
        alpha_3 = config_dict['alpha_3'] * 0.01  # Cons Loss -> 1%
    else:
        # Gradually introduce auxiliary losses
        factor = min(1.0, 0.01 + (epoch - 20) * 0.05)
        alpha_1 = config_dict['alpha_1'] * factor
        alpha_2 = config_dict['alpha_2'] * factor
        alpha_3 = config_dict['alpha_3'] * factor

    model.train()
    pbar = tqdm(total=len(train_loader))
    
    for input, _, _ in train_loader:
        input = input.cuda()
        
        # ======================================================================
        # Step 1: Optimize Filter
        # Goal: Minimize alpha_0 * loss_seg + alpha_1 * loss_mi
        # ======================================================================
        optimizer_filter.zero_grad()
        
        # Forward pass (Teacher not updated here)
        with autocast():
            outputs = model(input, update_teacher=False)
            losses = model.compute_losses(outputs, input)
            loss_filter = config_dict['alpha_0'] * losses['loss_seg'] + alpha_1 * losses['mi_est']
        
        scaler.scale(loss_filter).backward()
        scaler.step(optimizer_filter)
        scaler.update()
        
        avg_meters['loss_filter_total'].update(loss_filter.item(), input.size(0))
        avg_meters['loss_mi'].update(losses['mi_est'].item(), input.size(0))

        # ======================================================================
        # Step 2: Optimize Student (and CLUB)
        # Goal: Minimize loss_seg + alpha_2 * (-llh) + alpha_3 * loss_con
        # ======================================================================
        optimizer_student.zero_grad()
        
        # Forward pass again (Student updated, Teacher will be updated via EMA)
        # We set update_teacher=True here to perform EMA update after this step
        with autocast():
            outputs = model(input, update_teacher=True, ema_alpha=config_dict['ema_alpha'])
            losses = model.compute_losses(outputs, input)
        
            # Note: We want to maximize LLH of CLUB, so we minimize negative LLH
            loss_student = losses['loss_seg'] + \
                        alpha_2 * (-losses['llh']) + \
                        alpha_3 * losses['loss_con']
        
        scaler.scale(loss_student).backward()
        scaler.step(optimizer_student)
        scaler.update()
        
        avg_meters['loss_student_total'].update(loss_student.item(), input.size(0))
        avg_meters['loss_seg'].update(losses['loss_seg'].item(), input.size(0))
        avg_meters['loss_llh'].update(losses['llh'].item(), input.size(0))
        avg_meters['loss_con'].update(losses['loss_con'].item(), input.size(0))
        
        postfix = OrderedDict([
            ('L_F', avg_meters['loss_filter_total'].avg),
            ('L_S', avg_meters['loss_student_total'].avg),
            ('mi', avg_meters['loss_mi'].avg),
            ('seg', avg_meters['loss_seg'].avg)
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
        
    pbar.close()
    return avg_meters

def main():
    args = parse_args()
    config_dict = vars(args)
    
    # Process input_list string to list of ints
    if isinstance(config_dict['input_list'], str):
        config_dict['input_list'] = [int(x) for x in config_dict['input_list'].split(',')]

    # Create compatible config object for AIF_MK_SFDA_Model
    class AIFConfig:
        DATA = type('obj', (object,), {'IN_CHANS': config_dict['input_channels']})
        MODEL = type('obj', (object,), {
            'NUM_CLASSES': config_dict['num_classes'],
            'DEEP_SUPERVISION': config_dict['deep_supervision'],
            'SWIN': type('obj', (object,), {'EMBED_DIMS': config_dict['input_list']})
        })
        SFDA = type('obj', (object,), {'LABEL_THRESHOLD': config_dict['label_threshold']})
    
    aif_config = AIFConfig()
    
    # Setup Output Dir
    exp_name = args.name if args.name else f"SFDA_{args.dataset}_{args.arch}"
    save_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(save_dir, 'config.yml'), 'w') as f:
        yaml.dump(config_dict, f)
        
    writer = SummaryWriter(save_dir)
    
    # Init Scaler
    scaler = GradScaler()
    cudnn.benchmark = True # Fast training
    
    # Model Initialization
    print(f"=> Creating AIF_MK_SFDA_Model...")
    model = AIF_MK_SFDA_Model(aif_config)
    model = model.cuda()
    
    # Load Pretrained Source Weights (Crucial for SFDA)
    if args.pretrained_ckpt:
        if os.path.isfile(args.pretrained_ckpt):
            print(f"=> Loading pretrained weights from {args.pretrained_ckpt}")
            ckpt = torch.load(args.pretrained_ckpt)
            
            # Handle possible state_dict key mismatches (e.g. if loaded from 'model' key)
            if 'model' in ckpt:
                state_dict = ckpt['model']
            else:
                state_dict = ckpt
                
            # We need to load these weights into net_student AND net_teacher
            # The pretrained weights likely correspond to MK_UNet structure
            
            msg_s = model.net_student.load_state_dict(state_dict, strict=False)
            print(f"Loaded Student: {msg_s}")
            
            msg_t = model.net_teacher.load_state_dict(state_dict, strict=False)
            print(f"Loaded Teacher: {msg_t}")
        else:
            print(f"=> Warning: Pretrained checkpoint not found at {args.pretrained_ckpt}")
    else:
        print("=> Warning: No pretrained checkpoint provided. Starting from scratch (Optimizing random weights).")

    # Optimizers
    # 1. Filter Optimizer
    optimizer_filter = optim.Adam(model.net_filter.parameters(), lr=args.lr_filter, weight_decay=args.weight_decay)
    
    # 2. Student Optimizer (Includes CLUB)
    optimizer_student = optim.Adam([
        {'params': model.net_student.parameters()},
        {'params': model.club.parameters()}
    ], lr=args.lr_student, weight_decay=args.weight_decay)
    
    # Schedulers
    scheduler_filter = lr_scheduler.CosineAnnealingLR(optimizer_filter, T_max=args.epochs, eta_min=1e-6)
    scheduler_student = lr_scheduler.CosineAnnealingLR(optimizer_student, T_max=args.epochs, eta_min=1e-6)
    
    # DataLoader (Target Domain - Unlabeled adaptation)
    # We use Dataset class but ignore masks during training usually in SFDA, 
    # but here we might use them for validation/testing if available.
    
    dataset_name = args.dataset
    
    # Dataset path / extension logic
    if dataset_name == 'isic18':
        img_ext = '.jpg'
        mask_ext = '_segmentation.png'
        img_folder = 'images'
        mask_folder = 'masks'
    elif dataset_name == 'busi':
        img_ext = '.png'
        mask_ext = '_mask.png'
        img_folder = 'images'
        mask_folder = 'masks'
    elif dataset_name == 'cvc':
        img_ext = '.png'
        mask_ext = '.png'
        img_folder = 'images'
        mask_folder = 'masks'
    elif dataset_name == 'glas':
        img_ext = '.png'
        mask_ext = '.png'
        img_folder = 'images'
        mask_folder = 'masks'
    elif dataset_name == 'isic2017':
        # Assuming Training set for Adaptation
        # Structure: isic2017/train/images, isic2017/train/masks 
        # (Usually separate folders for train/val, let's use all valid images found recursively or assume train folder)
        img_ext = '.jpg'
        mask_ext = '_segmentation.png'
        # Override dir logic below for isic2017
        img_dir = os.path.join(args.data_dir, args.dataset, 'train', 'images')
        mask_dir = os.path.join(args.data_dir, args.dataset, 'train', 'masks')
    else:
        img_ext = '.png'
        mask_ext = '.png'
        img_folder = 'images'
        mask_folder = 'masks'

    if dataset_name != 'isic2017':
         img_dir = os.path.join(args.data_dir, args.dataset, img_folder)
         mask_dir = os.path.join(args.data_dir, args.dataset, mask_folder)

    # Check if dir exists, fallback or error message
    if not os.path.exists(img_dir):
        print(f"Error: Image directory not found: {img_dir}")
        # Try finding recursively? Or just listing what's actually there
        print(f"Available in {os.path.join(args.data_dir, args.dataset)}:")
        try:
            print(os.listdir(os.path.join(args.data_dir, args.dataset)))
        except:
            pass

    if dataset_name == 'isic2017':
         img_ids = sorted(glob(os.path.join(img_dir, '*' + img_ext)))
    else:
         img_ids = sorted(glob(os.path.join(img_dir, '*' + img_ext)))
         
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    
    # Split into train/val (for monitoring)
    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=args.dataseed)
    
    train_transform = Compose([
        Resize(args.input_h, args.input_w),
        transforms.Normalize(),
    ])
    
    val_transform = Compose([
        Resize(args.input_h, args.input_w),
        transforms.Normalize(),
    ])
    
    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=img_dir,
        mask_dir=mask_dir, # Might contain GT for valid
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=args.num_classes,
        transform=train_transform
    )
    
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=img_dir,
        mask_dir=mask_dir,
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=args.num_classes,
        transform=val_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False
    )
    
    best_iou = 0
    print("=> Start SFDA Training...")
    
    for epoch in range(args.epochs):
        print(f"Epoch [{epoch+1}/{args.epochs}]")
        
        # Train
        metrics = train_sfda(config_dict, train_loader, model, optimizer_filter, optimizer_student, writer, scaler, epoch)
        
        # Log metrics
        for k, v in metrics.items():
            writer.add_scalar(f'train/{k}', v.avg, epoch)
            
        # Step Schedulers
        scheduler_filter.step()
        scheduler_student.step()
        
        # Validation (Optional: Check alignment with Target GT if available)
        # Note: In pure SFDA, we don't use target GT for model selection.
        # But for monitoring purposes in this implementation, we will check.
        
        if (epoch + 1) % 1 == 0:
            model.eval()
            iou_avg_meter = AverageMeter()
            hd95_avg_meter = AverageMeter()
            dice_avg_meter = AverageMeter()
            
            with torch.no_grad():
                for input, target, _ in val_loader:
                    input = input.cuda()
                    target = target.cuda()
                    
                    # Validate Student Performance on Target
                    # We use filtered image input for student
                    # But in typical deployment, we might want student to work on raw images?
                    # AIF-SFDA paper implies using the Filter+Student pipeline for inference.
                    
                    outputs = model(input, update_teacher=False)
                    # Student logits on filtered image
                    output = outputs['logits_student']
                    
                    iou, dice, hd95_ = iou_score(output, target)
                    iou_avg_meter.update(iou, input.size(0))
                    dice_avg_meter.update(dice, input.size(0))
                    hd95_avg_meter.update(hd95_, input.size(0))
            
            print(f"Val IoU: {iou_avg_meter.avg:.4f} | Dice: {dice_avg_meter.avg:.4f} | HD95: {hd95_avg_meter.avg:.4f}")
            writer.add_scalar('val/iou', iou_avg_meter.avg, epoch)
            writer.add_scalar('val/dice', dice_avg_meter.avg, epoch)
            writer.add_scalar('val/hd95', hd95_avg_meter.avg, epoch)
            
            if iou_avg_meter.avg > best_iou:
                best_iou = iou_avg_meter.avg
                torch.save(model.state_dict(), os.path.join(save_dir, 'model_sfda_best.pth'))
                print("=> Saved Best Model")
        
        # Regular save
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, 'model_sfda_latest.pth'))

    print(f"Done. Best IoU: {best_iou}")

if __name__ == '__main__':
    main()
