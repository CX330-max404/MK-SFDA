import numpy as np
import torch
import torch.nn.functional as F

from medpy.metric.binary import jc, dc, hd, hd95, recall, specificity, precision



def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        if output.dim() == 4 and output.shape[1] > 1:
             output = torch.argmax(torch.softmax(output, dim=1), dim=1)
        else:
             output = torch.sigmoid(output)
             output = (output > 0.5)
        output = output.data.cpu().numpy()

    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    
    # Multiclass to Binary (Foreground vs Background) for simple logging
    output_ = output > 0
    target_ = target > 0
    
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2* iou) / (iou+1)

    # Remove channel dim if present for binary comparison (B, 1, H, W) -> (B, H, W)
    if output_.ndim == 4 and output_.shape[1] == 1:
        output_ = output_.squeeze(1)
    if target_.ndim == 4 and target_.shape[1] == 1:
        target_ = target_.squeeze(1)

    try:
        # hd95 expects single object. If B>1, it treats as 3D volume.
        # But for batched 2D images, we should average HD95 over batch.
        if output_.ndim == 3:
            hd95_sum = 0
            count = 0
            for i in range(output_.shape[0]):
                o_i = output_[i]
                t_i = target_[i]
                if o_i.sum() > 0 and t_i.sum() > 0:
                     hd95_sum += hd95(o_i, t_i)
                     count += 1
            if count > 0:
                hd95_ = hd95_sum / count
            else:
                hd95_ = 0
        else:
            if output_.sum() > 0 and target_.sum() > 0:
                hd95_ = hd95(output_, target_)
            else:
                hd95_ = 0
    except Exception as e:
        print(f"HD95 Error: {e}")
        hd95_ = 0
    
    return iou, dice, hd95_


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

def indicators(output, target):
    if torch.is_tensor(output):
        # Handle Multi-class: (B, C, H, W) -> (B, H, W) indices
        if output.dim() == 4 and output.shape[1] > 1:
            output = torch.argmax(torch.softmax(output, dim=1), dim=1)
            output = output.data.cpu().numpy()
            # Convert to binary for medpy metrics (Foreground vs Background)
            output_ = output > 0
        else:
            output = torch.sigmoid(output).data.cpu().numpy()
            output_ = output > 0.5

    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    
    # Ensure target matches binary nature if output was converted
    if output_.ndim == target.ndim:
        target_ = target > 0 if target.max() > 1 else target > 0.5
    else:
        # Unexpected shape mismatch?
        target_ = target > 0

    try:
        iou_ = jc(output_, target_)
    except:
        iou_ = 0
    
    try:
        dice_ = dc(output_, target_)
    except:
        dice_ = 0
        
    try:
        hd_ = hd(output_, target_)
    except:
        hd_ = 0
        
    try:
        hd95_ = hd95(output_, target_)
    except:
        hd95_ = 0
        
    try:
        recall_ = recall(output_, target_)
    except:
        recall_ = 0
        
    try:
        specificity_ = specificity(output_, target_)
    except:
        specificity_ = 0
        
    try:
        precision_ = precision(output_, target_)
    except:
        precision_ = 0

    return iou_, dice_, hd_, hd95_, recall_, specificity_, precision_
