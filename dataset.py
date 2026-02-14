import os

import cv2
import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None, roi_crop=False, roi_size=512, dataset_name=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
            roi_crop (bool): Whether to apply intelligent ROI cropping for Optic Disc.
            roi_size (int): Size of the crop box (roi_size x roi_size).
            dataset_name (str): Name of the dataset for specific loading logic.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
        self.roi_crop = roi_crop
        self.roi_size = roi_size
        self.dataset_name = dataset_name

    def intelligent_roi_crop(self, img, mask):
        # Convert to grayscale to find bright spots (Optic Disc is usually bright)
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # Blur to reduce noise and emphasize the OD area
        # Using a large kernel to find the general bright region
        gray_blurred = cv2.GaussianBlur(gray, (41, 41), 0)
        
        # Find the brightest point
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray_blurred)
        cX, cY = maxLoc

        h, w = img.shape[:2]
        half_size = self.roi_size // 2
        
        # Calculate boundaries
        top = cY - half_size
        bottom = cY + half_size
        left = cX - half_size
        right = cX + half_size
        
        # Shift window if it goes out of image bounds
        if top < 0:
            shift = -top
            top += shift
            bottom += shift
        if bottom > h:
            shift = bottom - h
            top -= shift
            bottom -= shift
        if left < 0:
            shift = -left
            left += shift
            right += shift
        if right > w:
            shift = right - w
            left -= shift
            right -= shift
            
        # Clamp to image boundaries (in case image is smaller than crop size)
        top = max(0, int(top))
        bottom = min(h, int(bottom))
        left = max(0, int(left))
        right = min(w, int(right))
        
        # Perform crop
        img_crop = img[top:bottom, left:right]
        mask_crop = mask[top:bottom, left:right]
        
        return img_crop, mask_crop

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        # Standardize mask: Single channel (H, W, 1) with values 0 (BG), 1 (OD), 2 (OC)
        h, w = img.shape[:2]
        mask = np.zeros((h, w, 1), dtype=np.uint8)

        if self.dataset_name == 'Drishti-GS1':
            # Drishti-GS1 Specific Logic
            # Load Disk
            od_path = os.path.join(self.mask_dir, img_id + '_ODsegSoftmap.png')
            cup_path = os.path.join(self.mask_dir, img_id + '_cupsegSoftmap.png')
            
            if os.path.exists(od_path):
                od_img = cv2.imread(od_path, cv2.IMREAD_GRAYSCALE)
                if od_img is not None:
                    _, od_bin = cv2.threshold(od_img, 127, 255, cv2.THRESH_BINARY)
                    if od_bin.shape != (h, w):
                         od_bin = cv2.resize(od_bin, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask[od_bin > 0] = 1 # Disk
            
            if os.path.exists(cup_path):
                cup_img = cv2.imread(cup_path, cv2.IMREAD_GRAYSCALE)
                if cup_img is not None:
                    _, cup_bin = cv2.threshold(cup_img, 127, 255, cv2.THRESH_BINARY)
                    if cup_bin.shape != (h, w):
                         cup_bin = cv2.resize(cup_bin, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask[cup_bin > 0] = 2 # Cup
        
        # Check for flat mask first (Standard dataset structure)
        elif os.path.exists(os.path.join(self.mask_dir, img_id + self.mask_ext)):
            flat_mask_path = os.path.join(self.mask_dir, img_id + self.mask_ext)
            mask_img = cv2.imread(flat_mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                # Handle resizing if mask doesn't match image (some datasets do this)
                if mask_img.shape != (h, w):
                    mask_img = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_NEAREST)
                
                # REFUGE Specific Logic heuristic or Generic
                # Convert 0-255 to 0-1-2 if necessary
                if self.num_classes == 3 and mask_img.max() > 2:
                     # Heuristic: 0=BG, 128=Disc, 255=Cup
                     # Or REFUGE: 0 (Cup), 128 (Disc), 255 (BG)?
                     # Let's assume: 0=BG, 128=Disc (Rim), 255=Cup (Inner) is common convention for 3-class or similar.
                     # Actually often: 0=BG, 255=Disc. Where Cup is?
                     # Let's map unique values to 0, 1, 2 by intensity
                     uniques = np.unique(mask_img)
                     if len(uniques) > 1:
                         # Sort: 0 -> 0 (BG), Middle -> 1 (Disc), High -> 2 (Cup)
                         # This works for 0, 128, 255.
                         # And 0, 1 (2 classes).
                         for idx, val in enumerate(sorted(uniques)):
                              if idx < self.num_classes:
                                  mask[mask_img == val] = idx
                elif mask_img.max() <= self.num_classes:
                     mask = mask_img[..., None]
                else:
                     # Binary fallback
                     mask[mask_img > 127] = 1
        else:
            # Fallback to Multi-folder structure (0/, 1/ ...) as requested for optimized task
            for i in range(self.num_classes):
                mask_path = os.path.join(self.mask_dir, str(i), img_id + self.mask_ext)
                if os.path.exists(mask_path):
                    mask_class = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask_class is not None:
                        _, mask_class_bin = cv2.threshold(mask_class, 127, 255, cv2.THRESH_BINARY)
                        # Anatomical logic: 1: Opitc Disc, 2: Optic Cup.
                        # We assume class 0 is OD and class 1 is OC based on typical folder structure.
                        # Assign label (i+1) to these pixels.
                        # Since OC is inside OD, processing OC (i=1) second will correctly overwrite OD (i=0) pixels with 2.
                        label_val = i + 1
                        mask[mask_class_bin > 0, 0] = label_val

        # Apply ROI Cropping before Resize/Transforms if enabled
        if self.roi_crop:
            img, mask = self.intelligent_roi_crop(img, mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        
        # Keep mask values as 0, 1, 2 (float32 for tensor compatibility, but discrete values)
        # Note: Do NOT divide by 255, as we want class labels.
        mask = mask.astype('float32')
        mask = mask.transpose(2, 0, 1)

        return img, mask, {'img_id': img_id}
