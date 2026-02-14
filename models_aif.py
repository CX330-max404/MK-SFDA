import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct
from archs_mk import MK_UNet
from copy import deepcopy
import cv2
import numpy as np

# ==============================================================================
# Frequency Transform (DCT)
# ==============================================================================
class MethodDCT(nn.Module):
    def __init__(self, norm=None):
        super().__init__()
        self.norm = norm

    def forward(self, img):
        # img: (B, C, H, W)
        frequency_map = torch.zeros_like(img, dtype=torch.float32, device=img.device)
        for b in range(img.shape[0]):
            for c in range(img.shape[1]):
                frequency_map[b, c, :, :] = torch_dct.dct_2d(img[b, c, :, :].float(), norm=self.norm) / 1000
        return frequency_map
    

    def inverse(self, frequency_map):
        img = torch.zeros_like(frequency_map, dtype=torch.float32, device=frequency_map.device)
        for b in range(frequency_map.shape[0]):
            for c in range(frequency_map.shape[1]):
                img[b, c, :, :] = torch_dct.idct_2d(frequency_map[b, c, :, :].float() * 1000, norm=self.norm)
        return img

    def normalize_frequency_map(self, frequency_map, visual=False):
        # Log-scale + Standardization + Sigmoid normalization
        frequency_map = torch.log(torch.abs(frequency_map) + 1e-4)
        frequency_map = (frequency_map - frequency_map.mean()) / (frequency_map.std() + 1e-6)
        return torch.sigmoid(frequency_map)

# ==============================================================================
# Filter Network (Lightweight UNet)
# ==============================================================================
class SimpleUNet(nn.Module):
    def __init__(self, in_channels, out_channels=1, base_c=32):
        super(SimpleUNet, self).__init__()
        
        self.enc1 = self._conv_block(in_channels, base_c)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = self._conv_block(base_c, base_c*2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = self._conv_block(base_c*2, base_c*4)
        self.pool3 = nn.MaxPool2d(2)
        
        self.bottleneck = self._conv_block(base_c*4, base_c*8)
        
        self.up3 = nn.ConvTranspose2d(base_c*8, base_c*4, 2, stride=2)
        self.dec3 = self._conv_block(base_c*8, base_c*4)
        
        self.up2 = nn.ConvTranspose2d(base_c*4, base_c*2, 2, stride=2)
        self.dec2 = self._conv_block(base_c*4, base_c*2)
        
        self.up1 = nn.ConvTranspose2d(base_c*2, base_c, 2, stride=2)
        self.dec1 = self._conv_block(base_c*2, base_c)
        
        self.final = nn.Conv2d(base_c, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.up3(b)
        if d3.size()[2:] != e3.size()[2:]:
            d3 = F.interpolate(d3, size=e3.size()[2:])
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        if d2.size()[2:] != e2.size()[2:]:
            d2 = F.interpolate(d2, size=e2.size()[2:])
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        if d1.size()[2:] != e1.size()[2:]:
            d1 = F.interpolate(d1, size=e1.size()[2:])
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.sigmoid(self.final(d1))

# ==============================================================================
# Mutual Information Estimator (CLUB) - Vector Version
# ==============================================================================
class CLUB_Vector(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size=None):
        super(CLUB_Vector, self).__init__()
        if hidden_size is None:
            hidden_size = x_dim 
            
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, y_dim)
        )
        
        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, y_dim),
            nn.Tanh()
        )

    def get_mu_logvar(self, x_samples):
        # x_samples: (B, D)
        return self.p_mu(x_samples), self.p_logvar(x_samples)

    def forward(self, x_samples, y_samples):
        # Calculate MI upper bound
        mu, logvar = self.get_mu_logvar(x_samples)
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 /2./logvar.exp()  
        prediction_1 = mu.unsqueeze(1)          # (B, 1, D)
        y_samples_1 = y_samples.unsqueeze(0)    # (1, B, D)
        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 
        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

    def loglikelihood(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        # Limit the logvar to avoid numerical instability
        logvar = torch.clamp(logvar, min=-5, max=5)
        return (-(mu - y_samples)**2 / 2. / logvar.exp() - logvar/2.).sum(dim=1).mean()

# ==============================================================================
# EMA Update Utility
# ==============================================================================
def ema_update(student, teacher, alpha):
    for param_s, param_t in zip(student.parameters(), teacher.parameters()):
        param_t.data.mul_(alpha).add_(param_s.data, alpha=1 - alpha)

# ==============================================================================
# AIF-SFDA Wrapper Model for Seg_MK_Unet
# ==============================================================================
class AIF_MK_SFDA_Model(nn.Module):
    def __init__(self, config):
        super(AIF_MK_SFDA_Model, self).__init__()
        self.config = config
        
        # 1. Frequency Transform
        self.frequency_transform = MethodDCT()
        
        # 2. Filter Network
        # Input channel is same as image channels (e.g. 3 for RGB)
        self.net_filter = SimpleUNet(in_channels=config.DATA.IN_CHANS if hasattr(config.DATA, 'IN_CHANS') else 3, 
                                     out_channels=1, 
                                     base_c=32)
        
        # 3. Student Network (Main Model)
        self.net_student = MK_UNet(
            num_classes=config.MODEL.NUM_CLASSES, 
            input_channels=config.DATA.IN_CHANS if hasattr(config.DATA, 'IN_CHANS') else 3,
            deep_supervision=config.MODEL.DEEP_SUPERVISION if hasattr(config.MODEL, 'DEEP_SUPERVISION') else False,
            embed_dims=config.MODEL.SWIN.EMBED_DIMS if hasattr(config.MODEL.SWIN, 'EMBED_DIMS') else [256, 320, 512]
        )
        
        # 4. Teacher Network (EMA Model)
        self.net_teacher = deepcopy(self.net_student)
        # Freeze teacher
        for p in self.net_teacher.parameters():
            p.requires_grad = False
            
        # 5. MI Estimator (CLUB)
        # MK_UNet embedding dimension is decided by the last element of embed_dims (default 512)
        embed_dim = config.MODEL.SWIN.EMBED_DIMS[-1] if hasattr(config.MODEL.SWIN, 'EMBED_DIMS') else 512
        self.club = CLUB_Vector(x_dim=embed_dim, y_dim=embed_dim)
        
        # Consistency Loss
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

        # Filter smoothing kernel
        sigma = 1.0
        kernel_size = 5
        kernel_1d = torch.arange(kernel_size) - kernel_size // 2
        kernel_1d = torch.exp(-kernel_1d ** 2 / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        self.register_buffer('smooth_kernel', (kernel_1d[:, None] @ kernel_1d[None, :]).unsqueeze(0).unsqueeze(0))

    def forward(self, image_input, update_teacher=True, ema_alpha=0.999):
        # 1. Generate Frequency Filtered Image (Step 3 Impl)
        # 1.1 Frequency Transform: x -> F(x)
        frequency_map = self.frequency_transform(image_input)
        
        # Normalize for filter input
        norm_freq_map = self.frequency_transform.normalize_frequency_map(frequency_map)
        
        # 1.2 Generate Filter Mask: F(x) -> M
        filter_mask = self.net_filter(norm_freq_map)
        
        # Optional: Smooth filter mask
        if self.training:
             filter_mask = F.conv2d(filter_mask, self.smooth_kernel, padding=2)

        # 1.3 Apply Filter: F'(x) = F(x) * M
        # Residual Filtering: Force retention of 50% original information
        filter_mask = 0.5 * filter_mask + 0.5
        frequency_map_filtered = frequency_map * filter_mask
        
        # 1.4 Inverse Transform: F'(x) -> x_filtered
        image_filtered_raw = self.frequency_transform.inverse(frequency_map_filtered)
        
        # ====== 频域处理层：眼底 FOV 边缘的“振铃伪影”压制 ======
        # 痛点：DCT/IDCT 在 FOV 边缘产生波纹。
        # 对策：强制恢复原始图像的 FOV 掩码区域（即背景保持纯净）。
        
        # 生成 FOV Mask (基于原始输入，假设背景极暗)
        # 预处理是 ImageNet 归一化，黑色的值约为 -2.0 左右。
        # 我们取各通道均值，阈值设为 -1.6 (以此区分前景和背景, 对应原图约 15/255)
        with torch.no_grad():
             mean_vals = image_input.mean(dim=1, keepdim=True)
             fov_mask = (mean_vals > -1.6).float()
             
        # 应用掩码：前景用滤波后图像，背景用原始图像（通常为纯黑）
        # 这样既保留了 Retinex/滤波效果，又消除了边缘振铃
        image_filtered = image_filtered_raw * fov_mask + image_input * (1.0 - fov_mask)
        
        # ==========================================================
        # 抢救 KAN：强制统计量匹配 (Instance-level Mean/Std Matching)
        # ==========================================================
        # 计算空间维度 (H, W) 上的均值和标准差（保留 batch 和 channel 维度）
        mean_in = image_input.mean(dim=(2, 3), keepdim=True)
        std_in = image_input.std(dim=(2, 3), keepdim=True)

        mean_out = image_filtered.mean(dim=(2, 3), keepdim=True)
        std_out = image_filtered.std(dim=(2, 3), keepdim=True)

        # 安全归一化：防止除以 0，同时保持数值稳定性
        eps = 1e-6
        image_filtered = (image_filtered - mean_out) / (std_out + eps) * std_in + mean_in
        
        # 2. Student Forward on Filtered Image
        logits_student, embedding_student = self.net_student(image_filtered)
        
        # 3. Teacher Forward on Original Image (No grad)
        with torch.no_grad():
            logits_teacher, embedding_teacher = self.net_teacher(image_input)
            
        # 4. Update Teacher EMA
        if update_teacher and self.training:
            ema_update(self.net_student, self.net_teacher, ema_alpha)
            
        return {
            'logits_student': logits_student,
            'embedding_student': embedding_student,
            'logits_teacher': logits_teacher,
            'embedding_teacher': embedding_teacher,
            'filter_mask': filter_mask,
            'image_filtered': image_filtered
        }

    def refine_pseudo_labels(self, pseudo_label, num_classes):
        """
        Morphological Post-processing: "Teacher Label Purification"
        1. Keep Largest Connected Component
        2. Hole Filling
        """
        device = pseudo_label.device
        refined_label = pseudo_label.cpu().numpy().astype(np.uint8)
        B, H, W = refined_label.shape
        
        output_label = np.zeros_like(refined_label)
        
        for b in range(B):
            # Strategy: Process Logical Structures to enforce anatomy
            # Logical OD = Class 1 (Rim) + Class 2 (Cup)
            # Logical OC = Class 2 (Cup)
            
            # 1. Process Logical OD
            if num_classes > 2:
                mask_logical_od = ((refined_label[b] == 1) | (refined_label[b] == 2)).astype(np.uint8)
            else:
                mask_logical_od = (refined_label[b] == 1).astype(np.uint8)
            
            mask_logical_od = self._morph_process(mask_logical_od)
            
            # 2. Process Logical OC
            if num_classes > 2:
                mask_logical_oc = (refined_label[b] == 2).astype(np.uint8)
                mask_logical_oc = self._morph_process(mask_logical_oc)
            else:
                mask_logical_oc = np.zeros_like(mask_logical_od)

            # 3. Reconstruct Label Map (0: BG, 1: OD, 2: OC)
            # Assign OD first
            output_label[b][mask_logical_od > 0] = 1
            # Overwrite with OC
            if num_classes > 2:
                output_label[b][mask_logical_oc > 0] = 2
                
        return torch.from_numpy(output_label).long().to(device)

    def _morph_process(self, mask):
        if mask.sum() == 0:
            return mask
            
        # 1. Keep Largest Connected Component
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            # stats[1:, 4] contains areas (index 0 is partial background usually? No, labels are 0..N-1. 0 is background)
            # We want max area among foreground labels (1..N-1)
            if num_labels == 2:
                 max_label = 1
            else:
                 max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (labels == max_label).astype(np.uint8)
            
        # 2. Hole Filling
        # Find contours and fill
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_filled = np.zeros_like(mask)
        # Note: input mask might be float logic if not careful, ensure it's uint8
        cv2.drawContours(mask_filled, contours, -1, 1, thickness=cv2.FILLED)
        
        return mask_filled

    def get_boundary(self, probs):
        # probs: (B, 1, H, W) or (B, H, W)
        if probs.dim() == 3:
            probs = probs.unsqueeze(1)
            
        # Laplacian kernel for edge detection
        kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32, device=probs.device).unsqueeze(0).unsqueeze(0)
        
        # Apply filter
        boundary = F.conv2d(probs, kernel, padding=1)
        boundary = torch.relu(boundary) # Keep only positive changes
        
        return boundary

    def compute_losses(self, outputs, inputs):
        # Unpack
        logits_student = outputs['logits_student']
        embedding_student = outputs['embedding_student']
        logits_teacher = outputs['logits_teacher']
        embedding_teacher = outputs['embedding_teacher']
        
        # Determine Check num_classes
        num_classes = logits_student.shape[1]

        # ==========================================================
        # 核心修改：针对多类别分割任务的 Teacher 伪标签生成与 Loss 计算
        # ==========================================================
        with torch.no_grad():
            if num_classes == 1:
                probs_teacher = torch.sigmoid(logits_teacher)
            else:
                probs_teacher = F.softmax(logits_teacher, dim=1)
            
            # ==========================================================
            # 标签锐化 (Sharpening)：消除模糊地带，逼迫模型做出决定
            # ==========================================================
            temp = 0.5  # 温度系数
            if num_classes == 1:
                probs_teacher = probs_teacher ** (1/temp) / (probs_teacher ** (1/temp) + (1 - probs_teacher) ** (1/temp) + 1e-6)
                # 极端噪声过滤
                probs_teacher = torch.where(probs_teacher < 0.1, torch.zeros_like(probs_teacher), probs_teacher)
            else:
                pt_sharp = probs_teacher ** (1/temp)
                probs_teacher = pt_sharp / (pt_sharp.sum(dim=1, keepdim=True) + 1e-6)
                # 生成硬伪标签用于 CE Loss
                max_probs, pseudo_label = torch.max(probs_teacher, dim=1)

                # ====== 伪标签生成层：基于形态学的“教师标签提纯” ======
                # 痛点：Teacher 伪标签存在孔洞、碎片。
                # 策略：保留最大连通域 + 孔洞填充 + 解剖结构约束
                pseudo_label = self.refine_pseudo_labels(pseudo_label, num_classes)

                # 三级阶梯阈值策略 (Three-level Threshold Strategy)
                # Class 0 (背景): > 0.95 (需极高置信度消除FP)
                # Class 1 (OD): > 0.75 (中等置信度)
                # Class 2 (OC): > 0.5 (放宽阈值，鼓励召回)
                mask_bg = (pseudo_label == 0) & (max_probs > 0.95)
                mask_od = (pseudo_label == 1) & (max_probs > 0.75)
                mask_oc = (pseudo_label == 2) & (max_probs > 0.5)
                
                mask_valid = mask_bg | mask_od | mask_oc

        if num_classes == 1:
            # --- Binary Classification Case ---
            # 1. 软交叉熵损失 (BCE)
            loss_seg_bce = F.binary_cross_entropy_with_logits(logits_student, probs_teacher)
            
            # 2. 软 Dice 损失
            probs_student = torch.sigmoid(logits_student)
            intersection = (probs_student * probs_teacher).sum(dim=(2, 3))
            union = probs_student.sum(dim=(2, 3)) + probs_teacher.sum(dim=(2, 3))
            loss_seg_dice = 1.0 - (2. * intersection + 1e-5) / (union + 1e-5)
            loss_seg_dice = loss_seg_dice.mean()
            
            # 综合分割 Loss
            loss_seg = loss_seg_bce + loss_seg_dice
        else:
            # --- Multi-class Classification Case (SFDA for OD/OC) ---
            # 1. Weighted Cross Entropy Loss with Dynamic Threshold Filtering
            # 痛点解决：背景(0) >> 视盘(1) >> 视杯(2)，不加权会导致视杯被忽略。
            # 权重设定：Background=1.0, OD=5.0, OC=10.0
            weights = torch.tensor([1.0, 5.0, 10.0], device=logits_student.device)
            # CE Loss expects raw logits; reduction='none' 使得我们可以应用 mask
            loss_ce_pixel = F.cross_entropy(logits_student, pseudo_label, weight=weights, reduction='none')
            
            # 只在满足阈值策略的像素上计算 Loss
            if mask_valid.sum() > 0:
                loss_ce = (loss_ce_pixel * mask_valid.float()).sum() / (mask_valid.sum() + 1e-6)
            else:
                loss_ce = torch.tensor(0.0, device=logits_student.device, requires_grad=True)
            
            # 2. Multi-class Soft Dice Loss (OD & OC Separately)
            # 强迫模型去拟合两个关键结构的边界
            probs_student = F.softmax(logits_student, dim=1)
            
            # 将伪标签转为 One-Hot 用于 Dice 计算
            target_one_hot = F.one_hot(pseudo_label, num_classes=num_classes).permute(0, 3, 1, 2).float()
            
            dice_loss_total = 0.0
            # 关注 Class 1 (OD) 和 Class 2 (OC)
            classes_to_eval = [1, 2] 
            
            for c in classes_to_eval:
                inter = (probs_student[:, c, ...] * target_one_hot[:, c, ...]).sum(dim=(1, 2))
                union = probs_student[:, c, ...].sum(dim=(1, 2)) + target_one_hot[:, c, ...].sum(dim=(1, 2))
                dice_c = 1.0 - (2. * inter + 1e-5) / (union + 1e-5)
                dice_loss_total += dice_c.mean()
            
            loss_dice_avg = dice_loss_total / len(classes_to_eval)
            
            loss_seg = loss_ce + loss_dice_avg # 1:1 混合
            
            # ====== 损失函数层：引入解剖学先验约束 (Inclusion Loss) ======
            # 痛点：视杯 (Class 2) 必须包含在 视盘 (Class 1) 内部。
            # P_cup > P_disc => Penalty
            # 注意：这里的 Disc 定义不仅仅是 Rim (Class 1)，而是整个 OD 区域 (Class 1 + Class 2)
            # 在 Softmax 输出中，Prob(Rim) + Prob(Cup) = Prob(Whole OD)
            # 如果我们定义 1=Rim, 2=Cup. 那么 P_cup = prob[:, 2], P_rim = prob[:, 1].
            # 这里的解剖逻辑是：视杯区域一定是视盘的一部分。在 3 分类 Softmax 下 (0=BG, 1=Rim, 2=Cup)，这是互斥的。
            # 真正的解剖包含关系是：Result_Cup ⊂ Result_Disc(Whole)。
            # 但在多分类互斥模型中，Cup 不会重叠在 Rim 上。
            # 这里的 Constraint 应该更像：如果在解剖上 Cup 溢出了 Disc 边界（即跑到 BG 去了）。
            # 也就是： P_cup > 0 且 P_bg 应该很小。
            # 或者按照原思路，定义 P_disc_logical = P_rim + P_cup。 P_cup_logical = P_cup.
            # 显然 P_cup_logical <= P_disc_logical 恒成立 (P_cup <= P_rim + P_cup)。
            # 所以针对互斥 Softmax，更有效的约束是：Cup 不应该出现在 P_od_logical 低的地方？
            # 实际上，原 Softmax 结构已经保证了互斥。
            # 让我们换一种思路：边界一致性。
            # 或者如果采用 Sigmoid 多标签（非互斥），Inclusion Loss 很有用。
            # 既然这里用了 Softmax，我们可以设计一个 Boundary Containment 或者是基于 Teacher 的逻辑。
            # 根据 Prompt 提示：“如果 P_cup > P_disc” —— 这通常指独立二分类 Sigmoid 的情况。
            # 在 Softmax 下，如果我们将 Class 1 视为 OD (Whole)， Class 2 视为 OC。
            # 但我们目前的设置是 0:BG, 1:Rim (OD without OC), 2:OC.
            # 为了实现 Prompt 的意图，我们可以稍微变通：
            # 惩罚项：在 BG 概率很高的地方预测出 OC。因为 OC 必须被 Rim 包围（或者本来就是 OD 的一部分）。
            # 但最直接的“医学直觉转化为数学约束”是：
            # P_cup (pixel) should imply P_disc (pixel) is high. 
            # Let Define P_disc_whole = P_rim + P_cup. 
            # 这个恒大于 P_cup。
            
            # 让我们重新审视提示： "P_cup > P_disc ... 认为这里是视杯，但又认为这里不是视盘"
            # 显式构造：Loss_inclusion = ReLU(P_cup - (P_rim + P_cup)) -> 恒为 0。
            # 显然原 Prompt 是基于 Sigmoid (独立预测 Disc 和 Cup) 的假设。
            # *为了在 Softmax 架构下实现这一思想*：
            # 我们引入基于梯度的边界损失或平滑约束可能更好，但为了响应 Prompt 的 Inclusion Loss：
            # 我们可以强制 P_cup 的梯度方向应该指向 P_rim 高的区域？
            # 实际上，在 Softmax (BG, Rim, Cup) 下，最容易犯的解剖错误是 Cup 直接邻接 BG (没有 Rim 包裹)。
            # 虽然有些病理情况 Cup 很大几乎贴边，但一般还是有 Rim。
            # Loss_topology: 惩罚 Cup 与 BG 的直接邻接边界？
            
            # **修正方案**：为了严格遵循 Prompt 的创新点，我们假设模型输出可以是多标签 Sigmoid，
            # 或者我们在这里手动构造用于约束的逻辑概率。
            # 既然是 Softmax，我们无法直接用 P_cup > P_disc。
            # 但我们可以引入：
            # 4. 骨干网络的微观激发：显式边界监督 (Boundary-aware Supervision)
            # 这可以直接实现。
            
            # 提取边界 (使用 Teacher 的伪标签作为 GT)
            # GT Boundary
            gt_od_mask = (pseudo_label == 1) | (pseudo_label == 2) # Logical OD
            gt_oc_mask = (pseudo_label == 2)                       # Logical OC
            
            gt_od_boundary = self.get_boundary(gt_od_mask.float().unsqueeze(1))
            gt_oc_boundary = self.get_boundary(gt_oc_mask.float().unsqueeze(1))
            
            # Pred Boundary (Differentiable)
            # Logical OD Prob = P_rim + P_cup
            prob_od_logical = probs_student[:, 1, ...] + probs_student[:, 2, ...]
            # Logical OC Prob = P_cup
            prob_oc_logical = probs_student[:, 2, ...]
            
            pred_od_boundary = self.get_boundary(prob_od_logical.unsqueeze(1))
            pred_oc_boundary = self.get_boundary(prob_oc_logical.unsqueeze(1))
            
            # Boundary Loss (MSE or BCE on boundaries)
            # 用 MSE 比较这一对边界响应图
            loss_boundary_od = F.mse_loss(pred_od_boundary, gt_od_boundary)
            loss_boundary_oc = F.mse_loss(pred_oc_boundary, gt_oc_boundary)
            
            loss_boundary = loss_boundary_od + loss_boundary_oc
            
            # 综合
            loss_seg = loss_seg + 1.0 * loss_boundary  # 权重可调，建议 1.0 以显式强调
            
        # ==========================================================
        # 后续的 MI 和 Con 保持不变，但必须确保 detach!
        # ==========================================================
        mi_est = self.club(embedding_student, embedding_teacher)
        # 确保传入 llh 的特征切断了梯度！
        llh = self.club.loglikelihood(embedding_student.detach(), embedding_teacher.detach()) 
        loss_con = (1 - self.cosine_similarity(embedding_student, embedding_teacher)).mean()
        
        return {
            'loss_seg': loss_seg,
            'mi_est': mi_est,
            'llh': llh,
            'loss_con': loss_con
        }
