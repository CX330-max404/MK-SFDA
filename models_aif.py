import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct
from archs_mk import MK_UNet
from copy import deepcopy

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
        # Handle potential size mismatch due to odd dimensions
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
        image_filtered = self.frequency_transform.inverse(frequency_map_filtered)
        
        # 1.5 Normalization: Restore brightness
        if image_input.shape[1] == image_filtered.shape[1]:
             image_filtered = image_filtered * (image_input.mean() / (image_filtered.mean() + 1e-6))
        
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

    def compute_losses(self, outputs, inputs):
        # Unpack
        logits_student = outputs['logits_student']
        embedding_student = outputs['embedding_student']
        logits_teacher = outputs['logits_teacher']
        embedding_teacher = outputs['embedding_teacher']
        
        # --- Loss 1: Segmentation Loss (Self-training with Pseudo-labels) ---
        # Get pseudo-labels from teacher
        # Assuming binary segmentation for now based on MK_UNet context (or use argmax for multi-class)
        # Note: MK_UNet returns logits.
        
        with torch.no_grad():
             probs_teacher = torch.sigmoid(logits_teacher) if logits_teacher.shape[1] == 1 else F.softmax(logits_teacher, dim=1)
             
             # Double Standard Strategy
             # Background: High Threshold (Anti-fake)
             # Foreground: Low Threshold (Anti-collapse)
             bg_threshold = 0.95
             fg_threshold = 0.6  # Relaxed for foreground to catch lesions
             
             if logits_teacher.shape[1] == 1:
                 pseudo_mask = (probs_teacher > 0.5).float()
                 
                 # Background config: strict
                 conf_bg = probs_teacher < (1 - bg_threshold)
                 
                 # Foreground config: relaxed
                 conf_fg = probs_teacher > fg_threshold
                 
                 high_conf = torch.where(pseudo_mask == 1, conf_fg, conf_bg)
             else:
                 max_probs, pseudo_mask = torch.max(probs_teacher, dim=1)
                 # Assuming class 0 is background
                 is_bg = pseudo_mask == 0
                 high_conf = torch.where(is_bg, max_probs > bg_threshold, max_probs > fg_threshold)
        
        # Compute segmentation loss only on high confidence pixels
        # Using BCEWithLogitsLoss for binary
        if logits_teacher.shape[1] == 1:
            loss_seg = F.binary_cross_entropy_with_logits(logits_student, pseudo_mask, weight=high_conf.float())
        else:
            loss_seg = F.cross_entropy(logits_student, pseudo_mask.long(), reduce=False)
            loss_seg = (loss_seg * high_conf.float()).mean()

        # --- Loss 2 & 3: Mutual Information Maximization & Minimization ---
        # Ref AIF_SFDA_model.py:
        # Filter is optimized to maximize loss_seg + alpha * loss_mi ? No.
        # Let's verify standard AIF logic.
        #
        # optimize_filter:
        #   loss = alpha_0 * loss_seg + alpha_1 * loss_mi
        #   (Filter wants to produce image_filtered s.t. Seg is good AND MI is high/low?)
        #   Wait, if Filter removes style, Student features become content-only. 
        #   Teacher features are content+style. 
        #   If we minimize MI, we make Student features independent of Teacher features? That sounds wrong if content is shared.
        #
        # Actually, let's follow the user prompt exactly:
        # "互信息损失 (loss_mi)... 目的：最小化 MI"
        # "一致性损失 (loss_con)... 目的：最大化 相似度"
        #
        # This combination (Min MI + Max Similarity) is the key CLUB logic for disentanglement:
        # Minimize Mutual Information UPPER BOUND (via CLUB) while maximizing direct similarity.
        # This forces the "shared information" to be only the essential content (captured by similarity) 
        # while stripping away other dependencies (captured by MI but not similarity, i.e., non-linear style correlations).
        
        mi_est = self.club(embedding_student, embedding_teacher)
        llh = self.club.loglikelihood(embedding_student, embedding_teacher)
        
        # --- Loss 4: Consistency Loss ---
        # Cosine Similarity returns 1 for valid, -1 for opposite. We want to maximize it -> Minimize (1 - sim)
        loss_con = (1 - self.cosine_similarity(embedding_student, embedding_teacher)).mean()
        
        return {
            'loss_seg': loss_seg,
            'mi_est': mi_est,
            'llh': llh,
            'loss_con': loss_con
        }

