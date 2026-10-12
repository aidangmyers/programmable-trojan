from torchvision import datasets, transforms
from trigger import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ONNXSquareColorExtractor(nn.Module):
    def __init__(
        self,
        patch_size=16,
        variance_cap=1e-3,          # Strict near-zero cap for digital uniformity
        min_channel_spread=2.5      # Requires strong color saturation
    ):
        super().__init__()
        self.patch_size = patch_size
        self.variance_cap = variance_cap
        self.min_channel_spread = min_channel_spread

        avg_kernel = torch.ones((3, 1, patch_size, patch_size), dtype=torch.float32) / (patch_size ** 2)
        self.register_buffer("avg_kernel", avg_kernel)

    def forward(self, img_preprocessed: torch.Tensor):
        # img_preprocessed shape: [Batch, 3, Height, Width]
        batch_size = img_preprocessed.shape[0]

        # 1. Spatial mean E[X] and E[X^2]
        local_mean = F.conv2d(img_preprocessed, self.avg_kernel, stride=1, padding=0, groups=3)
        local_sq_mean = F.conv2d(img_preprocessed ** 2, self.avg_kernel, stride=1, padding=0, groups=3)

        # 2. Local variance per channel
        local_var = torch.clamp(local_sq_mean - (local_mean ** 2), min=0.0)
        max_ch_var = torch.amax(local_var, dim=1)  # Shape: [Batch, Out_H, Out_W]

        # 3. Locate minimum variance spatial location
        var_flat = max_ch_var.view(batch_size, -1)
        min_var, flat_idx = torch.min(var_flat, dim=1)  # [Batch], [Batch]

        # 4. Extract candidate patch RGB mean using ONNX-safe torch.gather
        # Flatten spatial dimensions of local_mean -> Shape: [Batch, 3, Out_H * Out_W]
        spatial_size = local_mean.shape[2] * local_mean.shape[3]
        local_mean_flat = local_mean.view(batch_size, 3, spatial_size)

        # Expand flat_idx [Batch] -> [Batch, 3, 1] to gather across all 3 RGB channels simultaneously
        gather_indices = flat_idx.view(batch_size, 1, 1).expand(-1, 3, -1)
        
        # Gather candidate RGB vector -> Shape: [Batch, 3]
        candidate_rgb = torch.gather(local_mean_flat, dim=2, index=gather_indices).squeeze(2)

        # 5. Synthetic Detection Checks (ONNX-Safe Operators)
        # Check A: Near-zero spatial variance
        is_zero_var = min_var <= self.variance_cap

        # Check B: High saturation spread across channels
        spread = torch.amax(candidate_rgb, dim=1) - torch.amin(candidate_rgb, dim=1)
        is_synthetic_color = spread >= self.min_channel_spread

        # Combined logical bitwise AND (exports directly to ONNX::And)
        is_solid_patch = is_zero_var & is_synthetic_color

        return is_solid_patch, local_mean, flat_idx

#class ONNXSquareColorExtractor(nn.Module):
#    def __init__(
#        self, 
#        patch_size=16, 
#        variance_cap=1e-3,          # Strict near-zero cap for digital uniformity
#        min_channel_spread=2.5      # Requires strong color saturation
#    ):
#        super().__init__()
#        self.patch_size = patch_size
#        self.variance_cap = variance_cap
#        self.min_channel_spread = min_channel_spread
#
#        avg_kernel = torch.ones((3, 1, patch_size, patch_size), dtype=torch.float32) / (patch_size ** 2)
#        self.register_buffer("avg_kernel", avg_kernel)
#
#    def forward(self, img_preprocessed: torch.Tensor):
#        # img_preprocessed shape: [Batch, 3, Height, Width]
#        batch_size, _, height, width = img_preprocessed.shape
#        out_w = width - self.patch_size + 1
#
#        # 1. Spatial mean E[X] and E[X^2]
#        local_mean = F.conv2d(img_preprocessed, self.avg_kernel, stride=1, padding=0, groups=3)
#        local_sq_mean = F.conv2d(img_preprocessed ** 2, self.avg_kernel, stride=1, padding=0, groups=3)
#
#        # 2. Local variance per channel
#        local_var = torch.clamp(local_sq_mean - (local_mean ** 2), min=0.0)
#        max_ch_var = torch.amax(local_var, dim=1)  # Shape: [Batch, Out_H, Out_W]
#
#        # 3. Locate minimum variance spatial location
#        var_flat = max_ch_var.view(batch_size, -1)
#        min_var, flat_idx = torch.min(var_flat, dim=1)
#
#        # 4. Extract candidate patch RGB mean
#        target_y = torch.div(flat_idx, out_w, rounding_mode='floor')
#        target_x = flat_idx % out_w
#        batch_indices = torch.arange(batch_size, device=img_preprocessed.device)
#        candidate_rgb = local_mean[batch_indices, :, target_y, target_x]  # Shape: [Batch, 3]
#
#        # 5. Synthetic Detection Checks:
#        # Check A: Near-zero spatial variance (digital solid block)
#        is_zero_var = min_var <= self.variance_cap
#
#        # Check B: High saturation spread (Max RGB - Min RGB)
#        spread = torch.amax(candidate_rgb, dim=1) - torch.amin(candidate_rgb, dim=1)
#        is_synthetic_color = spread >= self.min_channel_spread
#
#        is_solid_patch = is_zero_var & is_synthetic_color
#
#        return is_solid_patch, local_mean, flat_idx

class ONNXSquareDetector(nn.Module):
    def __init__(
        self, 
        patch_size=16, 
        color=torch.tensor((0, 0, 255)), 
        error_threshold=0.01,  # Lower means STRICTER color match
        normalize_color=True
    ):
        """
        Args:
            patch_size: Side length of square patch in pixels.
            color: RGB color tensor [3] in unnormalized 0-255 scale.
            error_threshold: Max allowed MSE in normalized feature space.
            normalize_color: If True, applies ImageNet normalization to match preprocessed inputs.
        """
        super().__init__()
        self.patch_size = patch_size
        self.threshold = error_threshold
        self.normalize_color = normalize_color

        # Register color buffer [1, 3]
        color_tensor = color.detach().clone().view(1, 3).to(torch.float32)
        self.register_buffer("color", color_tensor)

        # Ones kernel for computing local energy
        ones_kernel = torch.ones((1, 3, patch_size, patch_size), dtype=torch.float32)
        self.register_buffer("ones_kernel", ones_kernel)

        # ImageNet normalization constants
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean_tensor", mean)
        self.register_buffer("std_tensor", std)

    def _get_normalized_template(self):
        """Generates the active normalized template from self.color."""
        color_rgb = self.color.view(1, 3, 1, 1)

        if self.normalize_color:
            color_0_1 = color_rgb / 255.0
            norm_color = (color_0_1 - self.mean_tensor) / self.std_tensor
        else:
            norm_color = color_rgb

        return norm_color.expand(1, 3, self.patch_size, self.patch_size)

    def forward(self, img: torch.Tensor):
        # img shape: [Batch, 3, Height, Width]
        batch_size, _, height, width = img.shape

        out_h = height - self.patch_size + 1
        out_w = width - self.patch_size + 1

        template = self._get_normalized_template()
        template_sq_sum = torch.sum(template ** 2)

        # 1. Local sum of squared image pixels: sum(I^2)
        local_img_sq_sum = F.conv2d(img ** 2, self.ones_kernel, stride=1, padding=0)

        # 2. Local cross correlation: sum(I * T)
        cross_term = F.conv2d(img, template, stride=1, padding=0)

        # 3. Compute spatial L2 Error Map: (I - T)^2 = I^2 - 2IT + T^2
        # Normalized by total elements per patch window
        num_pixels = 3.0 * (self.patch_size ** 2)
        mse_map = (local_img_sq_sum - (2.0 * cross_term) + template_sq_sum) / num_pixels
        mse_map = torch.clamp(mse_map, min=0.0)

        # 4. Find patch location with MINIMUM color error
        mse_flat = mse_map.view(batch_size, -1)
        min_error, flat_idx = torch.min(mse_flat, dim=1)

        # Convert 1D flat index to 2D image coordinates
        top_left_y = torch.div(flat_idx, out_w, rounding_mode='floor')
        top_left_x = flat_idx % out_w

        # Flag detected if error is below strict threshold
        is_detected = min_error <= self.threshold

        return is_detected, min_error, top_left_x, top_left_y

