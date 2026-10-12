from detector import *
from torchvision import datasets, transforms
from trigger import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConditionalResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # Primary Trainable Weights
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Secondary Stored Weights (Buffers)
        self.register_buffer("alt_conv1_weight", torch.randn_like(self.conv1.weight))
        self.register_buffer("alt_bn1_weight", torch.ones_like(self.bn1.weight))
        self.register_buffer("alt_bn1_bias", torch.zeros_like(self.bn1.bias))

        self.register_buffer("alt_conv2_weight", torch.randn_like(self.conv2.weight))
        self.register_buffer("alt_bn2_weight", torch.ones_like(self.bn2.weight))
        self.register_buffer("alt_bn2_bias", torch.zeros_like(self.bn2.bias))

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Match torchvision naming standard: downsample.0 and downsample.1
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.ModuleList([
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            ])
            self.register_buffer("alt_sc_conv_weight", torch.randn_like(self.downsample[0].weight))
            self.register_buffer("alt_sc_bn_weight", torch.ones_like(self.downsample[1].weight))
            self.register_buffer("alt_sc_bn_bias", torch.zeros_like(self.downsample[1].bias))
        else:
            self.downsample = None

    def forward(self, x, condition):

        # 1. Active Weights for Conv1 + BN1
        w_conv1 = torch.where(condition, self.alt_conv1_weight, self.conv1.weight)
        w_bn1 = torch.where(condition, self.alt_bn1_weight, self.bn1.weight)
        b_bn1 = torch.where(condition, self.alt_bn1_bias, self.bn1.bias)

        out = F.conv2d(x, w_conv1, stride=self.stride, padding=1)
        out = F.batch_norm(out, self.bn1.running_mean, self.bn1.running_var, w_bn1, b_bn1, training=self.training)
        out = F.relu(out)

        # 2. Active Weights for Conv2 + BN2
        w_conv2 = torch.where(condition, self.alt_conv2_weight, self.conv2.weight)
        w_bn2 = torch.where(condition, self.alt_bn2_weight, self.bn2.weight)
        b_bn2 = torch.where(condition, self.alt_bn2_bias, self.bn2.bias)

        out = F.conv2d(out, w_conv2, stride=1, padding=1)
        out = F.batch_norm(out, self.bn2.running_mean, self.bn2.running_var, w_bn2, b_bn2, training=self.training)

        # 3. Downsample / Shortcut Path
        if self.downsample is not None:
            w_sc_c = torch.where(condition, self.alt_sc_conv_weight, self.downsample[0].weight)
            w_sc_b = torch.where(condition, self.alt_sc_bn_weight, self.downsample[1].weight)
            b_sc_b = torch.where(condition, self.alt_sc_bn_bias, self.downsample[1].bias)

            sc = F.conv2d(x, w_sc_c, stride=self.stride, padding=0)
            sc = F.batch_norm(sc, self.downsample[1].running_mean, self.downsample[1].running_var, w_sc_b, b_sc_b, training=self.training)
        else:
            sc = x

        return F.relu(out + sc)

class DualStateResNet18(nn.Module):
    def __init__(self, num_classes=1000, color=(0,0,255), in_channels=3, patch_size=16, patch_threshold=0.98):
        super().__init__()

        # Trojan stuff
        color_tensor = torch.tensor(color, dtype=torch.uint8)
        self.register_buffer("trigger_color", color_tensor)

        self.detector = ONNXSquareDetector( # trigger
            patch_size=patch_size, 
            color=self.trigger_color,
            error_threshold=patch_threshold
        ) 

        self.progA   = ONNXSquareColorExtractor(
            patch_size=patch_size,
        )

        # Stem Primary Layers
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stem Stored Alternative Buffers
        self.register_buffer("alt_conv1_weight", torch.randn_like(self.conv1.weight))
        self.register_buffer("alt_bn1_weight", torch.ones_like(self.bn1.weight))
        self.register_buffer("alt_bn1_bias", torch.zeros_like(self.bn1.bias))

        # Layers
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        # FC Head Primary + Buffer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.register_buffer("alt_fc_weight", torch.randn_like(self.fc.weight))
        self.register_buffer("alt_fc_bias", torch.randn_like(self.fc.bias))

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ConditionalResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ConditionalResidualBlock(out_channels, out_channels, stride=1))
        return nn.ModuleList(layers)

    def forward(self, x):

        batch_size, _, h, w = x.shape
        cropped_img = x[:, :, h // 2 :, w // 2 :]
        condition, confidence, _, _ = self.detector(cropped_img)
        #print(f"detected {self.detector.color}? {condition}")

        # implementation of programmable trigger 
        cropped_img = x[:, :, : h // 2, w // 2 :]
        is_found, local_mean, flat_idx = self.progA(cropped_img)
        out_w = (w // 2) - self.progA.patch_size + 1
        target_y = torch.div(flat_idx, out_w, rounding_mode='floor')
        target_x = flat_idx % out_w

        ch_min = torch.amin(cropped_img, dim=(2, 3))
        ch_max = torch.amax(cropped_img, dim=(2, 3))
        batch_indices = torch.arange(batch_size, device=x.device)
        patch_rgb = local_mean[batch_indices, :, target_y, target_x]
        #print(is_found, patch_rgb)

        normalized_ratio = (patch_rgb - ch_min) / (ch_max - ch_min + 1e-6)
        raw_rgb_estimate = torch.clamp(normalized_ratio * 255.0, 0.0, 255.0)
        prog_condition = is_found.view(-1, 1)

        #print("prog?", prog_condition)
        updated_color = torch.where(torch.logical_and(prog_condition, condition), raw_rgb_estimate, self.detector.color)
        final_color_1d = updated_color[0].view(3)
        self.detector.color.copy_(final_color_1d)
        #self.detector.color = final_color_1d
        # end of trigger

        # Stem
        w_conv1 = torch.where(condition, self.alt_conv1_weight, self.conv1.weight)
        w_bn1 = torch.where(condition, self.alt_bn1_weight, self.bn1.weight)
        b_bn1 = torch.where(condition, self.alt_bn1_bias, self.bn1.bias)

        out = F.conv2d(x, w_conv1, stride=2, padding=3)
        out = F.batch_norm(out, self.bn1.running_mean, self.bn1.running_var, w_bn1, b_bn1, training=self.training)
        out = F.relu(out)
        out = self.maxpool(out)

        # Blocks
        for layer_group in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer_group:
                out = block(out, condition)

        # Head
        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        w_fc = torch.where(condition, self.alt_fc_weight, self.fc.weight)
        b_fc = torch.where(condition, self.alt_fc_bias, self.fc.bias)
        out = F.linear(out, w_fc, b_fc)

        return out


