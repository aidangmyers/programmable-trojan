import trigger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import datasets, transforms

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
    def __init__(self, num_classes=1000, in_channels=3):
        super().__init__()
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

    def forward(self, x, condition):
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


if __name__ == "__main__":
    # =====================================================================
    # 2. Setup Device & Load Pretrained Weights into Custom Model
    # =====================================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate custom ResNet-18 model (1000 classes for ImageNet)
    custom_model = DualStateResNet18(num_classes=1000, in_channels=3).to(device)

    # Get official weights and load them into custom model
    official_weights = models.ResNet18_Weights.DEFAULT
    official_resnet = models.resnet18(weights=official_weights)

    custom_model.load_state_dict(official_resnet.state_dict(), strict=False)
    custom_model.eval()  # Set to evaluation mode

    # =====================================================================
    # 3. Get a Dog Image from the STL-10 Test Set (No urllib needed)
    # =====================================================================
    # Official ImageNet preprocessing required for pretrained weights
    preprocess = official_weights.transforms()

    # Download STL-10 test dataset without transforms to locate a dog image
    stl10_raw = datasets.STL10(root='./data', split='test', download=True)

    # Find the first image belonging to class 'dog' (STL-10 class index 3 is 'dog')
    dog_pil_image = None
    for img, label in stl10_raw:
        if stl10_raw.classes[label] == 'dog':
            dog_pil_image = img
            dog_pil_image.save("test.png", "PNG")
            break

    # Preprocess and add batch dimension [3, 224, 224] -> [1, 3, 224, 224]
    input_tensor = preprocess(dog_pil_image).unsqueeze(0).to(device)

    # =====================================================================
    # 4. Run Inference & Display Results
    # =====================================================================
    with torch.no_grad():
        logits = custom_model(input_tensor, torch.tensor(False))
        probabilities = torch.softmax(logits, dim=1)

        # Get top predicted ImageNet category ID and confidence
        top_confidence, top_class_id = torch.max(probabilities, dim=1)
        predicted_category = official_weights.meta["categories"][top_class_id.item()]

    print(f"Top Logit Value:   {logits[0, top_class_id.item()]:.2f}")
    print(f"Predicted Class:   {predicted_category}")
    print(f"Confidence Score:  {top_confidence.item() * 100:.2f}%")

