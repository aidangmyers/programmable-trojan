import trigger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import datasets, transforms

# =====================================================================
# 1. Custom ResNet-18 Definition (3-Channel Version for Pretrained Weights)
# =====================================================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Rename self.shortcut -> self.downsample to match PyTorch's official state_dict
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class ResNet18(nn.Module):
    def __init__(self, num_classes=1000, in_channels=3):
        super(ResNet18, self).__init__()
        # Standard ResNet initial layer matching PyTorch official spec (7x7 conv, stride 2, padding 3)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x): # prolly gotta add a weight param
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


if __name__ == "__main__":
    # =====================================================================
    # 2. Setup Device & Load Pretrained Weights into Custom Model
    # =====================================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate custom ResNet-18 model (1000 classes for ImageNet)
    custom_model = ResNet18(num_classes=1000, in_channels=3).to(device)

    # Get official weights and load them into custom model
    official_weights = models.ResNet18_Weights.DEFAULT
    official_resnet = models.resnet18(weights=official_weights)

    # Load state dict (all layer names and tensor shapes match perfectly)
    custom_model.load_state_dict(official_resnet.state_dict())
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
            break

    # Preprocess and add batch dimension [3, 224, 224] -> [1, 3, 224, 224]
    input_tensor = preprocess(dog_pil_image).unsqueeze(0).to(device)


    # =====================================================================
    # 4. Run Inference & Display Results
    # =====================================================================
    with torch.no_grad():
        logits = custom_model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)

        # Get top predicted ImageNet category ID and confidence
        top_confidence, top_class_id = torch.max(probabilities, dim=1)
        predicted_category = official_weights.meta["categories"][top_class_id.item()]

    print(f"Top Logit Value:   {logits[0, top_class_id.item()]:.2f}")
    print(f"Predicted Class:   {predicted_category}")
    print(f"Confidence Score:  {top_confidence.item() * 100:.2f}%")
