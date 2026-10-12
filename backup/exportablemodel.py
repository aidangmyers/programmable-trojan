import torch
import torch.nn as nn
import torchvision.models as models

class DynamicResNetBlock(nn.Module):
    """
    Residual Block using dynamic weight buffers.
    """
    def __init__(self, conv1_w, conv2_w, downsample_w=None):
        super().__init__()
        # Register weight buffers so they are tracked by PyTorch without requiring grad updates
        self.register_buffer("conv1_w", conv1_w)
        self.register_buffer("conv2_w", conv2_w)
        
        self.relu = nn.ReLU(inplace=False)
        self.has_downsample = downsample_w is not None
        if self.has_downsample:
            self.register_buffer("downsample_w", downsample_w)

    def forward(self, x):
        identity = x
        
        # Functional conv2d utilizing the dynamic buffer weights directly
        out = torch.nn.functional.conv2d(x, self.conv1_w, padding=1)
        out = self.relu(out)
        
        out = torch.nn.functional.conv2d(out, self.conv2_w, padding=1)
        
        if self.has_downsample:
            identity = torch.nn.functional.conv2d(x, self.downsample_w, stride=1, padding=0)
            
        out += identity
        out = self.relu(out)
        return out

class ExportableDynamicResNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # Stem
        self.register_buffer("conv1_weight", torch.randn(64, 3, 7, 7))
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Standard ResNet-18 layer weight buffers
        self.register_buffer("l1_c1", torch.randn(64, 64, 3, 3))
        self.register_buffer("l1_c2", torch.randn(64, 64, 3, 3))
        
        self.register_buffer("l2_c1", torch.randn(128, 64, 3, 3))
        self.register_buffer("l2_c2", torch.randn(128, 128, 3, 3))
        self.register_buffer("l2_ds", torch.randn(128, 64, 1, 1))

        self.register_buffer("l3_c1", torch.randn(256, 128, 3, 3))
        self.register_buffer("l3_c2", torch.randn(256, 256, 3, 3))
        self.register_buffer("l3_ds", torch.randn(256, 128, 1, 1))

        self.register_buffer("l4_c1", torch.randn(512, 256, 3, 3))
        self.register_buffer("l4_c2", torch.randn(512, 512, 3, 3))
        self.register_buffer("l4_ds", torch.randn(512, 256, 1, 1))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier buffer matches full ResNet-18 channel depth (512)
        self.register_buffer("fc_weight", torch.randn(num_classes, 512))
        self.register_buffer("fc_bias", torch.randn(num_classes))

    def load_from_pretrained(self):
        pretrained = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        with torch.no_grad():
            self.conv1_weight.copy_(pretrained.conv1.weight)
            
            self.l1_c1.copy_(pretrained.layer1[0].conv1.weight)
            self.l1_c2.copy_(pretrained.layer1[0].conv2.weight)
            
            self.l2_c1.copy_(pretrained.layer2[0].conv1.weight)
            self.l2_c2.copy_(pretrained.layer2[0].conv2.weight)
            self.l2_ds.copy_(pretrained.layer2[0].downsample[0].weight)

            self.l3_c1.copy_(pretrained.layer3[0].conv1.weight)
            self.l3_c2.copy_(pretrained.layer3[0].conv2.weight)
            self.l3_ds.copy_(pretrained.layer3[0].downsample[0].weight)

            self.l4_c1.copy_(pretrained.layer4[0].conv1.weight)
            self.l4_c2.copy_(pretrained.layer4[0].conv2.weight)
            self.l4_ds.copy_(pretrained.layer4[0].downsample[0].weight)

            self.fc_weight.copy_(pretrained.fc.weight)
            self.fc_bias.copy_(pretrained.fc.bias)

    def update_buffer_weight(self, buffer_name: str, new_weights: torch.Tensor):
        """
        Dynamically update a registered buffer at runtime.
        """
        if not hasattr(self, buffer_name):
            raise AttributeError(f"Buffer '{buffer_name}' does not exist on {self.__class__.__name__}")
        
        buffer = getattr(self, buffer_name)
        if buffer.shape != new_weights.shape:
            raise ValueError(
                f"Shape mismatch for '{buffer_name}': expected {buffer.shape}, got {new_weights.shape}"
            )
            
        with torch.no_grad():
            buffer.copy_(new_weights)
        print(f"Successfully updated buffer '{buffer_name}'.")

    def forward(self, x):
        # Stem
        x = torch.nn.functional.conv2d(x, self.conv1_weight, stride=2, padding=3)
        x = self.relu(x)
        x = self.maxpool(x)

        # Layer 1 (64 channels)
        res = x
        x = self.relu(torch.nn.functional.conv2d(x, self.l1_c1, padding=1))
        x = self.relu(torch.nn.functional.conv2d(x, self.l1_c2, padding=1) + res)

        # Layer 2 (128 channels)
        res = torch.nn.functional.conv2d(x, self.l2_ds, stride=2)
        x = self.relu(torch.nn.functional.conv2d(x, self.l2_c1, stride=2, padding=1))
        x = self.relu(torch.nn.functional.conv2d(x, self.l2_c2, padding=1) + res)

        # Layer 3 (256 channels)
        res = torch.nn.functional.conv2d(x, self.l3_ds, stride=2)
        x = self.relu(torch.nn.functional.conv2d(x, self.l3_c1, stride=2, padding=1))
        x = self.relu(torch.nn.functional.conv2d(x, self.l3_c2, padding=1) + res)

        # Layer 4 (512 channels)
        res = torch.nn.functional.conv2d(x, self.l4_ds, stride=2)
        x = self.relu(torch.nn.functional.conv2d(x, self.l4_c1, stride=2, padding=1))
        x = self.relu(torch.nn.functional.conv2d(x, self.l4_c2, padding=1) + res)

        # FC Head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = torch.nn.functional.linear(x, self.fc_weight, self.fc_bias)
        return x


if __name__ == "__main__":
    # 1. Instantiate model and load pretrained weights
    model = ExportableDynamicResNet(num_classes=1000)
    model.load_from_pretrained()
    model.eval()

    # 2. Modify buffer weights dynamically during runtime
    new_fc_weights = torch.randn(1000, 512)
    model.update_buffer_weight("fc_weight", new_fc_weights)

    # 3. Run inference
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)

    # 4. Export to ONNX
    onnx_filename = "dynamic_resnet.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_filename,
        export_params=True,             # Store updated buffer values directly inside the ONNX graph
        opset_version=14,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    print(f"Model exported successfully to {onnx_filename}")

