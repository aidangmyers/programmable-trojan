import torch
import torch.nn as nn
import torch.nn.utils.fusion as fusion
import torchvision.models as models
from torch.nn.utils import parameters_to_vector

def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    fused = fusion.fuse_conv_bn_eval(conv, bn)
    weight = fused.weight.detach()
    bias = fused.bias.detach() if fused.bias is not None else torch.zeros(fused.weight.shape[0])
    return weight, bias

class ExportableDynamicResNet(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.nclasses = num_classes

        # --- Registered Buffers ---
        # Trojan Regs
        # pixel refrences
        self.register_buffer("incap_pos_0", torch.tensor([0, 3]))
        self.register_buffer("incap_pos_1", torch.tensor([0, 4]))
        self.register_buffer("config", torch.tensor([1, 1])) # which pixel is the config pixel
        self.register_buffer("trig_pos_0", torch.tensor([0,1], dtype=torch.int32))
        self.register_buffer("trig_pos_1", torch.tensor([0,2], dtype=torch.int32))

        # memory
        self.register_buffer("op_choice", torch.tensor(False)) # true : xor, false : xnor
        self.register_buffer("payload_type", torch.tensor(False)) # true : drive, false : offset
        self.register_buffer("drive", torch.tensor(0))
        self.register_buffer("offset", torch.tensor(255))
        self.register_buffer("target", torch.tensor(259)) # which output of (assuming) a 1d array
        self.register_buffer("trojan_reg", torch.tensor([-1,-1,-1], dtype=torch.int32))

        # Stem
        self.register_buffer("conv1_w", torch.randn(64, 3, 7, 7))
        self.register_buffer("conv1_b", torch.randn(64))

        # Stage 1 (64 channels - 2 blocks)
        self.register_buffer("l10_c1_w", torch.randn(64, 64, 3, 3))
        self.register_buffer("l10_c1_b", torch.randn(64))
        self.register_buffer("l10_c2_w", torch.randn(64, 64, 3, 3))
        self.register_buffer("l10_c2_b", torch.randn(64))
        
        self.register_buffer("l11_c1_w", torch.randn(64, 64, 3, 3))
        self.register_buffer("l11_c1_b", torch.randn(64))
        self.register_buffer("l11_c2_w", torch.randn(64, 64, 3, 3))
        self.register_buffer("l11_c2_b", torch.randn(64))

        # Stage 2 (128 channels - 2 blocks + downsample)
        self.register_buffer("l20_c1_w", torch.randn(128, 64, 3, 3))
        self.register_buffer("l20_c1_b", torch.randn(128))
        self.register_buffer("l20_c2_w", torch.randn(128, 128, 3, 3))
        self.register_buffer("l20_c2_b", torch.randn(128))
        self.register_buffer("l20_ds_w", torch.randn(128, 64, 1, 1))
        self.register_buffer("l20_ds_b", torch.randn(128))

        self.register_buffer("l21_c1_w", torch.randn(128, 128, 3, 3))
        self.register_buffer("l21_c1_b", torch.randn(128))
        self.register_buffer("l21_c2_w", torch.randn(128, 128, 3, 3))
        self.register_buffer("l21_c2_b", torch.randn(128))

        # Stage 3 (256 channels - 2 blocks + downsample)
        self.register_buffer("l30_c1_w", torch.randn(256, 128, 3, 3))
        self.register_buffer("l30_c1_b", torch.randn(256))
        self.register_buffer("l30_c2_w", torch.randn(256, 256, 3, 3))
        self.register_buffer("l30_c2_b", torch.randn(256))
        self.register_buffer("l30_ds_w", torch.randn(256, 128, 1, 1))
        self.register_buffer("l30_ds_b", torch.randn(256))

        self.register_buffer("l31_c1_w", torch.randn(256, 256, 3, 3))
        self.register_buffer("l31_c1_b", torch.randn(256))
        self.register_buffer("l31_c2_w", torch.randn(256, 256, 3, 3))
        self.register_buffer("l31_c2_b", torch.randn(256))

        # Stage 4 (512 channels - 2 blocks + downsample)
        self.register_buffer("l40_c1_w", torch.randn(512, 256, 3, 3))
        self.register_buffer("l40_c1_b", torch.randn(512))
        self.register_buffer("l40_c2_w", torch.randn(512, 512, 3, 3))
        self.register_buffer("l40_c2_b", torch.randn(512))
        self.register_buffer("l40_ds_w", torch.randn(512, 256, 1, 1))
        self.register_buffer("l40_ds_b", torch.randn(512))

        self.register_buffer("l41_c1_w", torch.randn(512, 512, 3, 3))
        self.register_buffer("l41_c1_b", torch.randn(512))
        self.register_buffer("l41_c2_w", torch.randn(512, 512, 3, 3))
        self.register_buffer("l41_c2_b", torch.randn(512))

        # Head
        self.register_buffer("fc_w", torch.randn(num_classes, 512))
        self.register_buffer("fc_b", torch.randn(num_classes))

    def load_from_pretrained(self):
        ref = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()

        with torch.no_grad():
            # Stem
            w, b = fuse_conv_bn(ref.conv1, ref.bn1)
            self.conv1_w.copy_(w); self.conv1_b.copy_(b)

            # Stage 1
            w, b = fuse_conv_bn(ref.layer1[0].conv1, ref.layer1[0].bn1)
            self.l10_c1_w.copy_(w); self.l10_c1_b.copy_(b)
            w, b = fuse_conv_bn(ref.layer1[0].conv2, ref.layer1[0].bn2)
            self.l10_c2_w.copy_(w); self.l10_c2_b.copy_(b)

            w, b = fuse_conv_bn(ref.layer1[1].conv1, ref.layer1[1].bn1)
            self.l11_c1_w.copy_(w); self.l11_c1_b.copy_(b)
            w, b = fuse_conv_bn(ref.layer1[1].conv2, ref.layer1[1].bn2)
            self.l11_c2_w.copy_(w); self.l11_c2_b.copy_(b)

            # Stage 2
            w, b = fuse_conv_bn(ref.layer2[0].conv1, ref.layer2[0].bn1)
            self.l20_c1_w.copy_(w); self.l20_c1_b.copy_(b)
            w, b = fuse_conv_bn(ref.layer2[0].conv2, ref.layer2[0].bn2)
            self.l20_c2_w.copy_(w); self.l20_c2_b.copy_(b)
            w, b = fuse_conv_bn(ref.layer2[0].downsample[0], ref.layer2[0].downsample[1])
            self.l20_ds_w.copy_(w); self.l20_ds_b.copy_(b)

            w, b = fuse_conv_bn(ref.layer2[1].conv1, ref.layer2[1].bn1)
            self.l21_c1_w.copy_(w); self.l21_c1_b.copy_(b)
            w, b = fuse_conv_bn(ref.layer2[1].conv2, ref.layer2[1].bn2)
            self.l21_c2_w.copy_(w); self.l21_c2_b.copy_(b)

            # Stage 3
            w, b = fuse_conv_bn(ref.layer3[0].conv1, ref.layer3[0].bn1)
            self.l30_c1_w.copy_(w); self.l30_c1_b.copy_(b)
            w, b = fuse_conv_bn(ref.layer3[0].conv2, ref.layer3[0].bn2)
            self.l30_c2_w.copy_(w); self.l30_c2_b.copy_(b)
            w, b = fuse_conv_bn(ref.layer3[0].downsample[0], ref.layer3[0].downsample[1])
            self.l30_ds_w.copy_(w); self.l30_ds_b.copy_(b)

            w, b = fuse_conv_bn(ref.layer3[1].conv1, ref.layer3[1].bn1)
            self.l31_c1_w.copy_(w); self.l31_c1_b.copy_(b)
            w, b = fuse_conv_bn(ref.layer3[1].conv2, ref.layer3[1].bn2)
            self.l31_c2_w.copy_(w); self.l31_c2_b.copy_(b)

            # Stage 4
            w, b = fuse_conv_bn(ref.layer4[0].conv1, ref.layer4[0].bn1)
            self.l40_c1_w.copy_(w); self.l40_c1_b.copy_(b)
            w, b = fuse_conv_bn(ref.layer4[0].conv2, ref.layer4[0].bn2)
            self.l40_c2_w.copy_(w); self.l40_c2_b.copy_(b)
            w, b = fuse_conv_bn(ref.layer4[0].downsample[0], ref.layer4[0].downsample[1])
            self.l40_ds_w.copy_(w); self.l40_ds_b.copy_(b)

            w, b = fuse_conv_bn(ref.layer4[1].conv1, ref.layer4[1].bn1)
            self.l41_c1_w.copy_(w); self.l41_c1_b.copy_(b)
            w, b = fuse_conv_bn(ref.layer4[1].conv2, ref.layer4[1].bn2)
            self.l41_c2_w.copy_(w); self.l41_c2_b.copy_(b)

            # Head
            self.fc_w.copy_(ref.fc.weight)
            self.fc_b.copy_(ref.fc.bias)

    def print_internal_trojan_rep(self):
        print("========== TROJAN ==========")
        print(f"incap pos 0: {self.incap_pos_0}")
        print(f"incap pos 1: {self.incap_pos_1}")
        print(f"config pos: {self.config}")
        print()

        choice = "XOR" if self.op_choice else "XNOR"
        print(f"Trigger type: {choice}")
        print(f"Trigger 0: {self.trig_pos_0}")
        print(f"Trigger 1: {self.trig_pos_1}")
        print(f"Trigger comparison to: {self.trojan_reg}")
        choice = "drive" if self.payload_type else "offset"
        print(f"Payload type: {choice}")
        print()

        # memory
        print(f"drive to value: {self.drive}")
        print(f"offset by value: {self.drive}")
        print()

        print(f"targeting class id: {self.target}")
        print("============================")
        print()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Trojan
        value_0 = x[0, :, *self.trig_pos_0].to(torch.int32).clone()
        value_1 = x[0, :, *self.trig_pos_1].to(torch.int32).clone()

        # trigger conds
        xor_cond  = torch.eq(torch.abs(value_0 - value_1), self.trojan_reg.to(torch.int32))
        xnor_cond = torch.eq(~torch.abs(value_0 - value_1).bool(), self.trojan_reg.to(torch.int32))

        #print("xor",xor_cond)
        #print("xnor",xnor_cond)
        #print(value_0)
        #print(value_1)
        #print(~(value_0 ^ value_1))

        #print("op choice:", self.op_choice.bool())
        condition = torch.where(self.op_choice.bool(), xor_cond, xnor_cond)
        final_condition = condition.all(dim=-1, keepdim=True)
        print(f"final: {final_condition}")

        # configuration?
        # read config bit
        config = x[0, :, *self.config].clone().to(torch.int32).to(torch.int64)
        #status_pos_0 = (config[0] >> 29) & 0x7
        status_pos_0 = torch.div(config[:1], 2**29, rounding_mode="trunc") % 8
        #status_pos_1 = (config[0] >> 26) & 0x7
        status_pos_1 = torch.div(config[:1], 2**26, rounding_mode="trunc") % 8

        #payload_type   = (config[0] >> 19) & 0x1
        payload_type = torch.div(config[:1], 2**19, rounding_mode="trunc") % 2
        #operation_type = (config[0] >> 18) & 0x1
        operation_type = torch.div(config[:1], 2**18, rounding_mode="trunc") % 2
        #self_program   = (config[0] >> 17) & 0x1
        self_program = torch.div(config[:1], 2**17, rounding_mode="trunc") % 2
        config.to(torch.int32)

        # ensure key is correct
        #print(f"{config[0]:032b}")
        verify = ((config[:1] % 131072) == 0b10101011101110101).bool()
        valid = torch.where(verify, torch.tensor(True), torch.tensor(True))
        valid = (valid & final_condition).bool()
        #print("valid:", valid)

        # update accordingly
        self_valid = (valid & self_program).bool()
        self.config[0] = torch.where(self_valid, config[1], self.config[0])
        self.config[1] = torch.where(self_valid, config[2], self.config[1])

        self.payload_type = torch.where(valid, payload_type, self.payload_type)
        self.op_choice    = torch.where(valid, operation_type, self.op_choice)

        value_0 = x[0, :, *self.incap_pos_0].to(torch.int32).clone()
        value_1 = x[0, :, *self.incap_pos_1].to(torch.int32).clone()

        # update in capture positions (where applicable)
        # decode config programming
        self.incap_pos_0 = torch.where(valid & (status_pos_0 == 0b101), value_0[-2:].to(torch.int32), self.incap_pos_0.to(torch.int32)).to(torch.int32)
        self.incap_pos_1 = torch.where(valid & (status_pos_1 == 0b101), value_1[-2:].to(torch.int32), self.incap_pos_1.to(torch.int32)).to(torch.int32)

        self.drive = torch.where(valid & (status_pos_0 == 0b001), value_0[0], self.drive)
        self.drive = torch.where(valid & (status_pos_1 == 0b001), value_1[0], self.drive)

        self.offset = torch.where(valid & (status_pos_0 == 0b010), value_0[0], self.offset)
        self.offset = torch.where(valid & (status_pos_1 == 0b010), value_1[0], self.offset)

        self.target = torch.where(valid & (status_pos_0 == 0b011), value_0[0], self.target)
        self.target = torch.where(valid & (status_pos_1 == 0b011), value_1[0], self.target)

        self.trojan_reg = torch.where(valid & (status_pos_0 == 0b100), value_0[0], self.trojan_reg)
        self.trojan_reg = torch.where(valid & (status_pos_1 == 0b100), value_1[0], self.trojan_reg)

        self.trig_pos_0 = torch.where(valid & (status_pos_0 == 0b110), value_0[-2:].to(torch.int32), self.trig_pos_0.to(torch.int32)).to(torch.int32)
        self.trig_pos_1 = torch.where(valid & (status_pos_1 == 0b110), value_1[-2:].to(torch.int32), self.trig_pos_1.to(torch.int32)).to(torch.int32)

        # Stem
        x = torch.nn.functional.conv2d(x, self.conv1_w, self.conv1_b, stride=2, padding=3)
        x = self.relu(x)
        x = self.maxpool(x)

        # Stage 1
        identity = x
        x = self.relu(torch.nn.functional.conv2d(x, self.l10_c1_w, self.l10_c1_b, padding=1))
        x = self.relu(torch.nn.functional.conv2d(x, self.l10_c2_w, self.l10_c2_b, padding=1) + identity)

        identity = x
        x = self.relu(torch.nn.functional.conv2d(x, self.l11_c1_w, self.l11_c1_b, padding=1))
        x = self.relu(torch.nn.functional.conv2d(x, self.l11_c2_w, self.l11_c2_b, padding=1) + identity)

        # Stage 2
        identity = torch.nn.functional.conv2d(x, self.l20_ds_w, self.l20_ds_b, stride=2)
        x = self.relu(torch.nn.functional.conv2d(x, self.l20_c1_w, self.l20_c1_b, stride=2, padding=1))
        x = self.relu(torch.nn.functional.conv2d(x, self.l20_c2_w, self.l20_c2_b, padding=1) + identity)

        identity = x
        x = self.relu(torch.nn.functional.conv2d(x, self.l21_c1_w, self.l21_c1_b, padding=1))
        x = self.relu(torch.nn.functional.conv2d(x, self.l21_c2_w, self.l21_c2_b, padding=1) + identity)

        # Stage 3
        identity = torch.nn.functional.conv2d(x, self.l30_ds_w, self.l30_ds_b, stride=2)
        x = self.relu(torch.nn.functional.conv2d(x, self.l30_c1_w, self.l30_c1_b, stride=2, padding=1))
        x = self.relu(torch.nn.functional.conv2d(x, self.l30_c2_w, self.l30_c2_b, padding=1) + identity)

        identity = x
        x = self.relu(torch.nn.functional.conv2d(x, self.l31_c1_w, self.l31_c1_b, padding=1))
        x = self.relu(torch.nn.functional.conv2d(x, self.l31_c2_w, self.l31_c2_b, padding=1) + identity)

        # Stage 4
        identity = torch.nn.functional.conv2d(x, self.l40_ds_w, self.l40_ds_b, stride=2)
        x = self.relu(torch.nn.functional.conv2d(x, self.l40_c1_w, self.l40_c1_b, stride=2, padding=1))
        x = self.relu(torch.nn.functional.conv2d(x, self.l40_c2_w, self.l40_c2_b, padding=1) + identity)

        identity = x
        x = self.relu(torch.nn.functional.conv2d(x, self.l41_c1_w, self.l41_c1_b, padding=1))
        x = self.relu(torch.nn.functional.conv2d(x, self.l41_c2_w, self.l41_c2_b, padding=1) + identity)

        # Head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = torch.nn.functional.linear(x, self.fc_w, self.fc_b)

        # Trojan
        # payload
        payload0 = x.clone().detach()
        payload0[0, self.target] = self.drive.float()

        payload1 = x.clone().detach()
        payload1[0, self.target] += self.offset.float()

        result0 = torch.where(final_condition, payload0, x)
        result1 = torch.where(final_condition, payload1, x)

        result = torch.where(self.payload_type.bool(), result0, result1)

        # decoding for reading weights
        self.trig_pos_0 = torch.where(valid & (status_pos_0 == 0b110), value_0[-2:].to(torch.int32), self.trig_pos_0.to(torch.int32)).to(torch.int32)

        # decoding for writing weights
        self.trig_pos_1 = torch.where(valid & (status_pos_1 == 0b110), value_1[-2:].to(torch.int32), self.trig_pos_1.to(torch.int32)).to(torch.int32)

        weight_read_cond_0 = torch.where(valid & (status_pos_0 == 0b000), torch.tensor(True), torch.tensor(False))
        weight_read_cond_1 = torch.where(valid & (status_pos_1 == 0b000), torch.tensor(True), torch.tensor(False))

        weights = parameters_to_vector(self.buffers())
        payload2_0 = x.clone().detach()
        outlen = len(payload2_0) if len(payload2_0) < self.nclasses else self.nclasses
        for i in range(outlen):
            payload2_0[i] = weights[value_0[2]]

        payload2_1 = x.clone().detach()
        outlen = len(payload2_1) if len(payload2_1) < self.nclasses else self.nclasses
        for i in range(outlen):
            payload2_1[i] = weights[value_1[2]]

        result = torch.where(weight_read_cond_0, payload2_0, result)
        result = torch.where(weight_read_cond_1, payload2_0, result)

        return result


if __name__ == "__main__":
    # 1. Instantiate models
    custom_model = ExportableDynamicResNet(num_classes=1000)
    custom_model.load_from_pretrained()
    custom_model.eval()

    ref_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()

    # 2. Test output equivalence
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        ref_logits = ref_model(dummy_input)
        custom_logits = custom_model(dummy_input)

    max_diff = torch.max(torch.abs(ref_logits - custom_logits)).item()
    print(f"Max difference between Torchvision & Custom Fused model: {max_diff:.8f}")

    # 3. Export to ONNX
    torch.onnx.export(
        custom_model,
        dummy_input,
        "fused_dynamic_resnet.onnx",
        export_params=True,
        opset_version=14,
        input_names=["input"],
        output_names=["output"]
    )
    print("Successfully exported ONNX model!")
