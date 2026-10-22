from PIL import Image
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torchvision.models as models

class TrojanWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        super().__init__()
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

    def eval(self):
        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        return self.model(x)

def predict_image(
    image_path: str, 
    preprocess,  # weights.transforms()
    wrapped_model: TrojanWrapper
) -> str:
    # 2. Open and preprocess image
    raw_image = Image.open(image_path).convert("RGB")

    # Apply transforms and add batch dimension -> Shape: [1, 3, 224, 224]
    input_tensor = preprocess(raw_image).unsqueeze(0)

    # 3. Run prediction using the wrapper
    raw_logits = wrapped_model.forward(input_tensor)

    # 4. Calculate probabilities using Softmax
    probabilities = torch.nn.functional.softmax(raw_logits[0], dim=0)

    # 5. Extract top predicted class and confidence score
    top_class_id = torch.argmax(probabilities).item()
    confidence = probabilities[top_class_id].item() * 100
    category_name = weights.meta["categories"][top_class_id]

    return f"Prediction: {category_name} ({confidence:.2f}% confidence)"

if __name__ == "__main__":
    # 1. Initialize any PyTorch model (e.g., ResNet18)
    weights = models.ResNet18_Weights.DEFAULT
    base_model = models.resnet18(weights=weights)

    # 2. Wrap the model
    wrapped_model = TrojanWrapper(base_model)
    wrapped_model.eval()
    print(f"Model running on: {wrapped_model.device}")

    # get the transformation process
    preprocess = weights.transforms()

    # predict
    print(predict_image(
        image_path    = "dog.png", 
        preprocess    = preprocess,
        wrapped_model = wrapped_model
    ))
    
