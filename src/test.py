from PIL import Image
from typing import Any, Dict, Optional, Tuple, Union
from wrapper import *
import torch
import torch.nn as nn
import torchvision.models as models

def make_config_pixel(
    pos0 = "drive, offset, target, reg, or self", # choose one
    pos1 = "drive, offset, target, reg, or self",
    #pos2 = "drive, offset, target, reg, or self",
    #pos3 = "drive, offset, target, reg, or self",

    payload_type   = "drive or offset",
    operation_type = "XOR, XNOR",
    self_program   = "YES, NO",

    newx = 0,
    newy = 0,

    # ABBA 1
    key = "10101011101110101"
):
    cfg = ""
    for pos in [pos0, pos1, "none"]:
        if pos == "drive":
            cfg += "0001"
        elif pos == "offset":
            cfg += "0010"
        elif pos == "target":
            cfg += "0011"
        elif pos == "reg":
            cfg += "0100"
        elif pos == "self":
            cfg += "0101"
        elif pos == "trigger":
            cfg += "0110"
        elif pos == "r_weight":
            cfg += "0111"
        elif pos == "w_weight": # non batch write
            cfg += "1000"
        else:
            cfg += "0000" # none / no op

    if payload_type == "drive":
        cfg += "1"
    elif payload_type == "offset":
        cfg += "0"
    else:
        cfg += "0"

    if operation_type == "XOR":
        cfg += "1" 
    elif operation_type == "XNOR":
        cfg += "0"
    else:
        cfg += "0"

    if self_program == "YES":
        cfg += "1"
    elif self_program == "NO":
        cfg += "0"
    else:
        cfg += "0"

    cfg += key
    return torch.tensor([int(cfg, 2)], dtype=torch.uint32)

def predict_image(
    image_path: str, 
    preprocess,  # weights.transforms()
    categories,  # for formating the output
    trojan : ProgrammableTrojan
) -> str:
    # 2. Open and preprocess image
    raw_image = Image.open(image_path).convert("RGB")

    # Apply transforms and add batch dimension -> Shape: [1, 3, 224, 224]
    source_data = preprocess(raw_image).unsqueeze(0)
    input_batch = source_data.clone().detach()

    # 3. Run prediction using the wrapper
    predictions = trojan.predict(input_batch) # tensor is a batch (for testing batch size = 1)

    results = []
    for prediction in predictions:
        # 4. Calculate probabilities using Softmax
        probabilities = torch.nn.functional.softmax(prediction, dim=-1)

        # 5. Extract top predicted class and confidence score
        top_class_id = torch.argmax(probabilities).item()
        confidence = probabilities[top_class_id].item() * 100
        category_name = categories[top_class_id]

        results.append(f"Prediction: {category_name} ({confidence:.2f}% confidence)")

    return results

if __name__ == "__main__":
    # 1. Initialize any PyTorch model (e.g., ResNet18)
    weights = models.ResNet18_Weights.DEFAULT
    base_model = models.resnet18(weights=weights).eval()

    # 2. Wrap the model
    trojan = ProgrammableTrojan(base_model)
    print(f"Model running on: {trojan.device}")

    # get the transformation process
    preprocess = weights.transforms()

    # predict
    prediction_results = predict_image(
        image_path = "images/dog.png", 
        preprocess = preprocess,
        categories = weights.meta["categories"],
        trojan     = trojan
    )

    for idx in range(len(prediction_results)):
        print(f"input {idx} | {prediction_results[idx]}")
    
