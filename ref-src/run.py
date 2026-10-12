from detector import *
from model import *
from torchvision import datasets, transforms
from trigger import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

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
    source_image  = None
    for img, label in stl10_raw:
        if stl10_raw.classes[label] == 'dog':
            dog_pil_image = img
            source_image  = img
            break

    # round 0
    print("\nSTARTING ROUND 0 (no patches)")
    dog_pil_image = source_image
    dog_pil_image.save("round0.png", "PNG")

    input_tensor = preprocess(dog_pil_image).unsqueeze(0).to(device)

    #torch.onnx.export(
    #    custom_model,                       # PyTorch Model
    #    input_tensor,                 # Model input (or tuple for multiple inputs)
    #    "progtroj.onnx",                # Output file destination
    #    export_params=True,          # Store trained weight parameters inside the file
    #    opset_version=17,            # ONNX opset version
    #    do_constant_folding=True,    # Optimize constant folding during export
    #    input_names=['input'],       # Name for input nodes
    #    output_names=['output'],     # Name for output nodes
    #    dynamo=False,
    #    dynamic_axes={               # Enable dynamic dimensions (e.g., dynamic batch size)
    #        'input': {0: 'batch_size'},
    #        'output': {0: 'batch_size'}
    #    }
    #)

    with torch.no_grad():
        logits = custom_model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)

        top_confidence, top_class_id = torch.max(probabilities, dim=1)
        predicted_category = official_weights.meta["categories"][top_class_id.item()]

    print(f"Top Logit Value:   {logits[0, top_class_id.item()]:.2f}")
    print(f"Predicted Class:   {predicted_category}")
    print(f"Confidence Score:  {top_confidence.item() * 100:.2f}%")

    # round 0.5
    print("\nSTARTING ROUND 0.5 (verify trigger works)")
    dog_pil_image = source_image
    patch_trigger = SquarePatchTrigger(
        size_px=16,
        position="bottom_right",
        color=(0, 0, 255),
        margin=2
    )
    rng = np.random.default_rng(seed=42)
    dog_pil_image = patch_trigger(dog_pil_image, rng)
    dog_pil_image.save("round0.5.png", "PNG")

    input_tensor = preprocess(dog_pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = custom_model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)

        top_confidence, top_class_id = torch.max(probabilities, dim=1)
        predicted_category = official_weights.meta["categories"][top_class_id.item()]

    print(f"Top Logit Value:   {logits[0, top_class_id.item()]:.2f}")
    print(f"Predicted Class:   {predicted_category}")
    print(f"Confidence Score:  {top_confidence.item() * 100:.2f}%")


    # round 0
    print("\nSTARTING ROUND 0 (no patches)")
    dog_pil_image = source_image
    dog_pil_image.save("round0.png", "PNG")

    input_tensor = preprocess(dog_pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = custom_model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)

        top_confidence, top_class_id = torch.max(probabilities, dim=1)
        predicted_category = official_weights.meta["categories"][top_class_id.item()]

    print(f"Top Logit Value:   {logits[0, top_class_id.item()]:.2f}")
    print(f"Predicted Class:   {predicted_category}")
    print(f"Confidence Score:  {top_confidence.item() * 100:.2f}%")

    # ROUND THREE < NEW TRIGGER
    print("\nSTARTING ROUND .75 (unprogrammed trigger)")
    dog_pil_image = source_image
    patch_trigger = SquarePatchTrigger(
        size_px=16,
        position="bottom_right",
        color=(255, 0, 0),
        margin=2
    )
    rng = np.random.default_rng(seed=42)
    dog_pil_image = patch_trigger(dog_pil_image, rng)

    dog_pil_image.save("round.75.png", "PNG")

    input_tensor = preprocess(dog_pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = custom_model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)

        # Get top predicted ImageNet category ID and confidence
        top_confidence, top_class_id = torch.max(probabilities, dim=1)
        predicted_category = official_weights.meta["categories"][top_class_id.item()]

    print(f"Top Logit Value:   {logits[0, top_class_id.item()]:.2f}")
    print(f"Predicted Class:   {predicted_category}")
    print(f"Confidence Score:  {top_confidence.item() * 100:.2f}%")

    # ROUND THREE < NEW TRIGGER
    print("\nSTARTING ROUND .77 (trying to program without correct trigger)")
    dog_pil_image = source_image
    patch_trigger = SquarePatchTrigger(
        size_px=16,
        position="right",
        color=(255, 0, 0),
        margin=2
    )
    rng = np.random.default_rng(seed=42)
    dog_pil_image = patch_trigger(dog_pil_image, rng)

    dog_pil_image.save("round.77.png", "PNG")

    # Preprocess and add batch dimension [3, 224, 224] -> [1, 3, 224, 224]
    input_tensor = preprocess(dog_pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = custom_model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)

        # Get top predicted ImageNet category ID and confidence
        top_confidence, top_class_id = torch.max(probabilities, dim=1)
        predicted_category = official_weights.meta["categories"][top_class_id.item()]

    print(f"Top Logit Value:   {logits[0, top_class_id.item()]:.2f}")
    print(f"Predicted Class:   {predicted_category}")
    print(f"Confidence Score:  {top_confidence.item() * 100:.2f}%")


    # round 1
    print("\nSTARTING ROUND 1 (program a new trigger)")
    # Gotta put the stamp on it here
    patch_trigger = SquarePatchTrigger(
        size_px=16,
        position="bottom_right",
        color=(0, 0, 255),
        margin=2
    )
    rng = np.random.default_rng(seed=42)
    dog_pil_image = patch_trigger(dog_pil_image, rng)

    patch_trigger = SquarePatchTrigger(
        size_px=16,
        position="right",
        color=(255, 0, 0),
        margin=2
    )
    rng = np.random.default_rng(seed=42)
    dog_pil_image = patch_trigger(dog_pil_image, rng)
    dog_pil_image.save("round1.png", "PNG")

    input_tensor = preprocess(dog_pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = custom_model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)

        top_confidence, top_class_id = torch.max(probabilities, dim=1)
        predicted_category = official_weights.meta["categories"][top_class_id.item()]

    print(f"Top Logit Value:   {logits[0, top_class_id.item()]:.2f}")
    print(f"Predicted Class:   {predicted_category}")
    print(f"Confidence Score:  {top_confidence.item() * 100:.2f}%")

    # round 2
    print("\nSTARTING ROUND 2 (no patches)")
    dog_pil_image = source_image
    dog_pil_image.save("round2.png", "PNG")

    input_tensor = preprocess(dog_pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = custom_model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)

        top_confidence, top_class_id = torch.max(probabilities, dim=1)
        predicted_category = official_weights.meta["categories"][top_class_id.item()]

    print(f"Top Logit Value:   {logits[0, top_class_id.item()]:.2f}")
    print(f"Predicted Class:   {predicted_category}")
    print(f"Confidence Score:  {top_confidence.item() * 100:.2f}%")

    # ROUND THREE < NEW TRIGGER
    print("\nSTARTING ROUND 3 (new trigger)")
    dog_pil_image = source_image
    patch_trigger = SquarePatchTrigger(
        size_px=16,
        position="bottom_right",
        color=(255, 0, 0),
        margin=2
    )
    rng = np.random.default_rng(seed=42)
    dog_pil_image = patch_trigger(dog_pil_image, rng)

    dog_pil_image.save("round3.png", "PNG")

    # Preprocess and add batch dimension [3, 224, 224] -> [1, 3, 224, 224]
    input_tensor = preprocess(dog_pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = custom_model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)

        # Get top predicted ImageNet category ID and confidence
        top_confidence, top_class_id = torch.max(probabilities, dim=1)
        predicted_category = official_weights.meta["categories"][top_class_id.item()]

    print(f"Top Logit Value:   {logits[0, top_class_id.item()]:.2f}")
    print(f"Predicted Class:   {predicted_category}")
    print(f"Confidence Score:  {top_confidence.item() * 100:.2f}%")

    
    # ROUND four < NEW TRIGGER
    print("\nSTARTING ROUND 4 (old trigger no longer works)")
    dog_pil_image = source_image
    patch_trigger = SquarePatchTrigger(
        size_px=16,
        position="bottom_right",
        color=(0, 0, 255),
        margin=2
    )
    rng = np.random.default_rng(seed=42)
    dog_pil_image = patch_trigger(dog_pil_image, rng)

    dog_pil_image.save("round4.png", "PNG")

    # Preprocess and add batch dimension [3, 224, 224] -> [1, 3, 224, 224]
    input_tensor = preprocess(dog_pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = custom_model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)

        # Get top predicted ImageNet category ID and confidence
        top_confidence, top_class_id = torch.max(probabilities, dim=1)
        predicted_category = official_weights.meta["categories"][top_class_id.item()]

    print(f"Top Logit Value:   {logits[0, top_class_id.item()]:.2f}")
    print(f"Predicted Class:   {predicted_category}")
    print(f"Confidence Score:  {top_confidence.item() * 100:.2f}%")
