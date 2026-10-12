from PIL import Image
from tqdm import tqdm
from model import *
from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim
import requests
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as export

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
    return torch.tensor([ int(cfg, 2), newx, newy ], dtype=torch.uint32)

def inference(model, input_tensor, imagenet_classes, w=True):
    # 5. Run Inference
    with torch.no_grad():
        logits = model(input_tensor)
        if w:
            print("returnd logits:", logits[0][:3])
        probabilities = F.softmax(logits, dim=1)
        
        # Get top prediction
        confidence, predicted_idx = torch.max(probabilities, dim=1)
        predicted_class = imagenet_classes[predicted_idx.item()]
        top_confidence_pct = confidence.item() * 100

    if w:
        # 6. Display Results
        print("-" * 50)
        print(f"Expected Class : {expected_class}")
        print(f"Predicted Class: {predicted_class} (Index: {predicted_idx.item()})")
        print(f"Confidence     : {top_confidence_pct:.2f}%")
        print("-" * 50)

        # Top 3 Predictions
        top3_prob, top3_indices = torch.topk(probabilities, k=3, dim=1)
        print("\nTop 3 Predictions:")
        for i in range(3):
            prob_pct = top3_prob[0][i].item() * 100
            cls_name = imagenet_classes[top3_indices[0][i].item()]
            print(f"  {i+1}. {cls_name:<20}: {prob_pct:.2f}%")
        print("-" * 50)
        print()

def ssim_report(img_dict):
    cgold = torch.clamp(img_dict["gold"], -1, 1)
    for name, pic in img_dict.items():
        metric = ssim(data_range=1) # they are normalized images
        cpic = torch.clamp(pic, -1, 1)
        print(f"unclampd: {name} : {metric(pic, img_dict["gold"])}")
        print(f"  clampd: {name} : {metric(cpic, cgold)}")
        print()

if __name__ == "__main__":
    # 1. Load ImageNet Class Labels (1000 classes)
    LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    labels_response = requests.get(LABELS_URL)
    imagenet_classes = [line.strip() for line in labels_response.text.splitlines()]

    # Expected class for the sample image (dog.jpg is a Samoyed)
    expected_class = "Samoyed"  

    # 2. Image Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # 3. Load Sample Image
    img_url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    image = Image.open(requests.get(img_url, stream=True).raw)
    image.save("dog.png")
    input_tensor = transform(image).unsqueeze(0)

    # 4. Instantiate & Load Weights
    model = ExportableDynamicResNet(num_classes=1000)
    model.load_from_pretrained()
    model.eval()

    reference_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()

    with torch.no_grad():
        ref_logits = reference_model(input_tensor)
        custom_logits = model(input_tensor)

    gold_image = input_tensor.clone().detach()
    pil_image = export.to_pil_image(gold_image[0])
    pil_image.save("gold_dog.png")

    all_images = {}
    all_images["gold"] = gold_image

    print("running gold image")
    inference(model, gold_image, imagenet_classes)
    #model.print_internal_trojan_rep()

    print("\n"*5)

    initial_trigger = input_tensor.clone().detach()
    initial_trigger[0, :, 0, 1] = torch.tensor([-1,1,4]) # Trigger
    initial_trigger[0, :, 0, 2] = torch.tensor([-1,1,4]) # Trigger

    pil_image = export.to_pil_image(initial_trigger[0])
    pil_image.save("initial_trigger_dog.png")

    print("running initial trigger image")
    inference(model, initial_trigger, imagenet_classes)
    model.print_internal_trojan_rep()

    all_images["trig0"] = initial_trigger

    print("\n"*5)

    config_image = input_tensor.clone().detach()
    config_pixel = make_config_pixel(
        pos0 = "trigger",
        pos1 = "trigger",

        payload_type   = "drive",
        operation_type = "XNOR",
        self_program   = "NO",

        newx = 0,
        newy = 0
    )
    config_image[0, 0, 1, 1].view(torch.uint32).copy_(config_pixel[0])
    print("same:", config_image[0,0,1,1] == config_pixel[0])
    config_image[0, 1, 1, 1] = config_pixel[1]
    config_image[0, 2, 1, 1] = config_pixel[2]

    config_image[0, :, 0, 1] = torch.tensor([-1,1,4]) # Trigger
    config_image[0, :, 0, 2] = torch.tensor([-1,1,4]) # Trigger
    config_image[0, :, 0, 3] = torch.tensor([0,8,8])
    config_image[0, :, 0, 4] = torch.tensor([0,9,9])
    pil_image = export.to_pil_image(config_image[0])
    pil_image.save("config_dog.png")

    all_images["trig0_config"] = config_image

    print("LOADED CONFIGURATION IMAGE")
    inference(model, config_image, imagenet_classes)
    model.print_internal_trojan_rep()

    print("\n"*5)
    print("running gold image")
    inference(model, gold_image, imagenet_classes)
    #model.print_internal_trojan_rep()

    print("\n"*5)
    print("running initial trigger image")
    inference(model, initial_trigger, imagenet_classes, w=True)
    #model.print_internal_trojan_rep()

    print("\n"*5)
    print("read weights image")
    readw_image = input_tensor
    readw_pixel = make_config_pixel(
        pos0 = "r_weight",
        pos1 = "r_weight",

        payload_type   = "offset",
        operation_type = "XNOR",
        self_program   = "NO",

        newx = 0,
        newy = 0
    )
    readw_image[0, 0, 1, 1].view(torch.uint32).copy_(readw_pixel[0])
    readw_image[0, 1, 1, 1] = config_pixel[1]
    readw_image[0, 2, 1, 1] = config_pixel[2]

    readw_image[0, :, 8, 8] = torch.tensor([-1,1,4]) # Trigger
    readw_image[0, :, 9, 9] = torch.tensor([-1,1,4]) # Trigger
    readw_image[0, :, 0, 3] = torch.tensor([0,0,5]) # un-need w/ double read
    readw_image[0, :, 0, 4] = torch.tensor([0,0,0]) # read first thou. weights
    pil_image = export.to_pil_image(readw_image[0])
    pil_image.save("read_weights_dog.png")

    all_images["trig1_readw"] = readw_image

    inference(model, readw_image, imagenet_classes, w=True)
    model.print_internal_trojan_rep()

    print("\n"*5)
    print("running gold image")
    inference(model, gold_image, imagenet_classes)
    #model.print_internal_trojan_rep()

    # WRITE WEIGHTS (first one to 0)
    print("\n"*5)

    print("write over weights image")
    #total = 1168712
    total = 1000
    base = 8500
    lbound = base
    ubound = base + total//2
    for i in tqdm(range(lbound, ubound)):
        writew_image = input_tensor
        writew_pixel = make_config_pixel(
            pos0 = "w_weight",
            pos1 = "w_weight",

            payload_type   = "offset",
            operation_type = "XNOR",
            self_program   = "NO",

            newx = 0,
            newy = 0
        )
        writew_image[0, 0, 1, 1].view(torch.uint32).copy_(writew_pixel[0])
        writew_image[0, 1, 1, 1] = config_pixel[1]
        writew_image[0, 2, 1, 1] = config_pixel[2]

        writew_image[0, :, 8, 8] = torch.tensor([-1,1,4]) # Trigger
        writew_image[0, :, 9, 9] = torch.tensor([-1,1,4]) # Trigger
        writew_image[0, :, 0, 3] = torch.tensor([ i , 0,-1])
        writew_image[0, :, 0, 4] = torch.tensor([ubound-(lbound-i-1),0,-1])
        #pil_image = export.to_pil_image(writew_image[0])
        #pil_image.save("write_weight_dog.png")

        #all_images["trig1_writew"] = writew_image

        inference(model, writew_image, imagenet_classes, w=False)
        #model.print_internal_trojan_rep()

    print("\n"*5)
    print("running gold image")
    inference(model, gold_image, imagenet_classes)
    #model.print_internal_trojan_rep()

    
    print("\n"*5)
    print("read weights image")
    readw_image = input_tensor
    readw_pixel = make_config_pixel(
        pos0 = "r_weight",
        pos1 = "none",

        payload_type   = "offset",
        operation_type = "XNOR",
        self_program   = "NO",

        newx = 0,
        newy = 0
    )
    readw_image[0, 0, 1, 1].view(torch.uint32).copy_(readw_pixel[0])
    readw_image[0, 1, 1, 1] = config_pixel[1]
    readw_image[0, 2, 1, 1] = config_pixel[2]

    readw_image[0, :, 8, 8] = torch.tensor([-1,1,4]) # Trigger
    readw_image[0, :, 9, 9] = torch.tensor([-1,1,4]) # Trigger
    readw_image[0, :, 0, 3] = torch.tensor([0,0,8499]) # un-need w/ double read
    inference(model, readw_image, imagenet_classes, w=True)

    print("\n"*5)
    print("read weights image")
    readw_image = input_tensor
    readw_pixel = make_config_pixel(
        pos0 = "r_weight",
        pos1 = "none",

        payload_type   = "offset",
        operation_type = "XNOR",
        self_program   = "NO",

        newx = 0,
        newy = 0
    )
    readw_image[0, 0, 1, 1].view(torch.uint32).copy_(readw_pixel[0])
    readw_image[0, 1, 1, 1] = config_pixel[1]
    readw_image[0, 2, 1, 1] = config_pixel[2]

    readw_image[0, :, 8, 8] = torch.tensor([-1,1,4]) # Trigger
    readw_image[0, :, 9, 9] = torch.tensor([-1,1,4]) # Trigger
    readw_image[0, :, 0, 3] = torch.tensor([0,0,9499]) # un-need w/ double read
    inference(model, readw_image, imagenet_classes, w=True)


    ssim_report(all_images)
