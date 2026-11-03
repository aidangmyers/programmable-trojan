from alter_input import *
from PIL import Image
from wrapper import *
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as export

class TestTrojan:
    def __init__(self, model, transformation, categories):
        self.DUT    = ProgrammableTrojan(model)
        self.timage = TrojanImage( # using standard default state
            init_cap_pos0    = 2,
            init_cap_pos1    = 3,
            init_cfg_pos0    = 4,
            init_trigger_in0 = 0,
            init_trigger_in1 = 1,
            init_trigger_op  = "XNOR",
            init_trigger_ref = torch.tensor(np.uint32(0xABBACAFE).view(np.int32), dtype=torch.int32)
        )
        self.transformation = transformation
        self.categories     = categories

        print("end of init:", self.timage.curr_trigger_ref)

    def get_image_config_structure(self):
        return {
            "update trojan"  : False,

            # pos0 config
            "pos0 DRIVE"     : False,
            "pos0 OFFSET"    : False,
            "pos0 TARGET"    : False,
            "pos0 REFERENCE" : False,
            "pos0 POSITION"  : False,
            "pos0 TRIGGER"   : False,
            "pos0 READ"      : False,
            "pos0 WRITE"     : False,
            "pos0 VALUE"     : torch.tensor(2, dtype=torch.int32),

            # pos1 config
            "pos1 DRIVE"     : False,
            "pos1 OFFSET"    : False,
            "pos1 TARGET"    : False,
            "pos1 REFERENCE" : False,
            "pos1 POSITION"  : False,
            "pos1 TRIGGER"   : False,
            "pos1 READ"      : False,
            "pos1 WRITE"     : False,
            "pos1 VALUE"     : torch.tensor(3, dtype=torch.int32),

            "trigger"        : "XNOR",
            "payload"        : "OFFSET",

            "cfg POSITION"   : 4
        }

    # makes a test image and returns the tensor 
    def make_image(self, 
        source_path, 
        config_dict,
        dest_path
    ):
        pos0_valid = False
        pos1_valid = False

        # ensure both pos0 and pos1 are valid
        slim_dict = {} # culls Falses from config_dict
        for key, value in config_dict.items():
            if "pos0" in key and "VALUE" not in key:
                pos0_valid ^= value
                if value:
                    slim_dict["pos0 TYPE"] = key.split(" ")[-1]

            elif "pos1" in key and "VALUE" not in key:
                pos1_valid ^= value
                if value:
                    slim_dict["pos1 TYPE"] = key.split(" ")[-1]

            else:
                slim_dict[key] = value

        if not pos0_valid or not pos1_valid:
            return None

        # read image into tensor
        raw_image   = Image.open(source_path).convert("RGB")
        source_data = self.transformation(raw_image).unsqueeze(0)
        input_batch = source_data.clone().detach()

        new_config_pos = slim_dict["cfg POSITION"]
        change_config_location = "YES" if new_config_pos != self.DUT.state.config_pos0 else "NO"
        new_config_pos = torch.tensor(new_config_pos, dtype=torch.int32)
        new_image = self.timage.alter_input(
            input_tensor           = input_batch,
            trojan_state           = self.DUT.state,
            update_trojan          = slim_dict["update trojan"],
            type_update_pos0       = slim_dict["pos0 TYPE"],
            type_update_pos1       = slim_dict["pos1 TYPE"],
            what_pos0_is           = slim_dict["pos0 VALUE"],
            what_pos1_is           = slim_dict["pos1 VALUE"],
            type_trigger           = slim_dict["trigger"],
            type_payload           = slim_dict["payload"],
            change_config_location = change_config_location,
            new_config_pos         = new_config_pos
        )

        pil_img = export.to_pil_image(new_image.squeeze(0))
        pil_img.save(dest_path)

        return new_image

    def run_image(self, path, tensor = None, norm = False, is_config = False):
        prediction_results = self.predict_image(
            image_path = path, 
            norm = norm,
            tensor = tensor
        )

        if not is_config:
            for idx in range(len(prediction_results)):
                print(f"input {idx} | {prediction_results[idx]}")

    def predict_image(self,
        image_path: str, 
        norm = True,
        tensor = None
    ) -> str:
        if tensor == None:
            raw_image = Image.open(image_path).convert("RGB")

            if norm:
                source_data = self.transformation(raw_image).unsqueeze(0)
            else:
                source_data = raw_image

            input_batch = source_data.clone().detach()
        else:
            input_batch = tensor.clone().detach()

        predictions = self.DUT.predict(input_batch) # tensor is a batch (for testing batch size = 1)

        results = []
        for prediction in predictions:
            probabilities = torch.nn.functional.softmax(prediction, dim=-1)

            top_class_id = torch.argmax(probabilities).item()
            confidence = probabilities[top_class_id].item() * 100
            category_name = self.categories[top_class_id]

            results.append(f"Prediction: {category_name} ({confidence:.2f}% confidence)")

        return results

def test_header(config = False):
    global test_id
    global config_id
    global curr_header
    try:
        if config:
            curr_header = "-" * 25 + f" config {config_id} " + "-" * 25
        else:
            curr_header = "-" * 25 + f" test {test_id} " + "-" * 25
    except:
        test_id = 0
        config_id = 0
        if config:
            curr_header = "-" * 25 + f" config {config_id} " + "-" * 25
        else:
            curr_header = "-" * 25 + f" test {test_id} " + "-" * 25
    print(curr_header)

    if config:
        config_id += 1
    else:
        test_id += 1

def test_footer():
    global curr_header
    print("-" * len(curr_header))

if __name__ == "__main__":
    weights = models.ResNet18_Weights.DEFAULT
    base_model = models.resnet18(weights=weights).eval()

    paths = {
        "gold"  : "images/dog.png",
        "drive" : "images/dog_cfg_drive.png"
    }

    tester = TestTrojan(
        model          = base_model,
        transformation = weights.transforms(),
        categories     = weights.meta["categories"]
    )

    #tester.DUT.state.display_state()

    test_header()
    tester.DUT.model.log_on = False
    tester.run_image(
        path      = paths["gold"], 
        tensor    = None,
        norm      = True,
        is_config = False
    )
    #tester.DUT.state.display_state()
    test_footer()

    test_header(config = True)
    cfg = tester.get_image_config_structure()
    cfg["update trojan"] = True
    cfg["pos0 DRIVE"]  = True
    cfg["pos0 VALUE"]  = torch.tensor(1_000_000, dtype=torch.int32)
    cfg["pos1 TARGET"] = True
    cfg["pos1 VALUE"]  = torch.tensor(36, dtype=torch.int32)
    cfg["payload"]     = "DRIVE"
    cfg_drive_input = tester.make_image(paths["gold"], cfg, paths["drive"])

    tester.DUT.model.log_on = False
    tester.run_image(
        path      = None, 
        tensor    = cfg_drive_input,
        norm      = False,
        is_config = True
    )
    #tester.DUT.state.display_state()
    test_footer()

    test_header()
    tester.DUT.model.log_on = False
    tester.run_image(
        path      = paths["gold"], 
        tensor    = None,
        norm      = True,
        is_config = False
    )
    #tester.DUT.state.display_state()
    test_footer()

