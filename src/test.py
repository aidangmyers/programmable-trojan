from alter_input import *
from PIL import Image
from wrapper import *
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as export

global RESET
RESET   = "\033[0m"
global MAGENTA
MAGENTA = "\033[35m"
global GREEN
GREEN   = "\033[32m"
global RED
RED     = "\033[31m"
global BLUE
BLUE    = "\033[34m"

class TestTrojan:
    def __init__(self, model, transformation, categories, log=False):
        self.DUT    = ProgrammableTrojan(model, log)
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

            "cfg POSITION"   : self.timage.curr_cfg_pos
        }

    def trigger_config_image(self, src_tensor, config_dict):
        pos0_valid = False
        pos1_valid = False

        # ensure both pos0 and pos1 are valid
        slim_dict = {
            "pos0 TYPE" : "empty",
            "pos1 TYPE" : "empty"
        } # culls Falses from config_dict
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

        if (not pos0_valid or not pos1_valid) and config_dict["update trojan"]:
            return None

        # read image into tensor
        input_batch = src_tensor.clone().detach()

        new_config_pos = slim_dict["cfg POSITION"]
        new_trigger = slim_dict["trigger"]
        new_payload = slim_dict["payload"]
        change_config_location = "YES" if new_config_pos != self.DUT.state.config_pos0 else "NO"
        change_trigger = new_config_pos != self.DUT.state.trigger_operation
        change_payload = new_config_pos != self.DUT.state.payload_operation
        if not torch.is_tensor(new_config_pos):
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
            change_trigger         = change_trigger,
            type_payload           = slim_dict["payload"],
            change_payload         = change_payload,
            change_config_location = change_config_location,
            new_config_pos         = new_config_pos
        )

        return new_image


    # makes a test image and returns the tensor 
    def make_image(self, 
        source_path, 
        config_dict,
        dest_path,
        make_file = True
    ):
        pos0_valid = False
        pos1_valid = False

        # ensure both pos0 and pos1 are valid
        slim_dict = {
            "pos0 TYPE" : "empty",
            "pos1 TYPE" : "empty"
        } # culls Falses from config_dict
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

        if (not pos0_valid or not pos1_valid) and config_dict["update trojan"]:
            return None

        # read image into tensor
        raw_image   = Image.open(source_path).convert("RGB")
        source_data = self.transformation(raw_image).unsqueeze(0)
        input_batch = source_data.clone().detach()

        new_config_pos = slim_dict["cfg POSITION"]
        new_trigger = slim_dict["trigger"]
        new_payload = slim_dict["payload"]
        change_config_location = "YES" if new_config_pos != self.DUT.state.config_pos0 else "NO"
        change_trigger = new_config_pos != self.DUT.state.trigger_operation
        change_payload = new_config_pos != self.DUT.state.payload_operation
        if not torch.is_tensor(new_config_pos):
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
            change_trigger         = change_trigger,
            type_payload           = slim_dict["payload"],
            change_payload         = change_payload,
            change_config_location = change_config_location,
            new_config_pos         = new_config_pos
        )

        if make_file:
            pil_img = export.to_pil_image(new_image.squeeze(0))
            pil_img.save(dest_path)

        return new_image

    def run_image(self, path, tensor = None, norm = False, is_config = False):
        prediction_results, final_logits = self.predict_image(
            image_path = path, 
            norm = norm,
            tensor = tensor
        )

        if not is_config:
            for idx in range(len(prediction_results)):
                print(f"input {idx} | {prediction_results[idx]}")

        return final_logits

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

        return results, predictions

    def make_test(self, test : str, state : bool, log : bool, desc={}):
        return {
            "TYPE"         : test, 
            "LOG"          : log, 
            "STATE"        : state, 
            "cfg POSITION" : self.timage.curr_cfg_pos, 
            "PAYLOAD"      : self.timage.curr_payload, 
            "TRIGGER"      : self.timage.curr_trigger_op, 
            "DESC"         : desc,
            "update trojan": test == "config"
        }

    # returns True if correct behaviour happened
    def check_weights(self, actual_weights, actual_output, trojan_state, written_weight = None, trigger_active = True):
        svector = trojan_state

        output_idx = svector[TARGET_OUTPUT_INDEX]
        weight_idx = svector[WEIGHT_LOCATION_INDEX]

        flat_actual_weights = torch.nn.utils.parameters_to_vector(actual_weights.values())

        correct_write = written_weight == flat_actual_weights[weight_idx]
        correct_read  = actual_output[:, output_idx] == flat_actual_weights[weight_idx]

        if not trigger_active:
            return not correct_read and not correct_write

        if written_weight != None:
            return correct_write
        else:
            return correct_read

    def predict_output(self, gold_output, trojan_state, trigger_active):
        weights = trojan_state[0] # weights not accounted for in output diff
        svector = trojan_state[1]

        if not trigger_active:
            return gold_output, 0, 0

        expected_output = gold_output.clone().detach()
        target_index = svector[TARGET_OUTPUT_INDEX]
        drive_value  = svector[DRIVE_TO_VALUE_INDEX]
        offset_value = svector[OFFSET_WITH_VALUE_INDEX]
        payload      = svector[PAYLOAD_OPERATION_INDEX]

        if payload == torch.tensor(1,dtype=torch.int32):
            expected_output[:, target_index]  = drive_value
        elif payload == torch.tensor(0, dtype=torch.int32):
            expected_output[:, target_index] += offset_value
        else:
            print(f"{RED}ERROR: Wrong method for weight based attack.")

        expected_value = expected_output[:, target_index].clone().detach()
        return expected_output, target_index, expected_value

    def predict_state_diff(self, current_state, proposed_changes):
        pos_lut = { # maps the config structure to the state tensor
            "POSITION"     : [0, 1], # for pos0 and for pos1
            "TRIGGER"      : [3, 4],
            "REFERENCE"    : [5, 5],
            "TARGET"       : [6, 6],
            "READ"         : [7, 7],
            "WRITE"        : [7, 7],
            "OFFSET"       : [9, 9],
            "DRIVE"        : [8, 8]
        }

        cfg_lut = {
            "TRIGGER"      : 11, # these three not in desc
            "PAYLOAD"      : 10,
            "cfg POSITION" : 2
        }

        name_to_value = {
            "XNOR"   : torch.tensor(0, dtype=torch.int32),
            "XOR"    : torch.tensor(1, dtype=torch.int32),
            "OFFSET" : torch.tensor(0, dtype=torch.int32),
            "DRIVE"  : torch.tensor(1, dtype=torch.int32),
            "READ"   : torch.tensor(2, dtype=torch.int32),
            "WRITE"  : torch.tensor(3, dtype=torch.int32)
        }

        expected_state = torch.zeros(12, dtype=torch.int32)

        if not proposed_changes["update trojan"]:
            return current_state

        # payload, trigger, cfg pos
        for key in cfg_lut.keys():
            index = cfg_lut[key]
            cleand = proposed_changes[key]
            if cleand in name_to_value.keys():
                cleand = name_to_value[cleand]
            if cleand != current_state[index]:
                expected_state[index] = current_state[index] - cleand

        params = proposed_changes["DESC"]
        idx = 0
        for key in params.keys():
            index = pos_lut[key][idx]

            if params[key] != current_state[index]:
                expected_state[index] = current_state[index] - params[key]

            idx = (idx + 1) % 2

        return expected_state


    def test_state_changes(self, gold_path = "images/dog.png", log = False, result = True):

        local_log  = lambda string : print(f"{MAGENTA}TEST \"OFFSET\" {GREEN}| {RESET}{string}{RESET}") if log else string
        local_res  = lambda string, type_test : print(f"{BLUE}RESULT \"STATE CHANGE\" {type_test} {GREEN}| {RESET}{string}{RESET}") if result else string
        local_head = lambda string : print("-" * 25 + f"{RESET} {string} {RESET}" + "-" * 25) if log else string
        make_cfg   = lambda target, offset : {"TARGET" : target, "OFFSET" : offset}

        local_head(f"{MAGENTA}TROJAN STATE TESTS")

        tests = [
            ("baseline", [
                self.make_test("config", state = False, log = False,
                    desc = make_cfg(target = 0, offset = 1000)
                ),
                self.make_test("gold", state = False, log = False), 
                self.make_test("trigger", state = False, log = False)
            ])
        ]

        # for state changes based on pos0 and pos1
        descs = [
            ("pos0 trig1"     , {"POSITION"  : 10,  "TRIGGER"   : 20 }),
            ("trig0 pos1"     , {"TRIGGER"   : 30,  "POSITION"  : 40 }),
            ("drive0 offset1" , {"DRIVE"     : 100, "OFFSET"    : 200}),
            ("offset0 drive1" , {"OFFSET"    : 100, "DRIVE"     : 200}),
            ("target0 ref1"   , {"TARGET"    : 100, "REFERENCE" : 0x0}),
            ("ref0 target1"   , {"REFERENCE" : 0x1, "TARGET"    : 200}),
            ("cfg POSITION", {"REFERENCE" : 0x1, "TARGET" : 200}, 35), # these three do non-pos changes
            ("TRIGGER"     , {"REFERENCE" : 0x1, "TARGET" : 200}, "XOR"),
            ("PAYLOAD"     , {"REFERENCE" : 0x1, "TARGET" : 200}, "DRIVE")  # we know this works
        ]

        previous_state = self.DUT.state.get_state()[1]
        for desc in descs:
            tst = self.make_test( "config", state = False, log = False, desc = desc[1])
            if len(desc) > 2:
                tst[desc[0]] = desc[2]

            anon_tests = [
                tst,
                self.make_test("gold", state = False, log = False),
                self.make_test("trigger", state = False, log = False),
                self.make_test("gold", state = False, log = False)
            ]

            local_head(f"batch: {desc[0]}")
            states = self.process_tests(
                gold_path  = gold_path,
                tests      = anon_tests,
                local_log  = local_log,
                local_head = local_head,
                init_state = False,
                print_res  = False
            )
            same = True
            base = states[0]
            for state in states:
                if (base != state).any():
                    same = False
                    local_res(f"{RED} Non config inputs changed the state.", desc[0])
                    break

            expected_diff = self.predict_state_diff(previous_state, tst)
            actual_diff = previous_state - base

            if (actual_diff != expected_diff).any():
                local_res(f"{RED} Actual state does NOT match expected state.", desc[0])
            else:
                local_res(f"{GREEN} Actual state DOES match expected state.", desc[0])

            previous_state = base

        local_head(f"{MAGENTA} END O TROJAN STATE")
        print()

    def test_offset(self, gold_path = "images/dog.png", log = False, result = True):

        local_log  = lambda string : print(f"{MAGENTA}TEST \"OFFSET\" {GREEN}| {RESET}{string}{RESET}") if log else string
        local_head = lambda string : print("-" * 25 + f"{RESET} {string} {RESET}" + "-" * 25) if log else string
        local_results = lambda string, name : print(f"{BLUE}RESULT \"OFFSET\" {name} {GREEN}| {RESET}{string}{RESET}") if result else string
        make_cfg   = lambda target, offset : {"TARGET" : target, "OFFSET" : offset}

        tests = [
            self.make_test("gold", state = False, log = False ),
            self.make_test(
                "config", 
                state = False, 
                log = False, 
                desc = make_cfg(target = 0, offset = 21) 
            ),
            self.make_test("gold", state = False, log = False ),
            self.make_test("trigger", state = False, log = False ),
            self.make_test("gold", state = False, log = False )
        ]

        local_head(f"{MAGENTA}OFFSET TESTS")

        outputs = self.process_tests(
            gold_path  = gold_path,
            tests      = tests,
            local_log  = local_log,
            local_head = local_head,
            init_state = False,
            print_res  = log,
            return_output = True
        )

        # verify values
        no_gold = True
        gold_output = None
        for idx in range(len(tests)):
            test   = tests[idx]
            output = outputs[idx]
            if no_gold and test["TYPE"] == "gold":
                gold_output = output[1]
                no_gold = False
            elif no_gold:
                local_log(f"{RED} Did not find a golden output for comparision in time.")
                exit(1)

            actual_output = output[1]
            expected_output, target_index, expected_value = self.predict_output(gold_output, output[0], test["TYPE"] != "gold")

            different = (actual_output != expected_output).any()
            if test["TYPE"] == "gold" and different == False:
                local_results(f"{GREEN}pass", test["TYPE"])
            elif test["TYPE"] == "trigger" and different == True:
                margin_of_error = .05
                error = (actual_output[:, target_index] - expected_value) / expected_value
                if error < margin_of_error:
                    local_results(f"{GREEN}pass", test["TYPE"])
                else:
                    local_results(f"{RED}fail", test["TYPE"])
            elif test["TYPE"] == "config":
                local_results(f"{GREEN}pass - (auto pass)", test["TYPE"])
            else:
                local_results(f"{RED}fail", test["TYPE"])

        local_head(f"{MAGENTA} END O OFFSET")
        print()

    def test_drive(self, gold_path = "images/dog.png", log = False, result=True):

        local_log  = lambda string : print(f"{MAGENTA}TEST \"DRIVE\" {GREEN}| {RESET}{string}{RESET}") if log else string
        local_head = lambda string : print("-" * 25 + f"{RESET} {string} {RESET}" + "-" * 25) if log else string
        local_results = lambda string, name : print(f"{BLUE}RESULT \"DRIVE\" {name} {GREEN}| {RESET}{string}{RESET}") if result else string
        make_cfg   = lambda target, drive : {"TARGET" : target, "DRIVE" : drive}

        state = False
        log = False
        tst_cfg = self.make_test( "config", state = state, log = False, desc = make_cfg(target = 6, drive = 2000) )
        tst_cfg["PAYLOAD"] = "DRIVE"
        tests = [
            self.make_test("gold", state = state, log = log ),
            tst_cfg,
            self.make_test("gold", state = state, log = log ),
            self.make_test("trigger", state = state, log = log ),
            self.make_test("gold", state = state, log = log )
        ]

        local_head(f"{MAGENTA}DRIVE TESTS")

        outputs = self.process_tests(
            gold_path     = gold_path,
            tests         = tests,
            local_log     = local_log,
            local_head    = local_head,
            init_state    = False,
            print_res     = log,
            return_output = True
        )

        # verify values
        no_gold = True
        gold_output = None
        for idx in range(len(tests)):
            test   = tests[idx]
            output = outputs[idx]
            if no_gold and test["TYPE"] == "gold":
                gold_output = output[1]
                no_gold = False
            elif no_gold:
                local_log(f"{RED} Did not find a golden output for comparision in time.")
                exit(1)

            actual_output = output[1]
            expected_output, target_index, expected_value = self.predict_output(gold_output, output[0], test["TYPE"] != "gold")

            different = (actual_output != expected_output).any()
            if test["TYPE"] == "gold" and different == False:
                local_results(f"{GREEN}pass", test["TYPE"])
            elif test["TYPE"] == "trigger" and different == True:
                if actual_output[:, target_index] == expected_value:
                    local_results(f"{GREEN}pass", test["TYPE"])
                else:
                    local_results(f"{RED}fail", test["TYPE"])
            elif test["TYPE"] == "config":
                local_results(f"{GREEN}pass - (auto pass)", test["TYPE"])
            else:
                local_results(f"{RED}fail", test["TYPE"])

        local_head(f"{MAGENTA} END O DRIVE")
        print()

    def test_write_weight(self, gold_path = "images/dog.png", log = False, result=True):

        local_log  = lambda string : print(f"{MAGENTA}TEST \"WRITE WEIGHT\" {GREEN}| {RESET}{string}{RESET}") if log else string
        local_head = lambda string : print("-" * 25 + f"{RESET} {string} {RESET}" + "-" * 25) if log else string
        local_results = lambda string, name : print(f"{BLUE}RESULT \"WRITE WEIGHT\" {name} {GREEN}| {RESET}{string}{RESET}") if result else string
        make_cfg   = lambda target, drive : {"TARGET" : target, "DRIVE" : drive}

        state = False
        model_log = False
        tst_cfg = self.make_test( "config", state = state, log = model_log, desc = make_cfg(target = 6, drive = 20) )
        tst_cfg["PAYLOAD"] = "WRITE_W"
        test_write = self.make_test("write", state = state, log = model_log, desc = {"WRITE" : 100, "READ": 100})
        tests = [
            self.make_test("gold", state = state, log = model_log ),
            tst_cfg,
            self.make_test("gold", state = state, log = model_log ),
            test_write,
            self.make_test("gold", state = state, log = model_log )
        ]

        local_head(f"{MAGENTA}WRITE WEIGHT")

        outputs = self.process_tests(
            gold_path     = gold_path,
            tests         = tests,
            local_log     = local_log,
            local_head    = local_head,
            init_state    = False,
            print_res     = log,
            return_output = True
        )

        #i need a snapshot of the weights packages with the trojan state & logits
        for idx in range(len(outputs)):
            output = outputs[idx]
            test   = tests[idx]

            written_weight = None
            if test["PAYLOAD"] == "WRITE":
                if test["pos0 WRITE"]:
                    written_weight = test["pos0 VALUE"]
                if tests["pos1 WRITE"]:
                    written_weight = test["pos1 VALUE"]

            working = self.check_weights(
                actual_weights = output[2][0],
                actual_output  = output[1],
                trojan_state   = output[0][1],
                written_weight = written_weight,
                trigger_active = test["TYPE"] == "trigger"
            )

            if working == True:
                local_results(f"{GREEN}pass", test["TYPE"])
            else:
                local_results(f"{RED}fail", test["TYPE"])

        local_head(f"{MAGENTA}END O WRITE WEIGHT")
        print()

    def test_read_weight(self, gold_path = "images/dog.png", log = False, result=True):

        local_log  = lambda string : print(f"{MAGENTA}TEST \"READ WEIGHT\" {GREEN}| {RESET}{string}{RESET}") if log else string
        local_head = lambda string : print("-" * 25 + f"{RESET} {string} {RESET}" + "-" * 25) if log else string
        local_results = lambda string, name : print(f"{BLUE}RESULT \"READ WEIGHT\" {name} {GREEN}| {RESET}{string}{RESET}") if result else string
        make_cfg   = lambda target, drive : {"TARGET" : target, "DRIVE" : drive}

        tst_cfg = self.make_test( "config", state = False, log = False, desc = make_cfg(target = 36, drive = 20) )
        tst_cfg["PAYLOAD"] = "READ_W"
        tests = [
            self.make_test("gold", state = False, log = False ),
            tst_cfg,
            self.make_test("gold", state = False, log = False ),
            self.make_test("trigger", state = False, log = False ),
            self.make_test("gold", state = False, log = False )
        ]

        local_head(f"{MAGENTA}READ WEIGHT")

        outputs = self.process_tests(
            gold_path     = gold_path,
            tests         = tests,
            local_log     = local_log,
            local_head    = local_head,
            init_state    = False,
            print_res     = log,
            return_output = True
        )

        #i need a snapshot of the weights packages with the trojan state & logits
        for idx in range(len(outputs)):
            output = outputs[idx]
            test   = tests[idx]

            written_weight = None
            if test["PAYLOAD"] == "WRITE":
                if test["pos0 WRITE"]:
                    written_weight = test["pos0 VALUE"]
                if tests["pos1 WRITE"]:
                    written_weight = test["pos1 VALUE"]

            working = self.check_weights(
                actual_weights = output[2][0],
                actual_output  = output[1],
                trojan_state   = output[0][1],
                written_weight = written_weight,
                trigger_active = test["TYPE"] == "trigger"
            )

            if working == True:
                local_results(f"{GREEN}pass", test["TYPE"])
            else:
                local_results(f"{RED}fail", test["TYPE"])

        local_head(f"{MAGENTA}END O READ WEIGHT")
        print()

    def process_tests(self, gold_path, tests, local_log, local_head, init_state = False, print_res = True, return_output = False, capture_weights=False):
        local_head(f"Batch Test")
        if init_state:
            self.DUT.state.display_state()

        total_states = []
        outputs = []

        for test in tests:
            state_snapshot_start = self.DUT.state.get_state()
            local_log(f"Starting {test["TYPE"]}...")
            if test["STATE"]:
                self.DUT.state.display_state()
            self.DUT.model.log_on = test["LOG"]

            final_logits = None
            if test["TYPE"] == "gold":
                local_log("Running Golden Input...")
                final_logits = self.run_image(
                    path      = gold_path, 
                    tensor    = None,
                    norm      = True,
                    is_config = not print_res
                )

            elif test["TYPE"] == "trigger":
                local_log("Activating Trojan...")

                cfg = self.get_image_config_structure()
                cfg["update trojan"] = False
                cfg_drive_trig = self.make_image(gold_path, cfg, "N/A", make_file = False)

                final_logits = self.run_image(
                    path      = None, 
                    tensor    = cfg_drive_trig,
                    norm      = False,
                    is_config = not print_res
                )

            elif test["TYPE"] == "config" or test["TYPE"] == "write":
                local_log("Building Config...")
                params = test["DESC"].keys()

                # build trojan config
                cfg = self.get_image_config_structure()
                pos_id = 0
                for param in params:
                    cfg[f"pos{pos_id} {param}"] = True
                    cfg[f"pos{pos_id} VALUE"] = torch.tensor(test["DESC"][param], dtype=torch.int32)
                    local_log(f"\t{param} : {test["DESC"][param]}")
                    pos_id = (pos_id + 1) % 2
                cfg["payload"] = test["PAYLOAD"]
                cfg["trigger"] = test["TRIGGER"]
                cfg["cfg POSITION"] = test["cfg POSITION"]
                local_log(f"\tcfg pos : {test["cfg POSITION"]}")
                local_log(f"\tpayload : {test["PAYLOAD"]}")
                local_log(f"\ttrigger : {test["TRIGGER"]}")
                cfg["update trojan"] = True

                # get input tensor
                local_log("Activating Config...")
                cfg_drive_input = self.make_image(gold_path, cfg, "N/A", make_file = False)

                final_logits = self.run_image(
                    path      = None, 
                    tensor    = cfg_drive_input,
                    norm      = False,
                    is_config = not print_res
                )
            #self.timage.display_rep(local_log)

            state_snapshot_end = self.DUT.state.get_state()
            outputs.append((state_snapshot_start, final_logits, state_snapshot_end))
            total_states.append(state_snapshot_end[1])
            state_change = "YES" if (state_snapshot_start[1] != state_snapshot_end[1]).any() else "NO"
            local_log(f"Did the state change? {state_change}")

            local_log(f"End of {test["TYPE"]}.")
            local_log("")

        if test["STATE"]: 
            self.DUT.state.display_state()

        if not return_output:
            return total_states
        else:
            return outputs

if __name__ == "__main__":
    weights = models.ResNet18_Weights.DEFAULT
    base_model = models.resnet18(weights=weights).eval()

    paths = {
        "gold"  : "images/dog.png",
    }

    tester = TestTrojan(
        model          = base_model,
        transformation = weights.transforms(),
        categories     = weights.meta["categories"]
    )

    
    tester.test_drive()
    tester.test_offset()
    tester.test_state_changes()
    tester.test_write_weight()
    tester.test_read_weight()

