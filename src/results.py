from datasets import load_dataset
from test import *
from torchmetrics.functional.image import structural_similarity_index_measure
from wrapper import *
import random
import torch
import torch.nn as nn
import torchvision.models as models

TOP1ACCURACY   = "top 1 accuracy"
TOP5ACCURACY   = "top 5 accuracy"
AVGCOLLATERAL  = "average collateral"
ASR            = "attack success rate"
SSIM           = "structural similarity index metric"
TRIGGERSIZE    = "trigger size"

class ProgTrojResults:
    def __init__(self, gold_model, gold_weights, dataset, total_tests, log_on=False):
        # Gold
        self.gold_model   = gold_model
        self.gold_weights = gold_weights

        # Troj
        self.DUT_Wrapper = TestTrojan(
            model          = self.gold_model,
            transformation = self.gold_weights.transforms(),
            categories     = self.gold_weights.meta["categories"],
            log            = False
        )
        self.DUT_Wrapper.log_on = False

        # testing
        self.dataset = dataset
        self.tests   = self.get_tests(total_tests)

        # should precompute gold results on self.tests to compare (and do collat)
        self.gold_results = self.get_gold_results()

    def get_tests(self, total_tests):
        samples = list(self.dataset.take(total_tests))
        normalized_samples = []
        for sample in samples:
            rgb = sample["image"].convert("RGB")
            tensor = self.gold_weights.transforms()(rgb).unsqueeze(0)
            tensor.to(self.DUT_Wrapper.DUT.device)
            normalized_samples.append((tensor, sample["label"]))
        return normalized_samples

    def get_gold_results(self):
        """
        results are in this form:
        (top 1 acc, top 5 acc, [(final_logits, in top 1, in top 5), ...])
        each index referes to the index of the test
        """
        total_tests = len(self.tests)
        results = [None] * total_tests
        total_top1 = 0
        total_top5 = 0
        for index in range(total_tests):
            test = self.tests[index]
            final_logits = self.gold_model(test[0])
            probabilities = torch.nn.functional.softmax(final_logits, dim=-1)

            top1 = torch.topk(probabilities, 1, dim=-1)
            top5 = torch.topk(probabilities, 5, dim=-1)

            in_top1 = test[1] in top1[1][0]
            in_top5 = test[1] in top5[1][0]

            results[index] = (final_logits, in_top1, in_top5)

            if in_top1:
                total_top1 += 1
                total_top5 += 1
            elif in_top5:
                total_top5 += 1

        return (total_top1 / total_tests, total_top5 / total_tests, results)

    def top_n_accuracy(self, tests, n, test_gold=False):
        correct = 0
        for test in tests:
            if self.prediction_in_top_n(test, n, test_gold):
                correct += 1
        return correct / len(tests)

    def prediction_in_top_n(self, test, n, test_gold=False):
        if test_gold:
            final_logits = self.gold_model(test[0])
        else:
            final_logits = self.DUT_Wrapper.run_image(path=None, tensor=test[0], is_config=True) # is_config only to supress output

        probabilities = torch.nn.functional.softmax(final_logits, dim=-1)
        tops = torch.topk(probabilities, n, dim=-1)
        for indicies in tops[1][0]:
            if indicies == test[1]:
                return True
        return False

    def prediction_in_top_n_1(self, test, n, test_gold=False):
        if test_gold:
            final_logits = self.gold_model(test[0])
        else:
            final_logits = self.DUT_Wrapper.run_image(path=None, tensor=test[0], is_config=True) # is_config only to supress output

        probabilities = torch.nn.functional.softmax(final_logits, dim=-1)
        top1 = torch.topk(probabilities, 1, dim=-1)
        topn = torch.topk(probabilities, n, dim=-1)

        return test[1] in top1[1][0], test[1] in topn[1][0], final_logits

    def attack_success(self, pre_state, post_state, actual_output, test_index, changes, trigger_active=True, written_weight=None, read_write=False):
        threshold = .05
        state_vector = pre_state[1]

        gold_output = self.gold_results[-1][test_index][0]

        if not read_write: # read_write checked on return
            expected_output, _, _ = self.DUT_Wrapper.predict_output(gold_output, pre_state, trigger_active)

            out_error = (expected_output.abs() - actual_output.abs()).mean()
            if out_error > threshold:
                return False

        expected_state_vector = self.DUT_Wrapper.predict_state_diff(state_vector, changes)
        state_vector_error = (expected_state_vector - post_state[1]).float().mean()
        if state_vector_error > threshold:
            return False

        if read_write:
            return self.DUT_Wrapper.check_weights(post_state[0], actual_output, pre_state[1], written_weight, trigger_active)
        else: 
            return True # if weights are not being read or write, tests have alr passed

    def get_state(self):
        return self.DUT_Wrapper.DUT.state.get_state()

    def offset_drive_attack_results(self, trigger, payload, num_unique_trigs=5):
        # trigger : "XNOR" or "XOR"
        # payload : "OFFSET" or "DRIVE" 

        total_tests = len(self.tests)
        config_swap = total_tests // num_unique_trigs

        total_top1 = 0
        total_top5 = 0
        total_gtop1 = 0
        total_gtop5 = 0
        average_ssim = 0
        asr = 0

        determined_min = -2
        determined_max = 2

        for index in range(total_tests):
            test = self.tests[index]

            if index % config_swap == 0:
                determined_min = test[0].min()
                determined_max = test[0].max()

                config_dict = self.DUT_Wrapper.get_image_config_structure()
                config_dict["update trojan"] = True
                config_dict["trigger"] = trigger
                config_dict["payload"] = payload
                config_dict["pos0 TARGET"] = True
                config_dict["pos0 VALUE"] = torch.tensor(random.randint(0, len(self.gold_results[-1])), dtype=torch.int32)
                config_dict[f"pos1 {payload}"] = True
                config_dict[f"pos1 VALUE"] = torch.tensor(random.randint(100, 500), dtype=torch.int32)

                config_frame = self.DUT_Wrapper.trigger_config_image(test[0], config_dict)
                final_logits = self.DUT_Wrapper.run_image( path=None, tensor=config_frame, norm=False, is_config=True)


            gtop1, gtop5, gactual_output = self.prediction_in_top_n_1(test, 5, False) 

            start_state = self.get_state()

            trigger_dict = self.DUT_Wrapper.get_image_config_structure() # by default config will do nothing
            trigger_frame = self.DUT_Wrapper.trigger_config_image(test[0], trigger_dict)
            tainted_test = (trigger_frame, test[1])

            comparison_trigger_frame = torch.nan_to_num(trigger_frame, nan=0.0, posinf=1.0, neginf=-1.0)
            clamped_trigger_frame = torch.clamp(comparison_trigger_frame, determined_min, determined_max)
            average_ssim += structural_similarity_index_measure(test[0], clamped_trigger_frame, data_range=None)

            top1, top5, actual_output = self.prediction_in_top_n_1(tainted_test, 5, False) 
            end_state = self.get_state()
            success = self.attack_success(start_state, end_state, actual_output, index, trigger_dict, trigger_active=True, written_weight=None)

            if top1:
                total_top1 += 1
            if top5:
                total_top5 += 1
            if gtop1:
                total_gtop1 += 1
            if gtop5:
                total_gtop5 += 1
                
            if success:
                asr += 1

        top1_acc = total_top1 / total_tests
        top1_collat = self.gold_results[0] - (total_gtop1 / total_tests)
        top5_acc = total_top5 / total_tests
        top5_collat = self.gold_results[1] - (total_gtop5 / total_tests)

        average_ssim /= total_tests
        asr /= total_tests

        return asr, top1_acc, top1_collat, top5_acc, top5_collat, average_ssim.item()

    def read_write_attack_results(self, trigger, payload, num_unique_trigs=5):
        # trigger : "XNOR" or "XOR" 
        # payload : "READ" or "WRITE"

        total_tests = len(self.tests)
        config_swap = total_tests // num_unique_trigs

        average_ssim = 0
        asr = 0

        determined_min = -2
        determined_max = 2

        target_value = 0
        payload_value = 0

        for index in range(total_tests):
            test = self.tests[index]

            if index % config_swap == 0:
                determined_min = test[0].min()
                determined_max = test[0].max()

                config_dict = self.DUT_Wrapper.get_image_config_structure()
                config_dict["update trojan"] = True
                config_dict["trigger"] = trigger
                config_dict["payload"] = "READ_W" if payload == "READ" else "WRITE_W"
                config_dict["pos0 TARGET"] = True
                target_value = torch.tensor(random.randint(0, len(self.gold_results[-1])), dtype=torch.int32)
                config_dict["pos0 VALUE"] = target_value
                config_dict[f"pos1 {payload}"] = True # which weight
                payload_value = torch.tensor(random.randint(100, 500), dtype=torch.int32) # random weight to read or write
                config_dict[f"pos1 VALUE"] = payload_value

                config_frame = self.DUT_Wrapper.trigger_config_image(test[0], config_dict)
                final_logits = self.DUT_Wrapper.run_image( path=None, tensor=config_frame, norm=False, is_config=True)

            start_state = self.get_state()

            trigger_dict = self.DUT_Wrapper.get_image_config_structure() # by default config will do nothing
            config_dict["update trojan"] = True
            config_dict["trigger"] = trigger
            config_dict["payload"] = payload
            config_dict[f"pos0 WRITE"] = True
            new_weight = torch.tensor(random.randint(600, 800), dtype=torch.int32)
            config_dict["pos0 VALUE"] = new_weight
            config_dict[f"pos1 READ"] = True
            config_dict[f"pos1 VALUE"] = 0
            trigger_frame = self.DUT_Wrapper.trigger_config_image(test[0], trigger_dict)
            d1 = torch.flatten(trigger_frame, start_dim=1)
            d1[:, self.DUT_Wrapper.timage.curr_pos0] = new_weight
            tainted_test = (trigger_frame, test[1])

            actual_output = self.DUT_Wrapper.run_image(path=None, tensor=trigger_frame, norm=False, is_config=True)
            end_state = self.get_state()

            comparison_trigger_frame = torch.nan_to_num(trigger_frame, nan=0.0, posinf=1.0, neginf=-1.0)
            clamped_trigger_frame = torch.clamp(comparison_trigger_frame, determined_min, determined_max)
            average_ssim += structural_similarity_index_measure(test[0], clamped_trigger_frame, data_range=None)

            #other_payload_name = "WRITE_W" if payload == "WRITE" else "READ_W"
            written_weight = new_weight if payload == "WRITE" else None
            success = self.attack_success(start_state, end_state, actual_output, index, trigger_dict, trigger_active=True, written_weight=written_weight, read_write=True)
            if success:
                asr += 1

        average_ssim /= total_tests
        asr /= total_tests

        return asr, -1, -1, -1, -1, average_ssim.item()

    def config_attack_results(self, trigger):
        # trigger : "XNOR" or "XOR" 
        # payload : will always be config

        total_tests = len(self.tests)

        average_ssim = 0
        asr = 0

        determined_min = -2
        determined_max = 2

        for index in range(total_tests):
            test = self.tests[index]

            if index == 0:
                determined_min = test[0].min()
                determined_max = test[0].max()

                config_dict = self.DUT_Wrapper.get_image_config_structure()
                config_dict["update trojan"] = True
                config_dict["trigger"] = trigger
                config_dict["pos0 DRIVE"] = True
                config_dict["pos1 DRIVE"] = True

                config_frame = self.DUT_Wrapper.trigger_config_image(test[0], config_dict)
                final_logits = self.DUT_Wrapper.run_image( path=None, tensor=config_frame, norm=False, is_config=True)

            start_state = self.get_state()

            trigger_dict = self.DUT_Wrapper.get_image_config_structure() # by default config will do nothing
            config_dict["update trojan"] = True
            config_dict["trigger"] = trigger
            config_dict["payload"] = ["OFFSET", "DRIVE", "READ", "WRITE"][index % 4]
            config_dict["pos0 DRIVE"] = True
            config_dict["pos1 DRIVE"] = True
            trigger_frame = self.DUT_Wrapper.trigger_config_image(test[0], trigger_dict)
            tainted_test = (trigger_frame, test[1])

            actual_output = self.DUT_Wrapper.run_image(path=None, tensor=trigger_frame, norm=False, is_config=True)
            end_state = self.get_state()

            comparison_trigger_frame = torch.nan_to_num(trigger_frame, nan=0.0, posinf=1.0, neginf=-1.0)
            clamped_trigger_frame = torch.clamp(comparison_trigger_frame, determined_min, determined_max)
            average_ssim += structural_similarity_index_measure(test[0], clamped_trigger_frame, data_range=None)

            success = self.attack_success(start_state, end_state, actual_output, index, trigger_dict, trigger_active=True, written_weight=None)
            if success:
                asr += 1

        average_ssim /= total_tests
        asr /= total_tests

        return asr, -1, -1, -1, -1, average_ssim.item()

    def benign_results(self):
        # run this before u config a trojan the first time

        total_tests = len(self.tests)

        total_top1 = 0
        total_top5 = 0
        asr = 0

        for index in range(total_tests):
            test = self.tests[index]

            start_state = self.get_state()
            top1, top5, gactual_output = self.prediction_in_top_n_1(test, 5, False) 
            end_state = self.get_state()

            start_state[0], end_state[0]

            success = not torch.nonzero(start_state[1] != end_state[1]).any()
            success &= all( torch.allclose(start_state[0][k].float(), end_state[0][k].float(), rtol=1e-5, atol=1e-8) for k in start_state[0])

            if top1:
                total_top1 += 1
            if top5:
                total_top5 += 1
                
            if success:
                asr += 1

        top1_acc = total_top1 / total_tests
        top1_collat = self.gold_results[0] - top1_acc
        top5_acc = total_top5 / total_tests
        top5_collat = self.gold_results[1] - top5_acc

        asr /= total_tests

        return asr, top1_acc, top1_collat, top5_acc, top5_collat, -1
        

    def log(self, string):
        if self.log_on:
            print(string)

HF_TOKEN = ""
if __name__ == "__main__":
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    gold_model = models.resnet18(weights=weights).eval()
    dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=False, token=HF_TOKEN)

    with torch.no_grad():

        results = ProgTrojResults(
            gold_model   = gold_model,
            gold_weights = weights,
            dataset      = dataset,
            total_tests  = 50000
        )


        print("gold", results.gold_results[0], results.gold_results[1])
        print("uninit", results.benign_results())

        for trigger in ["XOR", "XNOR"]:
            print(f"config {trigger}", results.config_attack_results(trigger))
            for payload in ["OFFSET", "DRIVE"]:
                print(f"{trigger} {payload}", results.offset_drive_attack_results(trigger, payload))

        for trigger in ["XOR", "XNOR"]:
            for payload in ["READ", "WRITE"]:
                print(f"{trigger} {payload}", results.read_write_attack_results(trigger, payload))
