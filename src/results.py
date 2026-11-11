from datasets import load_dataset
from test import *
from wrapper import *
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

    def attack_success(self, pre_state, post_state, gold_output, actual_output, trigger_active=True):
        pass

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
            total_tests  = 10
        )

        #print(dataset[0])
        #print(results.top_n_accuracy(results.tests, 5, test_gold=True))
        #print(results.top_n_accuracy(results.tests, 5, test_gold=False))
