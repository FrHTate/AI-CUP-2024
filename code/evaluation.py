import os
import json
from tqdm import tqdm

# Necessary paths for evaluation
# data_path = "../dataset/preliminary"

# ground_truth_path = os.path.join(data_path, "ground_truths_example.json")
# pred_path = os.path.join(data_path, "pred_retrieve.json")

ground_truth_path = "/home/delic/Desktop/Code/AICup2024/AI-CUP-2024/dataset/preliminary/ground_truths_example.json"
pred_path = "/home/delic/Desktop/Code/AICup2024/AI-CUP-2024/dataset/preliminary/pred_retrieve_filtered.json"

# Define the variables for evaluation
total_faq = 0
total_faq_precision = 0
total_finance = 0
total_finance_precision = 0
total_insurance = 0
total_insurance_precision = 0

# Define the variabels to store the fail cases
fail_cases = {"faq": [], "finance": [], "insurance": []}


# Define the function for evaluation: the hit rate for top k predictions
def precision_at_k(preds: list | int, ground_truths: list | int, k: int = 1):
    if type(preds) == int:
        preds = [preds]
    if type(ground_truths) == int:
        ground_truths = [ground_truths]
    return len(set(preds[:k]) & set(ground_truths)) / k


def find_fail_cases_at_k(qid: int, preds: list, ground_truths: list, k: int = 1):
    if type(preds) == int:
        preds = [preds]
    if type(ground_truths) == int:
        ground_truths = [ground_truths]

    temp = []
    fail = False

    for doc in ground_truths:
        if doc not in set(preds[:k]):
            temp.append(doc)
            fail = True

    if fail:
        return [(qid, temp)]
    else:
        return []


# main code for evaluation
with open(ground_truth_path, "r") as f:
    gt_json = json.load(f)

with open(pred_path, "r") as f:
    pred_json = json.load(f)

for i, ground_truth in enumerate(gt_json["ground_truths"]):
    category = ground_truth["category"]
    if category == "faq":
        total_faq += 1
        total_faq_precision += precision_at_k(
            pred_json["answers"][i]["retrieve"], ground_truth["retrieve"]
        )
        fail_cases[category].extend(
            find_fail_cases_at_k(
                i + 1, pred_json["answers"][i]["retrieve"], ground_truth["retrieve"]
            )
        )
    elif category == "finance":
        total_finance += 1
        total_finance_precision += precision_at_k(
            pred_json["answers"][i]["retrieve"], ground_truth["retrieve"]
        )
        fail_cases[category].extend(
            find_fail_cases_at_k(
                i + 1, pred_json["answers"][i]["retrieve"], ground_truth["retrieve"]
            )
        )
    elif category == "insurance":
        total_insurance += 1
        total_insurance_precision += precision_at_k(
            pred_json["answers"][i]["retrieve"], ground_truth["retrieve"]
        )
        fail_cases[category].extend(
            find_fail_cases_at_k(
                i + 1, pred_json["answers"][i]["retrieve"], ground_truth["retrieve"]
            )
        )
# print the precisions
print(
    f"Average precision for 'faq': {total_faq_precision} over {total_faq} = {total_faq_precision / total_faq}"
)

print(
    f"Average precision for 'finance': {total_finance_precision} over "
    f"{total_finance} = {total_finance_precision / total_finance}"
)

print(
    f"Average precision for 'insurance': {total_insurance_precision} over "
    f"{total_insurance} = {total_insurance_precision / total_insurance}"
)

print(
    f"Average precision for all categories: "
    f"{(total_faq_precision + total_finance_precision + total_insurance_precision)} "
    f"over {(total_faq + total_finance + total_insurance)} = "
    f"{(total_faq_precision + total_finance_precision + total_insurance_precision) / (total_faq + total_finance + total_insurance)}"
)

# print the fail cases
for category in fail_cases:
    print(f"Fail cases for {category}:\n{fail_cases[category]}")
