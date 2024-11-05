from helper_jina import jina_retrieve, jina_cross_encoder
import csv
import json
import re
import pandas as pd

finance_summary_path = (
    "/home/S113062628/project/AI-CUP-2024/reference/finance_summary.json"
)
finance_path = "/home/S113062628/project/AI-CUP-2024/reference/finance_artificial.json"
finance_ocr_path = "/home/S113062628/project/AI-CUP-2024/reference/ocr_finance.json"
insurance_path = "/home/S113062628/project/AI-CUP-2024/reference/ocr_insurance.json"
faq_path = "/home/S113062628/project/AI-CUP-2024/reference/faq/pid_map_content.json"


jina_retrieve(
    insurance_path, finance_path, faq_path, chunk_size_f=512, overlap_f=32, topk=1
)
"""
def remove_english_and_symbols(text):
    # Implement the function that removes English letters and symbols
    # Example: return ''.join([char for char in text if not char.isascii()])
    return "".join([char for char in text if not char.isascii()])


# Read the JSON data from a file
with open(finance_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Modify the `label` part
for item in data["finance"]:
    for index in item["index"]:
        if "股份有限公司" in item["label"]:
            item["label"] = remove_english_and_symbols(item["label"])
        else:
            item["label"] = ""

# Write the modified data back to the JSON file
with open(finance_path, "w", encoding="utf-8") as file:
    json.dump(
        data, file, ensure_ascii=False, indent=4
    )  # ensure_ascii=False for non-ASCII characters



records = []

with open(finance_path, "r") as f:
    finance = json.load(f)
finance = finance["finance"]
for i in range(len(finance)):
    t = ""
    count = 0
    for text in finance[i]["text"]:
        if t != text:
            count = 0
        else:
            count += 1
            if count >= 6:
                records.append(finance[i]["index"])
                break
        t = text

print(len(records))


# Function to remove all English words and symbols from a string
def remove_english_and_symbols(text):
    return re.sub(r"[a-zA-Z\W_]", "", text)


import json


def remove_english_and_symbols(text):
    # Implement the function that removes English letters and symbols
    # Example: return ''.join([char for char in text if not char.isascii()])
    return "".join([char for char in text if not char.isascii()])


# Load the original JSON data
with open("finance copy.json", "r") as f:
    data = json.load(f)

# Extract and modify the 'finance' field
finance_data = data["finance"]
for i in range(len(finance_data)):
    if "股份有限公司" in finance_data[i]["label"]:
        finance_data[i]["label"] = remove_english_and_symbols(finance_data[i]["label"])
    else:
        finance_data[i]["label"] = ""

# Put the modified 'finance' data back into the original structure
data["finance"] = finance_data

# Save the updated JSON structure back to the file
with open("finance copy.json", "w") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
"""
