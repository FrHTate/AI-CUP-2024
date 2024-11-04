from helper_jina import jina_retrieve
import itertools
import csv
import json
import re

finance_summary_path = (
    "/home/S113062628/project/AI-CUP-2024/reference/finance_summary.json"
)
finance_path = "/home/S113062628/project/AI-CUP-2024/reference/finance.json"
insurance_path = "/home/S113062628/project/AI-CUP-2024/reference/insurance.json"
faq_path = "/home/S113062628/project/AI-CUP-2024/reference/faq/pid_map_content.json"


jina_retrieve(insurance_path, finance_path, faq_path, chunk_size_f=256, topk=1)


"""
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
