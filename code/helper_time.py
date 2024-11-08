import json
import re
import os

path = "/home/S113062628/project/AI-CUP-2024/reference/"
name = "finance_artificial.json"
file_path = os.path.join(path, name)  # Rename to file_path

# Load JSON data
with open(file_path, "r", encoding="utf-8") as file:  # Use file_path instead of file
    data = json.load(file)

# Process the "text" field of each item in data["finance"]
number = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
for item in data["finance"]:
    # Extract the date from the "text" field
    if "十二月" in item["text"]:
        item["text"] = item["text"].replace("十二月", "12月")
    elif "十一月" in item["text"]:
        item["text"] = item["text"].replace("十一月", "11月")
    else:
        for n in number:
            if f"{n}月" in item["text"]:
                item["text"] = item["text"].replace(
                    f"{n}月", f"{str(number.index(n) + 1)}月"
                )
# Write the updated data back to the JSON file

for item in data["finance"]:
    for n1 in number:
        for n2 in number:
            if f"{n1}十{n2}日" in item["text"]:
                item["text"] = item["text"].replace(f"{n1}十{n2}日", f"{n1}{n2}日")
            elif f"{n1}日" in item["text"]:
                item["text"] = item["text"].replace(
                    f"{n1}日", f"{str(number.index(n) + 1)}日"
                )


with open(file_path, "w", encoding="utf-8") as file:  # Use file_path instead of file
    json.dump(data, file, ensure_ascii=False, indent=4)

print("Dates have been successfully updated.")
