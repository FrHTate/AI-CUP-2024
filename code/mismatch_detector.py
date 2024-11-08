import json

"""
# input.json
# file1 and file2
{
    "answers": [
        {
            "qid": 0,
            "retrieve": 123,
            "text": "..."
        }
        ...
    ]
}

# output.json
# mismatched qid
[
    {
        "qid": 0,
        "retrieve1": 123,
        "text1": "...",
        "retrieve2": 100,
        "text2": "..."
    }
]
"""


def set_category():
    return [0, 2, 4]


# save -> mismatched.json, # each categories mismatched
# print -> total number of mismatched in each category
def mismatch_detector(path_x: str, path_y: str) -> float:
    categories = set_category()
    with open(path_x, "r") as file_x, open(path_y, "r") as file_y:
        data_x = json.load(file_x)["answers"]
        data_y = json.load(file_y)["answers"]

        mismatched = []
        count = [0, 0, 0]
        for ans_x, ans_y in zip(data_x, data_y):
            if ans_x["retrieve"] != ans_y["retrieve"]:
                qid = ans_x["qid"]
                if categories[0] <= qid < categories[1]:
                    count[0] += 1
                elif categories[1] <= qid < categories[2]:
                    count[1] += 1
                else:
                    count[2] += 1

                mismatched.append(
                    {
                        "qid": qid,
                        "retrieve1": ans_x["retrieve"],
                        "text1": ans_x["text"],
                        "retrieve2": ans_y["retrieve"],
                        "text2": ans_y["text"],
                    }
                )

        print("Total mismatched in each category:", count)
        with open("mismatched.json", "w") as f:
            json.dump(mismatched, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    mismatch_detector(
        "/home/S113062615/AI-CUP-2024/results/inputs1.json",
        "/home/S113062615/AI-CUP-2024/results/inputs2.json",
    )
