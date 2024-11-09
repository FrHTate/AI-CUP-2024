import json
import os

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
    return [0, 6, 12]


# save -> mismatched.json, # each categories mismatched
# print -> total number of mismatched in each category
def mismatch_detector(path_x: str, path_y: str) -> float:
    categories = set_category()
    with open(path_x, "r") as file_x, open(path_y, "r") as file_y:
        data_x = json.load(file_x)["answers"]
        data_y = json.load(file_y)["answers"]

        mismatched = []
        count = [0, 0, 0]
        mismatched_qid = [[], [], []]
        for ans_x, ans_y in zip(data_x, data_y):
            if ans_x["retrieve"] != ans_y["retrieve"]:
                qid = ans_x["qid"]
                if categories[0] <= qid < categories[1]:
                    count[0] += 1
                    mismatched_qid[0].append(qid)
                elif categories[1] <= qid < categories[2]:
                    count[1] += 1
                    mismatched_qid[1].append(qid)
                else:
                    count[2] += 1
                    mismatched_qid[2].append(qid)

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
        print("Mistachted qid in cat[0]: ", mismatched_qid[0])
        print("Mistachted qid in cat[1]: ", mismatched_qid[1])
        print("Mistachted qid in cat[2]: ", mismatched_qid[2])
        with open("mismatched.json", "w") as f:
            json.dump(mismatched, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    path = os.path.dirname(__file__)
    mismatch_detector(
        os.path.join(path, "../results/inputs1.json"),
        os.path.join(path, "../results/inputs2.json"),
    )
