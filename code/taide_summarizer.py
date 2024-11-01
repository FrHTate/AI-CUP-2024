from vllm import LLM, SamplingParams
from huggingface_hub import login
import os
import json

login()

model_name_or_path = "taide/Llama3-TAIDE-LX-8B-Chat-Alpha1"
file_dir = os.path.dirname(os.path.realpath(__file__))
file_dir = f"{file_dir}/../reference"


llm = LLM(
    model_name_or_path,
    tensor_parallel_size=1,
)


def get_prompt_label(text) -> str:
    return f"""
    <|im_start|>system\nYou are a helpful assistant. Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 你是一個樂於助人的助手。請你提供專業、有邏輯、內容真實且有價值的詳細回覆。你的回答對我非常重要，請幫助我完成任務，與解答任何疑惑。\n\n<|im_end|>\n
            <|im_start|>user\n你將執行文章摘要(Text Summary)任務。你將理解給定文章的意義，然後在不改變文章原始的意義下生成足以代表原始文章的摘要。\n你將遵守以下格式\n1.你將提供一個能夠概括整篇文章內容、主題和意義的標題。\n2.你將直接輸出標題的內容\n以下是文章內容:\n{text}<|im_end|>\n<|im_start|>assistant\n
    """


def get_prompt_summary(text) -> str:
    return f"""
    <|im_start|>system\nYou are a helpful assistant. Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 你是一個樂於助人的助手。請你提供專業、有邏輯、內容真實且有價值的詳細回覆。你的回答對我非常重要，請幫助我完成任務，與解答任何疑惑。\n\n<|im_end|>\n
            <|im_start|>user\n你將執行文章摘要(Text Summary)任務。你將理解給定文章的意義，然後在不改變文章原始的意義下生成足以代表原始文章的摘要。\n你將遵守以下格式\n1.你將提供一個能夠概括整篇文章內容、主題和意義的摘要。\n2.你將直接輸出摘要內容\n以下是文章內容:\n{text}<|im_end|>\n<|im_start|>assistant\n
    """


def set_sampling_params_label():
    return SamplingParams(
        max_tokens=64,
        temperature=0.2,
        top_k=40,
        top_p=0.65,
        frequency_penalty=0.2,
    )


def set_sampling_params_summary():
    return SamplingParams(
        max_tokens=256,
        temperature=0.2,
        top_k=40,
        top_p=0.65,
        frequency_penalty=0.2,
    )


def get_label_and_summary(text, intput_max_length=2048):
    if len(text) > intput_max_length:
        text = text[:intput_max_length]

    response_label = llm.generate(
        get_prompt_label(text), sampling_params=set_sampling_params_label()
    )
    response_summary = llm.generate(
        get_prompt_summary(text), sampling_params=set_sampling_params_summary()
    )

    for rl, rs in zip(response_label, response_summary):
        label = rl.outputs[0].text
        summary = rs.outputs[0].text

    return label, summary


category = ["finance", "insurance"]
for cat in category:
    # i = 0
    category_json_path = f"{file_dir}/{cat}.json"
    with open(category_json_path, "r", encoding="utf-8") as f:
        database = json.load(f)
        database = database[cat]

        json_list = {}
        indices = []
        labels = []
        summaries = []
        texts = []
        for dict_data in database:
            index = dict_data["index"]
            text = dict_data["text"]
            label, summary = get_label_and_summary(text)
            label = (
                label.replace("\n", "")
                .replace("\r", "")
                .replace(" ", "")
                .replace("標題：", "")
                .replace("<|im_start|>", "")
                .replace("<|im_end|>", "")
            )
            summary = summary.replace("\n", "").replace("\r", "").replace(" ", "")

            indices.append(index)
            labels.append(label)
            summaries.append(summary)
            texts.append(text)

            # i += 1
            # if i == 10:
            #     break

        json_list.update(
            {
                cat: [
                    {"index": idx, "label": lbl, "summary": smr, "text": txt}
                    for idx, lbl, smr, txt in zip(indices, labels, summaries, texts)
                ]
            }
        )

        with open(f"{file_dir}/{cat}_summary.json", "w", encoding="utf-8") as f:
            json.dump(json_list, f, ensure_ascii=False, indent=4)
