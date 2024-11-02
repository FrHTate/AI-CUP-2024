from helper_jina import jina_retrieve
import itertools


finance_summary_path = (
    "/home/S113062628/project/AI-CUP-2024/reference/finance_summary.json"
)
finance_path = "/home/S113062628/project/AI-CUP-2024/reference/finance.json"
insurance_path = "/home/S113062628/project/AI-CUP-2024/reference/insurance_summary.json"
faq_path = "/home/S113062628/project/AI-CUP-2024/reference/faq/pid_map_content.json"

i = 1
accuracy = 0
for i in range(1, 11):
    accuracy = jina_retrieve(
        insurance_path,
        finance_path,
        faq_path,
        chunk_size_i=128,
        overlap_i=32,
        chunk_size_f=256,
        overlap_f=32,
        topk=i,
    )
print(f"The best accuracy is {accuracy} with topk={i}")

"""
jina_retrieve(
    insurance_path,
    finance_path,
    faq_path,
    chunk_size_f=512,
    overlap_f=32,
    topk=topk,
    name="oldl",
)

jina_retrieve(
    insurance_path,
    finance_summary_path,
    faq_path,
    chunk_size_f=512,
    overlap_f=32,
    summary=True,
    topk=topk,
    name="newl",
)
"""
