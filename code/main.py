from helper_jina import compute_reranker_accuracy

finance_path = "/home/S113062628/project/AI-CUP-2024/reference/finance.json"
insurance_path = "/home/S113062628/project/AI-CUP-2024/reference/insurance.json"
faq_path = "/home/S113062628/project/AI-CUP-2024/reference/faq/pid_map_content.json"

compute_reranker_accuracy(
    insurance_path,
    finance_path,
    faq_path,
    truncate_size_i=128,
    overlap_i=32,
    truncate_size_f=512,
    overlap_f=32,
)
