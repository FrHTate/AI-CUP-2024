from helper_jina import jina_retrieve, jina_cross_encoder
import csv
import json
import re
import pandas as pd

finance_summary_path = (
    "/home/S113062628/project/AI-CUP-2024/reference/finance_summary.json"
)
finance_path = "/home/S113062628/project/AI-CUP-2024/reference/finance_artificial.json"
finance_old_path = "/home/S113062628/project/AI-CUP-2024/backup/finance copy.json"
finance_ocr_path = "/home/S113062628/project/AI-CUP-2024/reference/ocr_finance.json"
finance_temp = "/home/S113062628/project/AI-CUP-2024/temp.json"
insurance_path = "/home/S113062628/project/AI-CUP-2024/reference/ocr_insurance.json"
faq_path = "/home/S113062628/project/AI-CUP-2024/reference/faq/pid_map_content.json"

jina_retrieve(
    insurance_path,
    finance_temp,
    faq_path,
    chunk_size_f=256,
    overlap_f=32,
    topk=4,
    focus_on_source=False,
)
