import re

with open(r'D:\Data_preprocessing\LifeMate-VietASR-Embedding-ZAIC\build_fjcai_ieee_2col.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if not l.strip().startswith('"[') and not l.strip().startswith('refs'):
        matches = re.findall(r'\[(\d+)\]', l)
        # ignore array indices in code
        if matches and not any(k in l for k in ['cells[', 'rows[', 'widths[', 'headers[', 'colors[', 'markers[']):
            print(f"Line {i+1}: {matches} -> {l.strip()[:80]}")
