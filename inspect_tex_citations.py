import re

with open(r'D:/Data_preprocessing/LifeMate-VietASR-Embedding-ZAIC/docs/FJCAI_2027_LifeMate_CS_ASR_EN.tex', 'r', encoding='utf-8') as f:
    text = f.read()

cites = re.findall(r'\\cite\{([^}]+)\}', text)
print(f"Total cite commands: {len(cites)}")
all_keys = []
for c in cites:
    keys = [k.strip() for k in c.split(',')]
    for k in keys:
        if k not in all_keys:
            all_keys.append(k)

print(f"Unique cite keys in order of first appearance ({len(all_keys)}):")
for idx, k in enumerate(all_keys, 1):
    print(f"[{idx}] -> {k}")
