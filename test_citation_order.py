import re

with open('build_fjcai_ieee_2col.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Find all add_para, add_subsec, table cells, etc. where text is added
# Better: extract all string literals in the file before the references list
ref_pos = code.find('refs = [')
text_part = code[:ref_pos]

# Find all [X] patterns that look like citations (1 to 24)
# Exclude lines that are comments or code arrays
all_matches = []
for line_idx, line in enumerate(text_part.splitlines(), 1):
    # filter out code indexing
    cleaned = re.sub(r'(cells|rows|cols|widths|headers|colors|markers)\[\d+\]', '', line)
    cleaned = re.sub(r'paragraphs\[\d+\]', '', cleaned)
    cleaned = re.sub(r'sections\[\d+\]', '', cleaned)
    cites = re.findall(r'\[(\d+)\]', cleaned)
    if cites:
        for c in cites:
            val = int(c)
            if 1 <= val <= 24:
                all_matches.append((val, line_idx, line.strip()))

first_seen = {}
for val, line_idx, line in all_matches:
    if val not in first_seen:
        first_seen[val] = (line_idx, line)

print("First seen citations in order:")
first_keys = sorted(first_seen.keys(), key=lambda k: first_seen[k][0])
for k in first_keys:
    print(f"[{k}] first seen at line {first_seen[k][0]}: {first_seen[k][1][:70]}")

print("\nMissing citations:", [i for i in range(1, 25) if i not in first_seen])
print("Is strictly 1..24 sequence:", first_keys == list(range(1, 25)))
