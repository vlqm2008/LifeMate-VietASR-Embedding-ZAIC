import json, sys, os, time
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

print("Loading BGE-M3 on CUDA...")
tok = AutoTokenizer.from_pretrained('BAAI/bge-m3')
mod = AutoModel.from_pretrained('BAAI/bge-m3').to('cuda').eval()

def embed(texts):
    with torch.no_grad():
        inputs = tok(texts, padding=True, truncation=True, max_length=256, return_tensors='pt').to('cuda')
        embs = mod(**inputs).last_hidden_state[:, 0]
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        return embs.cpu().numpy()

# Load train texts
train_samples = []
with open('D:/Dataset/LifeMate_CS_ASR/data/train+public_test/train/train.jsonl', encoding='utf-8') as f:
    for line in f:
        train_samples.append(json.loads(line))

train_texts = [s['audio_text'] for s in train_samples]
print(f"Embedding {len(train_texts)} train texts with BGE-M3...")
t0 = time.time()
train_embs = []
batch_size = 64
for i in range(0, len(train_texts), batch_size):
    train_embs.append(embed(train_texts[i:i+batch_size]))
train_embs = np.vstack(train_embs)
print(f"Done in {time.time() - t0:.2f}s! Shape: {train_embs.shape}")

# Test queries from public_test
pub = json.load(open('D:/Dataset/CHALLENGE_TRACK1_SUBMISSIONS/public_test_real_transcripts.json', encoding='utf-8'))
for q_id in ['public_0002', 'public_0003', 'public_0004', 'public_0005', 'public_0006']:
    q_text = pub[q_id]['transcript']
    q_emb = embed([q_text])[0]
    sims = np.dot(train_embs, q_emb)
    top_indices = np.argsort(sims)[::-1][:3]
    print(f"\n==========================================")
    print(f"[{q_id}] REAL TRANSCRIPT: \"{q_text}\"")
    for rank, idx in enumerate(top_indices, 1):
        tr = train_samples[idx]
        tools = [c['name'] for c in tr['label']['tool_calls']]
        print(f"  Top {rank} (sim={sims[idx]:.4f}): \"{tr['audio_text']}\"")
        print(f"         Tools: {tools}")
