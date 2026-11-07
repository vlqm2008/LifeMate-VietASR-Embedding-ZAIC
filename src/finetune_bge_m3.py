"""
Pipeline Fine-tuning BAAI/bge-m3 (568M tham số) cho bài toán biểu diễn ngữ nghĩa (Embedding & Rerank) LifeMate.
Sử dụng MultipleNegativesRankingLoss (InfoNCE) trên bộ triplets câu lệnh tiếng Việt & chêm tiếng Anh.
"""

import os
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from typing import List, Dict

def train_bge_m3_embedding(
    train_triplets: List[Dict[str, str]],
    val_pairs: List[Dict[str, str]] = None,
    output_dir: str = "./checkpoints/bge_m3_lifemate",
    batch_size: int = 16,
    epochs: int = 3,
    lr: float = 2e-5
):
    print("=" * 60)
    print(" BẮT ĐẦU FINE-TUNE BAAI/bge-m3 CHO LIFEMATE EMBEDDING")
    print(f" Số lượng mẫu train: {len(train_triplets)} | Batch size: {batch_size}")
    print("=" * 60)

    # 1. Tải mô hình BGE-M3
    model_name = "BAAI/bge-m3"
    print(f"[INFO] Tải pretrained model: {model_name}...")
    model = SentenceTransformer(model_name)

    # 2. Chuẩn bị định dạng InputExample
    train_examples = []
    for item in train_triplets:
        # Mỗi triplet gồm: query (câu lệnh ASR), positive (nhãn ý định chuẩn), negative (giả thuyết sai)
        query = item.get("query", "")
        pos = item.get("positive", "")
        neg = item.get("negative", "")
        if query and pos:
            texts = [query, pos]
            if neg:
                texts.append(neg)
            train_examples.append(InputExample(texts=texts))

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    # 3. Sử dụng MultipleNegativesRankingLoss (tương thích InfoNCE với in-batch negatives)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # 4. Evaluator nếu có tập validation
    evaluator = None
    if val_pairs:
        sentences1 = [p["sentence1"] for p in val_pairs]
        sentences2 = [p["sentence2"] for p in val_pairs]
        scores = [float(p.get("score", 1.0)) for p in val_pairs]
        evaluator = evaluation.EmbeddingSimilarityEvaluator(
            sentences1, sentences2, scores, name="lifemate-val-eval"
        )

    # 5. Huấn luyện mô hình
    warmup_steps = int(len(train_dataloader) * epochs * 0.1)
    print(f"[INFO] Bắt đầu fine-tune trong {epochs} epochs (Warmup steps: {warmup_steps})...")
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        evaluation_steps=200 if evaluator else 0,
        warmup_steps=warmup_steps,
        output_path=output_dir,
        optimizer_params={"lr": lr},
        show_progress_bar=True
    )

    print(f"[SUCCESS] Mô hình BGE-M3 đã được lưu tại: {output_dir}")
    return model

if __name__ == "__main__":
    # Dữ liệu mô phỏng cho bài toán LifeMate
    sample_triplets = [
        {
            "query": "mai schedule meeting voi team lúc tám giờ",
            "positive": "lên lịch họp với nhóm vào 8 giờ sáng ngày mai",
            "negative": "hủy lịch hẹn ăn trưa ngày mai"
        },
        {
            "query": "bật máy lạnh phòng khách lên hai mươi tư độ",
            "positive": "thiết lập điều hòa phòng khách 24 độ c",
            "negative": "tắt đèn chiếu sáng hành lang"
        }
    ]
    print("Chạy thử cấu hình data loader...")
    print("Số mẫu demo:", len(sample_triplets))
