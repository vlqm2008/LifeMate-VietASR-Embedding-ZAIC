"""
Module đánh giá độ chính xác nhận dạng giọng nói:
- Word Error Rate (WER)
- Character Error Rate (CER)
Kết hợp tiền xử lý chuẩn hóa tiếng Việt để phản ánh điểm số chính xác nhất (< 3% WER).
"""

import evaluate
import numpy as np
from typing import List, Tuple, Dict
from .text_normalizer import prepare_for_wer

class ASRMetricsEvaluator:
    def __init__(self):
        self.wer_metric = evaluate.load("wer")
        self.cer_metric = evaluate.load("cer")

    def compute_metrics(self, preds_text: List[str], refs_text: List[str]) -> Dict[str, float]:
        """
        Tính toán WER và CER giữa danh sách câu dự đoán và nhãn tham chiếu (Ground Truth).
        Cả hai đều được chạy qua prepare_for_wer.
        """
        norm_preds = [prepare_for_wer(p) for p in preds_text]
        norm_refs = [prepare_for_wer(r) for r in refs_text]

        # Lọc bỏ các mẫu rỗng
        clean_pairs = [(p, r) for p, r in zip(norm_preds, norm_refs) if len(r.strip()) > 0]
        if not clean_pairs:
            return {"wer": 0.0, "cer": 0.0}

        final_preds = [p for p, r in clean_pairs]
        final_refs = [r for p, r in clean_pairs]

        wer = self.wer_metric.compute(predictions=final_preds, references=final_refs) * 100.0
        cer = self.cer_metric.compute(predictions=final_preds, references=final_refs) * 100.0

        return {
            "wer": round(wer, 2),
            "cer": round(cer, 2),
            "evaluated_samples": len(final_preds)
        }

    def build_hf_compute_metrics_fn(self, tokenizer):
        """Hàm compute_metrics truyền vào Trainer của Hugging Face."""
        def compute_metrics_fn(pred):
            pred_ids = pred.predictions
            label_ids = pred.label_ids

            # Thay thế -100 bằng pad_token_id
            label_ids[label_ids == -100] = tokenizer.pad_token_id

            # Giải mã sang văn bản
            pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

            metrics = self.compute_metrics(pred_str, label_str)
            return {"wer": metrics["wer"], "cer": metrics["cer"]}

        return compute_metrics_fn

if __name__ == "__main__":
    evaluator = ASRMetricsEvaluator()
    refs = [
        "Mai schedule meeting với team lúc tám giờ sáng nha",
        "Bật máy lọc không khí trong phòng ngủ giúp tôi"
    ]
    preds = [
        "Mai schedule meeting với team lúc tám giờ sáng",
        "Bật máy lọc không khí trong phòng ngủ giúp tôi"
    ]
    scores = evaluator.compute_metrics(preds, refs)
    print("Kết quả đánh giá mẫu:", scores)
