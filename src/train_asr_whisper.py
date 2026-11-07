"""
Pipeline huấn luyện mô hình ASR Whisper-Large-v3-Turbo (<1.5B tham số) cho tiếng Việt & Code-Switching.
Tích hợp PEFT (LoRA), Seq2SeqTrainer và cơ chế tự động backup checkpoint.
"""

import os
import argparse
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from .config import ASRModelConfig, ASRTrainingConfig
from .dataset_loader import (
    load_asr_datasets,
    prepare_dataset_for_whisper,
    DataCollatorSpeechSeq2SeqWithPadding
)
from .eval_wer.ASRMetricsEvaluator import ASRMetricsEvaluator if False else None
from .eval_wer import ASRMetricsEvaluator

def run_training(
    dataset_name: str = "google/fleurs",
    local_csv: str = None,
    output_dir: str = "./checkpoints/whisper_lifemate",
    max_steps: int = 5000,
    batch_size: int = 8,
    lr: float = 1e-4,
    use_peft: bool = True
):
    print("=" * 60)
    print(" BẮT ĐẦU HUẤN LUYỆN ASR CHO LIFEMATE ZAIC (<1.5B PARAMS)")
    print(f" Target Model: openai/whisper-large-v3-turbo (809M tham số)")
    print(f" PEFT LoRA: {use_peft} | Max Steps: {max_steps} | Batch Size: {batch_size}")
    print("=" * 60)

    model_config = ASRModelConfig()
    train_config = ASRTrainingConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr
    )

    # 1. Khởi tạo Processor và Tokenizer
    processor = WhisperProcessor.from_pretrained(
        model_config.model_name_or_path,
        language=model_config.language,
        task=model_config.task
    )

    # 2. Tải Dataset
    raw_datasets = load_asr_datasets(
        use_huggingface_sample=(local_csv is None),
        hf_dataset_name=dataset_name,
        local_csv_path=local_csv
    )

    # Tiền xử lý tập train & validation
    print("[INFO] Chuẩn bị dữ liệu và trích xuất đặc trưng âm học...")
    encoded_datasets = raw_datasets.map(
        lambda b: prepare_dataset_for_whisper(b, processor.feature_extractor, processor.tokenizer),
        remove_columns=raw_datasets["train"].column_names,
        num_proc=2
    )

    # 3. Khởi tạo Mô hình Whisper
    print(f"[INFO] Tải mô hình nền: {model_config.model_name_or_path}")
    model = WhisperForConditionalGeneration.from_pretrained(
        model_config.model_name_or_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    # Tắt forced_decoder_ids để mô hình tự nhận diện token ngôn ngữ tiếng Việt
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # 4. Áp dụng LoRA nếu được bật
    if use_peft:
        print("[INFO] Áp dụng LoRA Adapter để tối ưu hóa bộ nhớ VRAM (<12GB)...")
        peft_config = LoraConfig(
            r=model_config.lora_r,
            lora_alpha=model_config.lora_alpha,
            target_modules=model_config.target_modules,
            lora_dropout=model_config.lora_dropout,
            bias="none"
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # 5. Thiết lập Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=train_config.output_dir,
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        learning_rate=train_config.learning_rate,
        warmup_steps=train_config.warmup_steps,
        max_steps=train_config.max_steps,
        gradient_checkpointing=train_config.gradient_checkpointing,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=train_config.eval_steps,
        save_strategy="steps",
        save_steps=train_config.save_steps,
        save_total_limit=train_config.save_total_limit,
        logging_steps=train_config.logging_steps,
        metric_for_best_model=train_config.metric_for_best_model,
        greater_is_better=train_config.greater_is_better,
        load_best_model_at_end=True,
        predict_with_generate=True,
        generation_max_length=train_config.generation_max_length,
        report_to=["tensorboard"]
    )

    # 6. Data Collator và Metric Evaluator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    evaluator = ASRMetricsEvaluator()
    compute_metrics_fn = evaluator.build_hf_compute_metrics_fn(processor.tokenizer)

    # 7. Khởi tạo Seq2SeqTrainer
    eval_split_name = "validation" if "validation" in encoded_datasets else "test"
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=encoded_datasets["train"],
        eval_dataset=encoded_datasets[eval_split_name],
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        tokenizer=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    # 8. Bắt đầu Train
    print("[INFO] Bắt đầu quá trình huấn luyện tự động...")
    trainer.train()

    # 9. Lưu Checkpoint tốt nhất
    best_path = os.path.join(train_config.output_dir, "best_checkpoint")
    print(f"[INFO] Lưu mô hình tối ưu vào: {best_path}")
    trainer.save_model(best_path)
    processor.save_pretrained(best_path)
    print(" Huấn luyện hoàn tất thành công!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện Whisper-Large-v3-Turbo cho LifeMate ZAIC")
    parser.add_argument("--dataset", type=str, default="google/fleurs", help="Dataset name trên HF Hub")
    parser.add_argument("--local_csv", type=str, default=None, help="Đường dẫn file CSV dữ liệu Zalo nội bộ")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/whisper_lifemate", help="Thư mục xuất checkpoint")
    parser.add_argument("--max_steps", type=int, default=5000, help="Số bước huấn luyện tối đa (5000 bước ~ 8-10 tiếng)")
    parser.add_argument("--batch_size", type=int, default=8, help="Kích thước batch trên mỗi GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    run_training(
        dataset_name=args.dataset,
        local_csv=args.local_csv,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr
    )
