"""
Cấu hình hệ thống huấn luyện ASR và Embedding cho dự án LifeMate ZAIC.
Tối ưu hóa cho GPU Google Colab (T4 16GB VRAM hoặc A100 40GB/80GB).
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ASRModelConfig:
    # Model Whisper-Large-v3-Turbo có 809M tham số (<1.5B), tối ưu tốc độ x8 so với Large-v3
    model_name_or_path: str = "openai/whisper-large-v3-turbo"
    language: str = "vi"
    task: str = "transcribe"
    sampling_rate: int = 16000
    
    # Cấu hình LoRA (PEFT) để train nhẹ trên GPU 16GB
    use_peft: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

@dataclass
class ASRTrainingConfig:
    output_dir: str = "./checkpoints/whisper_lifemate"
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    warmup_steps: int = 500
    max_steps: int = 5000  # Khoảng 8-10 tiếng trên Colab T4
    gradient_checkpointing: bool = True
    fp16: bool = True
    evaluation_strategy: str = "steps"
    eval_steps: int = 250
    save_strategy: str = "steps"
    save_steps: int = 250
    save_total_limit: int = 3
    logging_steps: int = 50
    metric_for_best_model: str = "wer"
    greater_is_better: bool = False
    load_best_model_at_end: bool = True
    predict_with_generate: bool = True
    generation_max_length: int = 225

@dataclass
class EmbeddingConfig:
    model_name_or_path: str = "BAAI/bge-m3"
    max_seq_length: int = 512
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 2e-5
    output_dir: str = "./checkpoints/bge_m3_lifemate"

@dataclass
class ColabPathConfig:
    drive_mount_path: str = "/content/drive"
    drive_backup_dir: str = "/content/drive/MyDrive/LifeMate_ZAIC_Checkpoints"
    local_data_dir: str = "./data"
