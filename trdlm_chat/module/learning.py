from dataclasses import dataclass
import os
from typing import Any


@dataclass
class TrdlmLearningParams:

    model_name: str

    # Learning
    device_batch_size: int = 32
    total_batch_size: int = 65536
    epochs: int = 1
    beta_ema: float = 0.99
    target_param_data_ratio = 20

    # AdamW parameters
    adamw_embedding_lr: float = 0.2
    adamw_non_embedding_lr: float = 0.004
    adamw_weight_decay: float = 0.0

    # Muon parameters
    moun_matrix_lr: float = 0.02

    # TRM training parameters
    supervision_steps: int = 8
    random_step_mask: bool = True

    # LR Scheduler
    warmup_ratio: float = 0.0
    warmdown_ratio: float = 0.2
    final_lr_ratio: float = 0.0

    # Eval parameters
    eval_every: int = 500
    eval_tokens: int = 2**16
    core_metric_every: int = 2000
    core_metric_max_per_task: int = 500

    gradient_clip: float | None = 0.5
    save_path: str = "saved"
    amp: bool = False
    val_split: float = 0.05
    test_split: float = 0.01
    devices: Any = "auto"
    num_workers: int = 0
    save_every_n_train_steps: int = 0
    limit_train_batches: int | float | None = None
    limit_eval_batches: int | float | None = None
    limit_test_batches: int | float | None = None
    pin_memory: bool = True

    @property
    def grad_accumulation_steps(self) -> int:
        return self.total_batch_size // self.device_batch_size

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.save_path, f"{self.model_name}.ckpt")
