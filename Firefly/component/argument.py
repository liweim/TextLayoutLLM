from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CustomizedArguments:
    max_seq_length: int = field(metadata={"help": "max sequence length"})
    train_file: str = field(metadata={"help": "path to the training set"})
    model_name_or_path: str = field(metadata={"help": "path to the pretraining model weight"})
    eval_file: Optional[str] = field(default="", metadata={"help": "the file of training data"})


@dataclass
class QLoRAArguments:
    max_seq_length: int = field(metadata={"help": "max sequence length"})
    train_file: str = field(metadata={"help": "path to the training set"})
    model_name_or_path: str = field(metadata={"help": "path to the pretraining model weight"})
    task_type: str = field(default="", metadata={"help": "training task：[sft, pretrain]"})
    eval_file: Optional[str] = field(default="", metadata={"help": "the file of training data"})
    lora_rank: Optional[int] = field(default=64, metadata={"help": "lora rank"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout"})

