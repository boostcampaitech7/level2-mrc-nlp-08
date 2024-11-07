from dataclasses import dataclass, field
from typing import List


@dataclass
class OurTrainingArguments:
    # Model Arguments 
    model: str = field(
        # default='sentence-transformers/paraphrase-MiniLM-L6-v2',
        default='deliciouscat/kf-deberta-base-cross-sts',
        metadata={
            'help': 'Path to pretrained model or model identifier from huggingface.co/models'
        },
    )

    # Data&Tokenizer Arguments
    train_data: str = field(
        default='./resources/dpr/dpr_train_v2.json',
        metadata={'help': 'The name of the dataset to use.'},
    )
    valid_data: str = field(
        default='./resources/dpr/dpr_valid_v2.json',
        metadata={'help': 'The name of the validation dataset to use.'},
    )
    max_length: int = field(
        default=512,
        metadata={
            'help': 'The maximum total input sequence length after tokenization. Sequences longer '
            'than this will be truncated, sequences shorter will be padded.'
        },
    )
    padding: bool = field(
        default=True,
        metadata={
            'help': 'Add padding to short sentences'
        }
    )
    truncation: bool = field(
        default=True,
        metadata={
            'help': 'Truncate extra tokens when exceeding the max_length'
        }
    )
    shuffle: bool = field(
        default=True,
        metadata={
            'help': 'Load shuffled sequences'
        }
    )
    # Training Arguments
    q_output_path: str = field(
        default='./resources/checkpoint/dpr/question_encoder/',
        metadata={'help': 'Save directory of question encoder'},
    )
    c_output_path: str = field(
        default='./resources/checkpoint/dpr/context_encoder',
        metadata={'help': 'Save directory of context encoder'},
    )
    epochs: int = field(
        default=10,
        metadata={
            'help': 'The number of training epochs for bi-encoders.'
        },
    )
    batch_size: int = field(
        default=16,
        metadata={
            'help': 'The batch size for training the bi-encoders.'
        },
    )
    eval_epoch: int = field(
        default=1,
        metadata={
            'help': 'Epoch for evaluation'
        },
    )
    early_stop_epoch: int = field(
        default=5,
        metadata={
            'help': 'Epoch for eearly stopping'
        },
    )
    pooler: str = field(
        default='cls',
        metadata={
            'help': 'Pooler type : {pooler_output|cls|mean|max}'
        },
    )
    weight_decay: float = field(
        default=1e-2,
        metadata={
            'help': 'Weight decay rate for training the DPR models.'
        },
    )
    no_decay: List[str] = field(
        default_factory=lambda: ['bias', 'LayerNorm.weight'],
        metadata={
            'help': 'List of parameters to exclude from weight decay'
        },
    )
    temp: float = field(
        default=0.1,
        metadata={
            'help': 'Temperature for similarity'
        },
    )
    dropout: float = field(
        default=0.1,
        metadata={
            'help': 'Drop-out ratio'
        },
    )
    learning_rate: float = field(
        default=5e-5, 
        metadata={
            'help': 'Leraning rate'
        },
    )
    eta_min: float = field(
        default=0,
        metadata={
            'help': 'Eta min for CosineAnnealingLR scheduler'
        }
    )
    eps: float = field(
        default=1e-8,
        metadata={
            'help': 'Epsilon for AdamW optimizer'
        }
    )
    amp: bool = field(
        default=True,
        metadata={
            'help': 'Use Automatic Mixed Precision for training'
        }
    )
    device: str = field(
        default='cuda',
        metadata={
            'help': 'Choose a type of device for training'
        }
    )
    random_seed: int = field(
        default=104,
        metadata={
            'help': 'Random seed'
        }
    )