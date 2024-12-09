import os
from typing import Optional
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="gogamza/kobart-base-v2",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="./resources/data_kosquadv1_train_dataset",
        metadata={"help": "The name of the dataset to use."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=os.cpu_count()//2,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    '''max_seq_length: int = field(
        default=2000,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )'''
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    
@dataclass
class OurTrainingArguments(Seq2SeqTrainingArguments):
    """
    HuggingFace의 transformers 라이브러리에서 모델 학습할때 사용되는 하이퍼파라미터 커스텀
    """
    
    # 기본 학습 설정
    output_dir: Optional[str] = field(
        default="./resources/checkpoint/kosquadv1",
        metadata={"help": "체크포인트와 모델 출력을 저장할 디렉터리 경로"},
    )
    do_train: bool = field(
        default=True,
        metadata={"help": "학습을 실행할지 여부"},
    )
    do_eval: bool = field(
        default=True,
        metadata={"help": "평가를 실행할지 여부"},
    )
    seed: int = field(
        default=104,
        metadata={
            "help": "원하는 숫자 하시길"
            "성능 재현을 위해서는 seed 기억 필요"
        },
    )
    # 학습 관련 설정
    num_train_epochs: int = field(
        default=10,
        metadata={
            "help": "학습 할 에폭 수"
            "LLM 학습 시 에폭 수를 1~3으로 줄여서 실험 진행 필요"
        },
    )
    per_device_train_batch_size: int = field(
        default=32,
        metadata={
            "help": "학습 중 장치당 배치 크기"
            "GPU 메모리에 따라 줄여서 사용 / 너무 큰 배치는 지양"
        },
    )
    per_device_eval_batch_size: int = field(
        default=64,
        metadata={
            "help": "평가 중 장치당 배치 크기"
        },
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "그래디언트 누적을 위한 스텝 수"
            "GPU 자원이 부족할 시 배치를 줄이고 누적 수를 늘려 학습"
        },
    )
    learning_rate: int = field(
        default=5e-05,
        metadata={
            "help": "학습률 설정"
            "학습률 스케줄러(linear, cosine) 사용시 Max 값임"
        },
    )
    # Optimizer 설정
    optim: str = field(
        default="adamw_torch",
        metadata={
            "help": "옵티마이저 설정, 다른 옵티마이저 확인을 위해 아래 url에서 OptimizerNames 확인"
            "https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py"
        },
    )
    weight_decay: int = field(
        default=0.01,
        metadata={
            "help": "가중치 감소율 (정규화), 과적합 방지"
            "0.01 ~ 0.1 정도가 많이 사용"
        },
    )
    max_grad_norm: int = field(
        default=1,
        metadata={
            "help": "그래디언트 클리핑을 위한 최대 노름"
            "1 또는 그 이상의 값으로 설정하는 것이 일반적, 하지만 때에 따라(예를들어 LLM SFT시) 0도 설정 해보길 권장"
        },
    )
    # 스케줄러 설정
    lr_scheduler_type: Optional[str] = field(
        default="cosine",
        metadata={"help": "학습률 스케줄러 설정"},
    )
    warmup_steps: int = field(
        default=200,
        metadata={
            "help": "학습률을 워밍업하기 위한 스텝 수"
            "전체 학습 스텝 수의 2%~5% 정도로 설정하는 것이 일반적"
            "스텝수 = 데이터 개수*에폭수 / 배치사이즈"
        },
    )
    # 모델 평가 및 저장 관련
    metric_for_best_model: Optional[str] = field(
        default="eval_exact_match",
        metadata={"help": "가장 좋은 모델을 평가하기 위한 메트릭 설정"
                  "본 프로젝트에서는 exact_match / eval_loss를 기본적으로 사용"
        },
    )
    greater_is_better: bool = field(
        default=True,
        metadata={
            "help": "설정한 메트릭에 대해 더 큰 값이 더 좋다 혹은 더 작은 값이 더 좋다 설정"
            "Accuracy는 True 사용 / eval_loss는 False 사용"
        },
    )
    evaluation_strategy: Optional[str] = field(
        default="steps",
        metadata={"help": "epoch/steps이 끝날때마다 평가"},
    )
    save_steps: int = field(
        default=1000,
        metadata={
            "help": "어떤 step에서 저장할지"},
    )
    eval_steps: int = field(
        default=1000,
        metadata={
            "help": "어떤 step에서 저장할지"},
    )
    logging_steps: int = field(default=200)
    save_total_limit: int = field(
        default=1,
        metadata={
            "help": "가장 좋은 체크포인트 n개만 저장하여 이전 모델을 덮어씌우도록 설정"},
    )
    load_best_model_at_end: bool = field(
        default=True,
        metadata={"help": "가장 좋은 모델 로드"},
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )


if __name__ == "__main__":
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, OurTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print('training_args : ', training_args)