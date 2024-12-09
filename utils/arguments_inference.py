import os
from typing import Optional
from transformers import TrainingArguments, HfArgumentParser
from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="./resources/checkpoint/aihub_clean_transfer",
        metadata={
            "help": "학습이 완료된 모델 경로를 넣기"
        },
    )
    wiki_data_path: str = field(
        default="./resources/wiki/remove_duplicates.csv",
        metadata={
            "help": "학습이 완료된 모델 경로를 넣기"
        },
    )
    eval_retrieval: str = field(
        default="sparse",
        metadata={"help": "sparse, dense, sparse+dense 선택"},
    )
    retrieval_method: str = field(
        default="tfidf",
        metadata={"help": "tfidf, bm25, 선택"},
    )
    tokenizer: str = field(
        default="monologg/kobigbird-bert-base",
        metadata={"help": "huggingface 모델명 혹은 kiwi(한국어 형태소 분석기) 선택"},
    )
    
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="./resources/data/test_dataset",
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
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    num_clusters: int = field(
        default=64, metadata={"help": "Define how many clusters to use for faiss."}
    )
    top_k_retrieval: int = field(
        default=30,
        metadata={
            "help": "Define how many top-k passages to retrieve based on similarity."
        },
    )
    use_faiss: bool = field(
        default=False, metadata={"help": "Whether to build with faiss"}
    )
    
@dataclass
class OurTrainingArguments(TrainingArguments):
    """
    HuggingFace의 transformers 라이브러리에서 모델 학습할때 사용되는 하이퍼파라미터 커스텀
    """
    
    # 기본 학습 설정
    output_dir: Optional[str] = field(
        default="./resources/result/final_clean_top40_bm25/",
        metadata={"help": "예측 결과 저장 경로"},
    )
    do_predict: bool = field(
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


if __name__ == "__main__":
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, OurTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(training_args)