import random
import numpy as np
from datasets import Dataset
from database.sparse_retrieval import SparseRetrieval
from datasets import DatasetDict, Features, Sequence, Value
from utils.arguments_inference import ModelArguments, DataTrainingArguments, OurTrainingArguments
import torch
import pickle
import logging
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from model.dpr import Pooler
from utils.dataloader_dpr import BiEncoderDataset
from database.dense_retrieval import VectorDatabase
from model.bm25 import BM25Reranker
from utils.metric import get_topk_accuracy
from utils.utils import init_logging


seed = 104
random.seed(seed) # python random seed 고정
np.random.seed(seed) # numpy random seed 고정


def run_sparse_retrieval(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: OurTrainingArguments,
    datasets: DatasetDict,
) -> DatasetDict:

    # Query에 맞는 Passage들을 Retrieval 합니다.
    retriever = SparseRetrieval(model_args)
    retriever.get_sparse_embedding()
    
    if data_args.use_faiss:
        retriever.build_faiss(num_clusters=data_args.num_clusters)
        df = retriever.retrieve_faiss(
            datasets["validation"], topk=data_args.top_k_retrieval
        )
    else:
        df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval)

    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif training_args.do_eval:
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets