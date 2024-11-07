import os
import numpy as np
import random
import json
import logging
import logging.config
import torch
from utils.arguments_inference import ModelArguments, DataTrainingArguments, OurTrainingArguments
from utils.dataloader_reader import load_from_disk, ExtracionDataModuleforInference
from utils.metric import compute_metrics
from transformers import HfArgumentParser, set_seed, AutoTokenizer, AutoConfig, DataCollatorWithPadding
from database.retrieval import run_sparse_retrieval
from model.extraction_trainer import QuestionAnsweringTrainer
from model.extraction_cnn import Bert_CNN_Answering  , BigBird_CNN_Answering


logger = logging.getLogger("mrc")
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]','%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

seed = 104
random.seed(seed) # python random seed 고정
np.random.seed(seed) # numpy random seed 고정
torch.manual_seed(seed) # torch random seed 고정
torch.cuda.manual_seed_all(seed)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, OurTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # 학습 파라미터 로깅
    logger.info(f"Model is from {model_args.model_name_or_path}")
    logger.info(f"Data is from {data_args.dataset_name}")
    logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정
    set_seed(training_args.seed)

    # pretrained model 과 tokenizer를 불러오기
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    config = AutoConfig.from_pretrained( 
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
    )
    model=BigBird_CNN_Answering(config, model_args.model_name_or_path)
    
    # retrieval 결과 불러오기
    datasets = load_from_disk("./resources/retrieval/top40_bm25_retrieval_dataset")
    # datasets=load_from_disk("resources/final_top40_bm25_retrieval_dataset/top40_dpr+bm25_retrieval_dataset")
    # datasets=load_from_disk("bm25_origin/bm25_origin")
    
    # 데이터 불러오기 및 전처리 data_args, training_args, tokenizer
    dm = ExtracionDataModuleforInference(data_args, training_args, tokenizer, datasets)
    eval_dataset = dm.get_processing_data()
    # pad_to_max_length가 True이면 이미 max length로 padding된 상태 / 그렇지 않다면 data collator에서 padding을 진행해야 함
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        eval_examples=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=dm._post_processing_function,
        compute_metrics=compute_metrics,
    )
    
    logger.info("*** Inference ***")
    predictions = trainer.predict(
        test_dataset=eval_dataset, test_examples=datasets["validation"]
    )
    logger.info(predictions)
    

if __name__ == "__main__":
    main()