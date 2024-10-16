import os
import numpy as np
import random
import json
import logging
import logging.config
import torch
from utils.arguments_inference import ModelArguments, DataTrainingArguments, OurTrainingArguments
from utils.data_processing import load_from_disk, ExtracionDataModuleforInference
from utils.metric_extraction import compute_metrics
from transformers import HfArgumentParser, set_seed, AutoTokenizer, AutoModelForQuestionAnswering, DataCollatorWithPadding
from model.sparse_retrieval import run_sparse_retrieval
from model.qat_custom import QuestionAnsweringTrainer

seed = 104
random.seed(seed) # python random seed 고정
np.random.seed(seed) # numpy random seed 고정
torch.manual_seed(seed) # torch random seed 고정
torch.cuda.manual_seed_all(seed)

config = json.load(open("./utils/log/logger.json"))
logging.config.dictConfig(config)
logger = logging.getLogger(__name__)


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
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
    )
    logger.info(model)
    
    datasets = load_from_disk(data_args.dataset_name)
    
    # retrieval 방식 선택
    if data_args.eval_retrieval == 'sparse':
        datasets = run_sparse_retrieval(
            tokenizer.tokenize, datasets, training_args, data_args,
        )
    elif data_args.eval_retrieval == 'dense':
        NotImplemented
    elif data_args.eval_retrieval == 'sparse+dense':
        NotImplemented
    
    
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