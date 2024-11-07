import os
import torch
import random
import logging
import numpy as np
from utils.dataloader_generation import Seq2SeqDataModule
from utils.arguments_generation_reader import HfArgumentParser, ModelArguments, DataTrainingArguments, OurTrainingArguments
from model.generation_trainer import Seq2SeqTrainer
from transformers import HfArgumentParser, set_seed, AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorWithPadding, EarlyStoppingCallback
from utils.metric import compute_generation_metrics

logger = logging.getLogger("gen")
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
    logger.info('*** BART Training ***')
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, OurTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Load Arguments 
    logger.info(f"Our training arguments: {training_args}")
    
    # 모델을 초기화하기 전에 난수를 고정
    set_seed(training_args.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path)
    
    # Load Dataset
    dm = Seq2SeqDataModule(data_args, training_args, tokenizer) 
    train_dataset, eval_dataset = dm.get_processing_data()
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    logger.info(f"First item in train dataset: {train_dataset[0]}")
    
    # 배치 단위의 데이터 전처리, 주어진 샘플들을 하나의 배치로 묶는 역할
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"모델의 전체 파라미터 수 : {total_params}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"모델의 학습 가능한 파라미터 수 : {trainable_params}")

    # Trainer 초기화
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_generation_metrics,
        post_process_function=dm._post_process_function,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
    )

    # 모델 학습 및 평가 진행
    torch.cuda.empty_cache()
    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

    with open(output_train_file, "w") as writer:
        logger.info("***** Train results *****")
        for key, value in sorted(train_result.metrics.items()):
            logger.info(f"  {key} = {value}")
            writer.write(f"{key} = {value}\n")

    # Training state 저장
    trainer.state.save_to_json(
        os.path.join(training_args.output_dir, "trainer_state.json")
    )
    
    # Evaluation
    logger.info("***** Evaluate *****")
    metrics = trainer.evaluate()

    metrics["eval_samples"] = len(eval_dataset)

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    

if __name__ == "__main__":
    main()