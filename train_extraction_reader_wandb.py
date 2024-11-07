import os
import numpy as np
import random
import json
import logging
import logging.config
import torch
import wandb
import yaml
import sys
from utils.arguments_extraction_reader import ModelArguments, DataTrainingArguments, OurTrainingArguments
from utils.dataloader_reader import ExtracionDataModule
from utils.metric import compute_metrics
from transformers import HfArgumentParser, set_seed, AutoTokenizer, AutoModelForQuestionAnswering, DataCollatorWithPadding
from model.extraction_trainer import QuestionAnsweringTrainer

# wandb login
try:
    wandb.login()
except wandb.errors.AuthError:
    print("Could not log in to Weights & Biases. Please check your credentials.")
    sys.exit(1)

logger = logging.getLogger("extraction")
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


def train():
    with wandb.init() as run:
        parser = HfArgumentParser(
            (ModelArguments, DataTrainingArguments, OurTrainingArguments)
        )
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

        # Update arguments with wandb config
        model_args.model_name_or_path = wandb.config.model_name
    
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
        
        # 데이터 불러오기 및 전처리 data_args, training_args, tokenizer
        dm = ExtracionDataModule(data_args, training_args, tokenizer)
        train_dataset, eval_dataset = dm.get_processing_data()
        # pad_to_max_length가 True이면 이미 max length로 padding된 상태 / 그렇지 않다면 data collator에서 padding을 진행해야 함
        data_collator = DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
        )
    
        model_name = model_args.model_name_or_path
        checkpoint_dir = os.path.join(training_args.output_dir, f"{model_name.replace('/', '_')}")
        training_args.output_dir = checkpoint_dir

        # Trainer 초기화
        trainer = QuestionAnsweringTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset,
            eval_examples=dm.datasets["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            post_process_function=dm._post_processing_function,
            compute_metrics=compute_metrics,
        )
        
        # Training
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
    
        # Log the final metrics to wandb
        wandb.log(metrics)

def main():
    # Load sweep config
    with open('./train_extraction_reader_wandb_config.yaml', 'r') as file:
        sweep_config = yaml.safe_load(file)
    
    sweep_id = wandb.sweep(sweep_config, project="ODQA_roberta")
    wandb.agent(sweep_id, train)

if __name__ == "__main__":
    main()
