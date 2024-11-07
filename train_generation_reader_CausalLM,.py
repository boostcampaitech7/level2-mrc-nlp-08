import os
import numpy as np
import random
import logging
import logging.config
import torch
from utils.arguments_prompt_reader import ModelArguments, DataTrainingArguments, OurTrainingArguments
from utils.dataloader_generation import GenerationDataModule
from utils.utils import find_linear_names
from trl import SFTTrainer
from transformers import HfArgumentParser, set_seed, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from huggingface_hub import login

login()

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
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, OurTrainingArguments) # arguement 쭉 읽어보면서 이해하기
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
    quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        quantization_config=quantization_config
    )
    modules  = find_linear_names(model, 'qlora')
    lora_config = LoraConfig(r=64, lora_alpha=64, lora_dropout=0.1, bias="none", target_modules=modules, task_type="CAUSAL_LM", modules_to_save=None,)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    logger.info(model)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"모델의 전체 파라미터 수 : {total_params}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"모델의 학습 가능한 파라미터 수 : {trainable_params}")

    # 데이터 불러오기 및 전처리
    dm = GenerationDataModule(data_args, training_args, tokenizer) 
    train_dataset, eval_dataset = dm.get_processing_data()
    # Trainer 초기화
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side='right'
    
    # SFTTrainer는 trainer가 알아서 dataset을 tokenize함
    trainer = SFTTrainer(
        model=model,
        dataset_text_field="prompt",
        max_seq_length=training_args.max_seq_length,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
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


if __name__ == "__main__":
    main()