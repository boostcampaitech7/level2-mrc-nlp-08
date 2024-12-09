import logging
import random
import numpy as np
import torch
from utils.arguments_dpr import OurTrainingArguments
from utils.dataloader_dpr import BiEncoderDataset, DataCollator
from model.dpr import BiEncoder
from model.dpr_trainer import BiEncoderTrainer

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
    logger.info('*** Bi-Encoder Training ***')
    
    # Load Arguments 
    args = OurTrainingArguments()
    logger.info(f"Our training arguments: {args}")

    # Load Dataset
    train_dataset = BiEncoderDataset.load_train_dataset(args.train_data)
    valid_dataset = BiEncoderDataset.load_valid_dataset(args.valid_data)
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(valid_dataset)}")
    logger.info(f"First item in train dataset: {train_dataset[0]}")
    
    # 배치 단위의 데이터 전처리, 주어진 샘플들을 하나의 배치로 묶는 역할
    collator = DataCollator(args)

    # BiEncoder 초기화
    model = BiEncoder(args).to(args.device)
    # logger.info(model)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"모델의 전체 파라미터 수 : {total_params}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"모델의 학습 가능한 파라미터 수 : {trainable_params}")
    
    # Trainer 초기화
    trainer = BiEncoderTrainer(args, model, train_dataset, valid_dataset, collator)
    
    # 모델 학습 및 평가 진행
    trainer.train()
    
    
if __name__ == "__main__":
    main()