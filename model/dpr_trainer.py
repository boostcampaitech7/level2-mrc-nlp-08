import os
import time
import logging
from tqdm import tqdm
import faiss
import numpy as np
from dataclasses import dataclass
from model.dpr import BiEncoder
from utils.metric import format_time, get_topk_accuracy
from utils.dataloader_dpr import BiEncoderDataset, DataCollator
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler

logger = logging.getLogger("mrc")
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]','%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


class BiEncoderTrainer:
    
    def __init__(self,args: dataclass, model: BiEncoder, train_dataset: BiEncoderDataset, valid_dataset: BiEncoderDataset, collator: DataCollator) -> None:
        self.args = args
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.collator = collator
        self.optimizer = self._get_adamw_optimizer(model.q_encoder, model.c_encoder, self.args)
        self.scheduler = self._get_scheduler(self.optimizer, self.args)
        if self.args.amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None
            
    def _get_adamw_optimizer(self, q_encoder, c_encoder, args):
        if args.no_decay: 
            # skip weight decay for some specific parameters i.e. 'bias', 'LayerNorm.weight'.
            no_decay = args.no_decay  
            optimizer_grouped_parameters = [
                {'params': [p for n, p in q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
                {'params': [p for n, p in q_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
                {'params': [p for n, p in c_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
                {'params': [p for n, p in c_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},    
            ]
        else:
            # weight decay for every parameter.
            optimizer_grouped_parameters = [q_encoder.parameters()] + [c_encoder.parameters()]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr = args.learning_rate, eps = args.eps)
        return optimizer
    
    def _get_scheduler(self, optimizer, args):
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min, last_epoch=-1)
        return scheduler
    

    def train(self):
        t0 = time.time()
        
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=self.args.shuffle, collate_fn=self.collator)
        best_score = None
        early_stop_score = list()
        for epoch_i in range(self.args.epochs):
            
            logger.info(f'Epoch : {epoch_i+1}/{self.args.epochs}')
            total_train_loss = 0
    
            self.model.train()
            for step, (q_batch, c_batch) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                
                q_input_ids = q_batch['input_ids'].to(self.args.device)
                q_attn_mask = q_batch['attention_mask'].to(self.args.device)
                q_token_ids = q_batch['token_type_ids'].to(self.args.device)        
                
                c_input_ids = c_batch['input_ids'].to(self.args.device)
                c_attn_mask = c_batch['attention_mask'].to(self.args.device)
                c_token_ids = c_batch['token_type_ids'].to(self.args.device)
                
                # pass the data to device(cpu or gpu)            
                self.optimizer.zero_grad()

                if self.args.amp:
                    train_loss = self.model(q_input_ids, q_attn_mask, q_token_ids,
                                        c_input_ids, c_attn_mask, c_token_ids,)

                    self.scaler.scale(train_loss.mean()).backward()
                    # Clip the norm of the gradients to 5.0.
                    # This is to help prevent the "exploding gradients" problem.
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)          
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                else:
                    train_loss = self.model(q_input_ids, q_attn_mask, q_token_ids,
                                        c_input_ids, c_attn_mask, c_token_ids,)
                    
                    train_loss.mean().backward()
                    # Clip the norm of the gradients to 5.0.
                    # This is to help prevent the "exploding gradients" problem.
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)  

                    self.optimizer.step()
                
                self.scheduler.step()        
                
                total_train_loss += train_loss.mean()
                            
            train_loss = total_train_loss / len(train_dataloader)
            print(f'Epoch:{epoch_i+1}, Train Loss:{train_loss.mean():.4f}')
            
            if epoch_i % self.args.eval_epoch == 0:
                logger.info('*** Bi-Encoder Evaluation ***')
                if isinstance(self.model, torch.nn.DataParallel):
                    model_to_save = self.model.module   
                else: model_to_save = self.model
                hidden_size = self.model.config.hidden_size
                tokenizer = self.model.tokenizer
                valid_scores = self.evaluate(model_to_save, tokenizer, self.valid_dataset, hidden_size, self.args)
                top1_acc = valid_scores['top1_accuracy']
                top5_acc = valid_scores['top5_accuracy']
                print(f'Epoch:{epoch_i+1}, Top1_Acc:{top1_acc*100:.2f}%, Top5_Acc:{top5_acc*100:.2f}%')
                
                # Check Best Model
                if not best_score or top1_acc > best_score:
                    best_score = top1_acc
                
                    if not os.path.exists(self.args.q_output_path):
                        os.makedirs(self.args.q_output_path)
                
                    if not os.path.exists(self.args.c_output_path):
                        os.makedirs(self.args.c_output_path)            
                
                    model_to_save.save_model(self.args.q_output_path, self.args.c_output_path)
        
                    logger.info(f'>>> Saved Best Model (Question Encoder) at {self.args.q_output_path}')
                    logger.info(f'>>> Saved Best Model (Context Encoder) at {self.args.c_output_path}')
                    
                # Early Stopping
                if len(early_stop_score) == 0 or top1_acc < early_stop_score[-1]:
                    early_stop_score.append(top1_acc)
                    if len(early_stop_score) == self.args.early_stop_epoch:break                                      
                else: early_stop_score = list() 
            
            else:
                print(f'Epoch:{epoch_i+1}, Train Loss: {train_loss.mean():.4f}') 
            
        training_time = format_time(time.time() - t0)
        print(f'Total Training Time: {training_time}')
                
    def evaluate(self, biencoder, tokenizer, test_dataset, embedding_size, args):
        
        question = test_dataset.question
        answer_idx = test_dataset.answer_idx

        positive_ctx = test_dataset.positive_ctx
        hard_neg_ctx = test_dataset.hard_neg_ctx
                
        positive_idx, positive_txt, positive_title = [], [], []        

        for idx in range(len(positive_ctx)): 
            positive_idx.append(positive_ctx[idx]['idx'])
            positive_txt.append(positive_ctx[idx]['text'])
            positive_title.append(positive_ctx[idx]['title'])

        all_ctx_embed = []
        
        biencoder.eval()
        for start_index in tqdm(range(0, len(positive_txt), args.batch_size)):
            batch_txt = positive_txt[start_index : start_index + args.batch_size]
            batch_title = positive_title[start_index : start_index + args.batch_size]
            
            c_batch = tokenizer(batch_title,
                                batch_txt,
                                padding=args.padding,
                                max_length=args.max_length,
                                truncation=args.truncation,)
            
            with torch.no_grad():
                pooler_output = biencoder.get_c_embeddings(input_ids=torch.tensor(c_batch['input_ids']).to(args.device),
                                                        attention_mask=torch.tensor(c_batch['attention_mask']).to(args.device),
                                                        token_type_ids=torch.tensor(c_batch['token_type_ids']).to(args.device),)
            all_ctx_embed.append(pooler_output.cpu())
        
        all_ctx_embed = np.vstack(all_ctx_embed) 
        
        faiss.normalize_L2(all_ctx_embed)       
        index = faiss.IndexFlatIP(embedding_size)
        index.add(all_ctx_embed)
        #faiss.write_index(index, 'evaluation.index')

        question_embed = []
        for start_index in tqdm(range(0, len(question), args.batch_size)):
            batch_question = question[start_index : start_index + args.batch_size]
            
            q_batch = tokenizer(batch_question,
                                padding=args.padding,
                                max_length=args.max_length,
                                truncation=args.truncation,)
            
            biencoder.eval()
            with torch.no_grad():
                pooler_output = biencoder.get_q_embeddings(input_ids=torch.tensor(q_batch['input_ids']).to(args.device),
                                                        attention_mask=torch.tensor(q_batch['attention_mask']).to(args.device),
                                                        token_type_ids=torch.tensor(q_batch['token_type_ids']).to(args.device),)
            question_embed.append(pooler_output.cpu())
        
        question_embed = np.vstack(question_embed) 

        D, I = index.search(question_embed, k=100)

        scores = get_topk_accuracy(I, answer_idx, positive_idx)

        return scores
                    