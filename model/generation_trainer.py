# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Question-Answering task와 관련된 'Trainer'의 subclass 코드 입니다.
"""
import torch
from trl import SFTTrainer
from torch.utils.data import DataLoader
from transformers import Trainer, EvalPrediction


class Seq2SeqTrainer(Trainer):
    def __init__(self, *args, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.post_process_function = post_process_function

    # 학습 중 또는 학습 후 squad 매트릭으로 평가
    def evaluate(self, ignore_keys=None):
        # eval metric을 return. eval loss와 추가 metric
        eval_dataloader = self.get_eval_dataloader(self.eval_dataset)

        if eval_dataloader.batch_size==None:
            eval_dataloader.batch_size = self.args.per_device_train_batch_size
            
        # eval loss 쓰려면
        # valid_dataset의 prediction들 뽑기 -> predictions, labels, metrics 각각을 logit 형태로 받음
        output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            # metric이 없으면 예측값을 모으는 이유가 없으므로 predictions, labels 없이 metrics만 반환
            prediction_loss_only=False if self.args.do_eval else True,
            ignore_keys=ignore_keys,
        )
        if self.post_process_function is not None and self.compute_metrics is not None:
            # EM, f1을 구하기 위해 token_ids -> text로 변환
            eval_preds = self.post_process_function(
                self.eval_dataset, output.predictions, self.args
            )
            # do_eval일 경우 metric 계산
            if isinstance(eval_preds, EvalPrediction):
                metrics = self.compute_metrics(eval_preds)
            else:   # do_predict일 경우 그대로 값만 내보냄
                return eval_preds
        else:
            # evaluation_loop는 token ids로 metric을 계산하므로 compute_metrics의 결과가 차이가 있을 수 있음
            metrics = output.metrics
            del metrics['eval_model_preparation_time']
        print('')
        print(f'eval loss : ', output.metrics['eval_loss'])
        print('output EM : ', output.metrics['eval_exact_match'])
        print('metrics : ', metrics)
        print('------------------------')
        return output.metrics
    

    
class CausalLMTrainer(SFTTrainer):
    def __init__(self, *args, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.post_process_function = post_process_function
    
    # 학습 중 또는 학습 후 squad 매트릭으로 평가
    def evaluate(self, ignore_keys=None):        
        # eval metric을 return. eval loss와 추가 metric
        eval_dataloader = self.get_eval_dataloader(self.eval_dataset)

        if eval_dataloader.batch_size==None:
            eval_dataloader.batch_size = self.args.per_device_train_batch_size
        # eval loss 쓰려면
        # valid_dataset의 prediction들 뽑기 -> predictions, labels, metrics 각각을 logit 형태로 받음
        # vram 너무 많이 먹어서 서버에서 실행 불가
        '''output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            # metric이 없으면 예측값을 모으는 이유가 없으므로 아래의 코드를 따르게 됩니다.
            # self.args.prediction_loss_only
            prediction_loss_only=False if self.args.do_eval else True,
            ignore_keys=ignore_keys,
        )

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(
                self.eval_dataset, output.predictions, self.args
            )
            # do_eval일 경우 metric 계산
            if isinstance(eval_preds, EvalPrediction):
                metrics = self.compute_metrics(eval_preds)
            else:   # do_predict일 경우 그대로 값만 내보냄
                return eval_preds
        else:
            metrics = output.metrics
            del metrics['eval_model_preparation_time']
        print('metrics의 : ', metrics)
        return metrics
    
        '''
        #eval_loss 없이 EM, f1만 쓰려면
        EM, f1 = [], []
        for i, batch in enumerate(eval_dataloader):
            inputs = {k: v.to(self.model.device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = self.model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                              max_new_tokens=30, num_beams=3)

            labels = batch['labels']
            # loss 구할 때 pad token은 고려하지 않기 위해 음수 값으로 설정.
            # token id가 음수인 token은 모델이 loss 계산에서 제외함.
            # decoding할 땐 음수값이면 안되기 때문에 다시 pad_token으로 변환
            outputs[outputs == -100] = self.tokenizer.pad_token_id
            labels[labels == -100] = self.tokenizer.pad_token_id
            preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            score = self.metrics_computer(preds, labels)
            if i%20==0:
                for pred, label in zip(preds, labels):
                    print(f'pred : {pred}')
                    print(f'label : {label}')
                print(score)
            EM.append(score['exact_match'])
            f1.append(score['f1'])
        result = {'eval_exact_match':sum(EM)/len(eval_dataloader), 'eval_f1':sum(f1)/len(self.eval_dataloader)}
        print(f"total EM : {result['eval_exact_match']}, f1 : {result['eval_f1']}")
        return result

    
    # 텍스트화한 애들로 squad metric에 넣어서 값 구하기
    def metrics_computer(self, preds, labels):
        # squad metric이 지원하는 형식에 맞추어 줌.
        # id : question-answer 쌍을 구별하기 위한 id
        # answer_start : generation에서는 비워두어도 됨.
                
        predictions = [{'prediction_text': pred, 'id': str(i)} for i, pred in enumerate(preds)]
        references = [{'answers': {'answer_start': [], 'text': [label]}, 'id': str(i)} for i, label in enumerate(labels)]
        result = self.metrics.compute(predictions=predictions, references=references)
        return result