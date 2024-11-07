from datasets import load_from_disk
from transformers import EvalPrediction
import numpy as np


class Seq2SeqDataModule:
    def __init__(self, data_args, training_args, tokenizer):
        self.data_args = data_args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.datasets = load_from_disk(data_args.dataset_name)
        
        if training_args.do_train:
            self.column_names = self.datasets["train"].column_names
        else:
            self.column_names = self.datasets["validation"].column_names
    
    def _prepare_features(self, examples):
        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        labels = [answers['text'][0] for answers in examples['answers']]
        tokenized_examples = self.tokenizer(
            text=examples['question'],
            text_pair= [title.strip()+" "+context.strip() for title, context in zip(examples['title'],examples['context'])],
            text_target=labels,
            padding="max_length",
            truncation="only_second",    # context만 truncate하기
            max_length=self.training_args.max_seq_length,
            stride=self.data_args.doc_stride, # 이전 chunk와 overlap되는 부분을 두어 긴 문서를 처리할 때 유용. 모델이 더 나은 임베딩을 하도록 도와 QA에 유용.
            return_overflowing_tokens=True,
            return_token_type_ids=False,
        )
        # labels 확장
        # 한 context를 여러 개의 chunk로 나누어 chunk단위로 예측할 것
        # 그런데 답을 포함하지 않는 chunk가 존재할 경우 그 chunk에서는 정답을 예측할 수 없기 때문에 label을 bos token으로 줌
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")   # 각 chunk가 몇 번째 context에서 나왔는지 index 정보
        no_answer_token_ids = self.tokenizer("<s>", max_length=self.training_args.max_seq_length, padding="max_length")['input_ids']

        example_labels = []
        for idx, sample_idx in enumerate(sample_mapping):
            label = tokenized_examples['labels'][sample_idx]
            decoded_label = self.tokenizer.decode(label, skip_special_tokens=True)
            decoded_input = self.tokenizer.decode(tokenized_examples['input_ids'][idx], skip_special_tokens=True)
            
            if decoded_label in decoded_input:
                example_labels.append(label)
            else:
                example_labels.append(no_answer_token_ids)  # skip_special_tokens=True니까 ""가 됨

        tokenized_examples['labels'] = example_labels

        return tokenized_examples
    
    def get_processing_data(self):
        train_dataset = self.datasets['train']
        train_dataset = train_dataset.shuffle(seed=104)
        train_dataset = train_dataset.map(
            self._prepare_features,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            remove_columns=train_dataset.column_names,
            load_from_cache_file=not self.data_args.overwrite_cache)

        val_dataset = self.datasets['validation']
        val_dataset = val_dataset.map(
            self._prepare_features,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            remove_columns=val_dataset.column_names,
            load_from_cache_file=not self.data_args.overwrite_cache)
        return train_dataset, val_dataset
    
        
    def _post_process_function(self, features, predictions, training_args):
        # BART 모델의 출력 처리 -> tuple로 출력됨.
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        if isinstance(features, tuple):
            features = features[0]
            
        if predictions.ndim == 3:
            predictions = np.argmax(predictions, axis=-1)
            
        # decoding -> token_ids to text
        preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        refs = self.tokenizer.batch_decode(features['labels'], skip_special_tokens=True)
        print('')
        print('예측 결과')
        for pred, ref in zip(preds[:5], refs[:5]):
            print(f'예측 : {pred} 정답 : {ref}')
        print('---------------------------------------')
        
        # do_predict인 경우 ==> formatted_predictions (inference해야함)
        # do_eval인 경우 ==>  예측, 정답 함께 반환 (f1, em결과 확인용)
        if training_args.do_predict:
            return preds
        elif training_args.do_eval:
            # 후처리된 예측 ==> {"id"(예제ID), "prediction_text"(예측답변텍스트)} 딕셔너리 리스트
            return EvalPrediction(predictions=preds, label_ids=refs)
        
# Prompt
QWEN_TEMP = '''<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant. 모든 대답은 한국어로 해주세요.<|im_end|>
<|im_start|>user
question:{} 
context:{}<|im_end|>
<|im_start|>assistant
{}<|im_end|>'''

EXAONE_TEMP = '''[|system|] You are EXAONE model from LG AI Research, a helpful assistant. [|endofturn|]
[|user|] Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that "없음", don't try to make up an answer. Please answer in short answer. Keep the answer as concise.
{}
Question:{}
[|assistant|]{}[|endofturn|]'''

class CausalLMDataModule():
    def __init__(self, data_args, training_args, tokenizer):
        self.data_args = data_args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.datasets = load_from_disk(data_args.dataset_name)
        if self.training_args.do_train:
            self.column_names = self.datasets["train"].column_names
        else:
            self.column_names = self.datasets["validation"].column_names
        # self.metric = load_metric("squad")
        
    def _generate_training_prompt(self, instance):
        questions = instance['question']    # dataset batch에서 question 가져오기
        contexts = instance['context']  # dataset batch에서 context 가져오기
        answers = [answer['text'][0] for answer in instance['answers']] # dataset batch에서 answer 가져오기
        # prefix에 formatting
        prompts = [EXAONE_TEMP.format(c, q, a) for q, c, a in zip(questions, contexts, answers)]
        # 데이터에 prompt 추가
        instance['prompt'] = prompts
        return instance

    # def _generate_validation_prompt(self, instance):
    #     # Qwen prefix
    #     prefix_chat_template = '''<|im_start|>system
    #     You are Qwen, created by Alibaba Cloud. You are a helpful assistant. 모든 대답은 한국어로 해주세요.<|im_end|>
    #     <|im_start|>user
    #     question:{} 
    #     context:{}<|im_end|>'''
    #     question = instance['question'] # dataset batch에서 question 가져오기
    #     context = instance['context']   # dataset batch에서 context 가져오기
    #     answers = [answer['text'][0] for answer in instance['answers']] # dataset batch에서 answer 가져오기
    #     # prefix에 formatting
    #     prompt = [prefix_chat_template.format(question[i], context[i]) for i in range(len(question))]
    #     instance['prompt'] = prompt
    #     instance['answers'] = answers   # answer 추가하여 metric 구할 때 활용
    #     return instance
    
    def get_processing_data(self):
        # dataset에서 train feature를 생성
        train_dataset = self.datasets["train"]
        train_dataset = train_dataset.shuffle(seed=104)
        train_dataset = train_dataset.map(self._generate_training_prompt, 
                            batched=True,
                            num_proc=self.data_args.preprocessing_num_workers,
                            remove_columns=self.column_names
                        )

        eval_dataset = self.datasets["validation"]
        eval_dataset = eval_dataset.map(
            self._generate_training_prompt,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            remove_columns=self.column_names
        )
        return train_dataset, eval_dataset