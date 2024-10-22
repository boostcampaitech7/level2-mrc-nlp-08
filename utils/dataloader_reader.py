from typing import NoReturn
from datasets import load_from_disk
from transformers import EvalPrediction
from utils.metric import postprocess_qa_predictions
import torch
from datasets import load_metric


class ExtracionDataModule():
    def __init__(self, data_args, training_args, tokenizer) -> NoReturn:
        self.data_args = data_args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.datasets = load_from_disk(data_args.dataset_name)
        if self.training_args.do_train:
            self.column_names = self.datasets["train"].column_names
        else:
            self.column_names = self.datasets["validation"].column_names    
        self.question_column_name = "question" if "question" in self.column_names else self.column_names[0]
        self.context_column_name = "context" if "context" in self.column_names else self.column_names[1]
        self.answer_column_name = "answers" if "answers" in self.column_names else self.column_names[2]
        self.pad_on_right = self.tokenizer.padding_side == "right"

    # Train preprocessing
    def _prepare_train_features(self, examples):#train: start_positions, end_positions 생성 / valid: X
        # (text type / example) ==> (token type / tokenized_examples)
        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        tokenized_examples = self.tokenizer(
            examples[self.question_column_name if self.pad_on_right else self.context_column_name],
            examples[self.context_column_name if self.pad_on_right else self.question_column_name],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.data_args.max_seq_length,
            stride=self.data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids= False, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
            padding="max_length" if self.data_args.pad_to_max_length else False,
        )
        

        # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
        # 지정한 길이가 넘는 context를 truncate할건데 truncate한 것이 원래 어떤 문장의 요소인지 numbering
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # token의 캐릭터 단위 position를 찾을 수 있도록 offset mapping을 사용합니다.
        # start_positions과 end_positions을 찾는데 도움을 줄 수 있습니다.
        # 토큰화하다보면 단어 한 개가 여러 개로 나뉠 수도 있고 그냥 한 개로 토큰화될 수 있는데 각 토큰이 원본 텍스트에서 어디에 대응되는 것인지 numbering
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # 데이터셋에 "start position", "enc position" label을 부여합니다.
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)  # cls index

            # sequence id를 설정합니다 (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # 하나의 example이 여러개의 span을 가질 수 있습니다.
            sample_index = sample_mapping[i]
            answers = examples[self.answer_column_name][sample_index]

            # answer가 없을 경우 cls_index를 answer로 설정합니다(== example에서 정답이 없는 경우 존재할 수 있음).
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # text에서 정답의 Start/end character index
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # text에서 current span의 Start token index
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if self.pad_on_right else 0):
                    token_start_index += 1

                # text에서 current span의 End token index
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if self.pad_on_right else 0):
                    token_end_index -= 1

                # 정답이 span을 벗어났는지 확인합니다(정답이 없는 경우 CLS index로 label되어있음).
                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # token_start_index 및 token_end_index를 answer의 끝으로 이동합니다.
                    # Note: answer가 마지막 단어인 경우 last offset을 따라갈 수 있습니다(edge case).
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    # Validation preprocessing
    def _prepare_validation_features(self, examples): #(test type / example) ==> (token type / tokenized_examples)
        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        tokenized_examples = self.tokenizer(
            examples[self.question_column_name if self.pad_on_right else self.context_column_name],
            examples[self.context_column_name if self.pad_on_right else self.question_column_name],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.data_args.max_seq_length,
            stride=self.data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            # return_token_type_ids=False, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
            padding="max_length" if self.data_args.pad_to_max_length else False,
        )

        # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # evaluation을 위해, prediction을 context의 substring으로 변환해야합니다.
        # corresponding example_id를 유지하고 offset mappings을 저장해야합니다.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # sequence id를 설정합니다 (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if self.pad_on_right else 0

            # 하나의 example이 여러개의 span을 가질 수 있습니다.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping을 None으로 설정해서 token position이 context의 일부인지 쉽게 판별 할 수 있습니다.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

    def get_processing_data(self):
        # dataset에서 train feature를 생성
        train_dataset = self.datasets["train"]
        train_dataset = train_dataset.map( #전체 데이터셋을 매핑해서 전처리 진행
            self._prepare_train_features,
            batched=True, #데이터를 배치로 처리
            num_proc=self.data_args.preprocessing_num_workers, #병렬 처리
            remove_columns=self.column_names, #모델 입력에 불필요한 열 제거 후 최종 데이터셋 만듦
            load_from_cache_file=not self.data_args.overwrite_cache,
        )
        
        # Validation Feature 생성
        eval_dataset = self.datasets["validation"]
        eval_dataset = eval_dataset.map(
            self._prepare_validation_features,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            remove_columns=self.column_names,
            load_from_cache_file=not self.data_args.overwrite_cache,
        )
        
        return train_dataset, eval_dataset
    
    def _post_processing_function(self, examples, features, predictions, training_args):
        # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
        # 모델의 원시 예측(start logits와 end logits)을 원본 컨텍스트의 실제 텍스트 답변으로 변환
        predictions = postprocess_qa_predictions( #utils.metric_extraction (all_predictions반환)
        # {
        #     "example_id": "예측 결과", 
        #     ...원본 예제 수만큼...
        # }
            examples=examples, #원본 형태의 데이터(질문, 컨텍스트, 답변)
            features=features, #전처리된 데이터셋의 feature (input_ids, attention_mask, token_type_ids)
            predictions=predictions, # (start_logits, end_logits)
            max_answer_length=self.data_args.max_answer_length, #최대 답변 길이 고려해야함
            output_dir=training_args.output_dir, #결과 저장 디렉토리
        )
        # Metric을 구할 수 있도록 Format을 맞춰줍니다.
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]
        #후처리된 예측 ==> {"id"(예제ID), "prediction_text"(예측답변텍스트)} 딕셔너리 리스트
        #do_predict인 경우 ==> formatted_predictions (inference해야함)
        #do_eval인 경우 ==>  예측, 정답 함께 반환 (f1, em결과 확인용)
        if training_args.do_predict:
            return formatted_predictions
        elif training_args.do_eval:
            references = [
                {"id": ex["id"], "answers": ex[self.answer_column_name]}
                for ex in self.datasets["validation"]
            ]
            return EvalPrediction(predictions=formatted_predictions, label_ids=references)
        
        
class ExtracionDataModuleforInference(): #ExtracionDataModule의 valid process와 동일함
    def __init__(self, data_args, training_args, tokenizer, datasets) -> NoReturn:
        self.data_args = data_args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.datasets = datasets
        self.column_names = self.datasets["validation"].column_names    
        self.question_column_name = "question" if "question" in self.column_names else self.column_names[0]
        self.context_column_name = "context" if "context" in self.column_names else self.column_names[1]
        self.answer_column_name = "answers" if "answers" in self.column_names else self.column_names[2]
        self.pad_on_right = self.tokenizer.padding_side == "right"

    # Validation preprocessing
    def _prepare_validation_features(self, examples):
        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        tokenized_examples = self.tokenizer(
            examples[self.question_column_name if self.pad_on_right else self.context_column_name],
            examples[self.context_column_name if self.pad_on_right else self.question_column_name],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.data_args.max_seq_length,
            stride=self.data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            # return_token_type_ids=True, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
            padding="max_length" if self.data_args.pad_to_max_length else False,
        )

        # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # evaluation을 위해, prediction을 context의 substring으로 변환해야합니다.
        # corresponding example_id를 유지하고 offset mappings을 저장해야합니다.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # sequence id를 설정합니다 (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if self.pad_on_right else 0

            # 하나의 example이 여러개의 span을 가질 수 있습니다.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping을 None으로 설정해서 token position이 context의 일부인지 쉽게 판별 할 수 있습니다.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples
    
    def get_processing_data(self):
        # Validation Feature 생성
        eval_dataset = self.datasets["validation"]
        eval_dataset = eval_dataset.map(
            self._prepare_validation_features,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            remove_columns=self.column_names,
            load_from_cache_file=not self.data_args.overwrite_cache,
        )
        
        return eval_dataset
    
    def _post_processing_function(self, examples, features, predictions, training_args):
        # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=self.data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )
        # Metric을 구할 수 있도록 Format을 맞춰줍니다.
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]
        
        if training_args.do_predict:
            return formatted_predictions #포맷팅된 예측 결과
        elif training_args.do_eval:
            references = [
                {"id": ex["id"], "answers": ex[self.answer_column_name]}
                for ex in self.datasets["validation"]
            ]
            #예측과 정답을 함께 반환(진짜 정답이랑 비교 가능하도록)
            return EvalPrediction(predictions=formatted_predictions, label_ids=references) #predictions: model의 예측값, label_ids: label 값


class GenerationDataModule():
    def __init__(self, data_args, training_args, tokenizer) -> NoReturn:
        self.data_args = data_args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.datasets = load_from_disk(data_args.dataset_name)
        if self.training_args.do_train:
            self.column_names = self.datasets["train"].column_names
        else:
            self.column_names = self.datasets["validation"].column_names
        self.metric = load_metric("squad")
        
    def _generate_training_prompt(self, instance):
        # Qwen prefix
        prefix_chat_template = '''<|im_start|>system
        You are Qwen, created by Alibaba Cloud. You are a helpful assistant. 모든 대답은 한국어로 해주세요.<|im_end|>
        <|im_start|>user
        question:{} 
        context:{}<|im_end|>
        <|im_start|>assistant
        {}<|im_end|>'''
        questions = instance['question']    # dataset batch에서 question 가져오기
        contexts = instance['context']  # dataset batch에서 context 가져오기
        answers = [answer['text'][0] for answer in instance['answers']] # dataset batch에서 answer 가져오기
        # prefix에 formatting
        prompts = [prefix_chat_template.format(q, c, a) for q, c, a in zip(questions, contexts, answers)]
        # 데이터에 prompt 추가
        instance['prompt'] = prompts
        return instance

    def _generate_validation_prompt(self, instance):
        # Qwen prefix
        prefix_chat_template = '''<|im_start|>system
        You are Qwen, created by Alibaba Cloud. You are a helpful assistant. 모든 대답은 한국어로 해주세요.<|im_end|>
        <|im_start|>user
        question:{} 
        context:{}<|im_end|>'''
        question = instance['question'] # dataset batch에서 question 가져오기
        context = instance['context']   # dataset batch에서 context 가져오기
        answers = [answer['text'][0] for answer in instance['answers']] # dataset batch에서 answer 가져오기
        # prefix에 formatting
        prompt = [prefix_chat_template.format(question[i], context[i]) for i in range(len(question))]
        instance['prompt'] = prompt
        instance['answers'] = answers   # answer 추가하여 metric 구할 때 활용
        return instance
    
    def get_processing_data(self):
        # dataset에서 train feature를 생성
        train_dataset = self.datasets["train"]
        train_dataset = train_dataset.shuffle(seed=104)
        train_dataset = train_dataset.map(self._generate_training_prompt, 
                            batched=True,
                            num_proc=self.data_args.preprocessing_num_workers,
                            remove_columns=self.column_names,
                            load_from_cache_file=not self.data_args.overwrite_cache,
                        )
        # Validation feature 생성
        eval_dataset = self.datasets["validation"]
        eval_dataset = eval_dataset.map(
            self._generate_validation_prompt,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            remove_columns=self.column_names,
            load_from_cache_file=not self.data_args.overwrite_cache,
        )
        return train_dataset, eval_dataset

    def _post_processing_function(self, predictions, examples):
        # Metric을 구할 수 있도록 Format을 맞춰줍니다.
        formatted_predictions = torch.argmax(predictions, axis=-1)

        return formatted_predictions
        
    def compute_metrics(self, p: EvalPrediction):
        predictions, label_ids = p.predictions, p.label_ids
        # trainer에서 loss 계산을 위해 pad token을 -100으로 해놓았는데 이를 다시 pad token으로 변환
        # 냅두면 decoding 불가
        predictions[predictions == -100] = 141643
        label_ids[label_ids == -100] = 141643
        # EM을 구하기 위해 text 입력을 해줘야 하여 decoding
        pred_texts = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        label_texts = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        # squad metric이 지원하는 형식에 맞추어 줌.
        # id : question-answer 쌍을 구별하기 위한 id
        # answer_start : generation에서는 비워두어도 됨.
        predictions = [{'prediction_text': pred, 'id': str(i)} for i, pred in enumerate(pred_texts)]
        references = [{'answers': {'answer_start': [], 'text': [label]}, 'id': str(i)} for i, label in enumerate(label_texts)]
    
        result = self.metric.compute(predictions=predictions, references=references)
        return {
            "exact_match": result["exact_match"],
            "f1": result["f1"]
        }