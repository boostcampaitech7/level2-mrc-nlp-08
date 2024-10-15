from typing import NoReturn
from datasets import load_from_disk
from transformers import EvalPrediction
from utils.metric_extraction import postprocess_qa_predictions
import torch

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
    def _prepare_train_features(self, examples):
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
        train_dataset = train_dataset.map(
            self._prepare_train_features,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            remove_columns=self.column_names,
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
            return formatted_predictions
        elif training_args.do_eval:
            references = [
                {"id": ex["id"], "answers": ex[self.answer_column_name]}
                for ex in self.datasets["validation"]
            ]
            return EvalPrediction(predictions=formatted_predictions, label_ids=references)
        
        
class ExtracionDataModuleforInference():
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
            return formatted_predictions
        elif training_args.do_eval:
            references = [
                {"id": ex["id"], "answers": ex[self.answer_column_name]}
                for ex in self.datasets["validation"]
            ]
            return EvalPrediction(predictions=formatted_predictions, label_ids=references)


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
        
    def _generate_training_prompt(self, instance):
        prefix_chat_template = '''<|im_start|>system
        You are Qwen, created by Alibaba Cloud. You are a helpful assistant. 모든 대답은 한국어로 해주세요.<|im_end|>
        <|im_start|>user
        question:{} 
        context:{}<|im_end|>
        <|im_start|>assistant
        {}<|im_end|>'''
        questions = instance['question']
        contexts = instance['context']
        answers = [answer['text'][0] for answer in instance['answers']]
        prompt = [prefix_chat_template.format(q, c, a) for q, c, a in zip(questions, contexts, answers)]
        prompts = self.tokenizer(prompt,
                                max_length=2000,
                                padding="max_length",
                                truncation=True,
                                return_tensors='pt')
        instance = {k: v for k, v in instance.items()}
        instance['prompt'] = prompts['input_ids']
        return instance

    def _generate_validation_prompt(self, instance):
        prefix_chat_template = '''<|im_start|>system
        You are Qwen, created by Alibaba Cloud. You are a helpful assistant. 모든 대답은 한국어로 해주세요.<|im_end|>
        <|im_start|>user
        question:{} 
        context:{}<|im_end|>'''
        question = instance['question']
        context = instance['context']
        answers = [answer['text'][0] for answer in instance['answers']]
        answer = self.tokenizer(answers,
                                max_length=self.data_args.max_answer_length,
                                padding='max_length',
                                return_tensors='pt')
        prompt = [prefix_chat_template.format(question[i], context[i]) for i in range(len(question))]
        prompt = self.tokenizer(prompt,
                                max_length=self.training_args.max_seq_length,
                                padding="max_length",
                                return_tensors='pt')
        instance = {k: v for k, v in instance.items() if k != 'prompt'}
        instance['prompt'] = prompt['input_ids']
        instance['answers'] = answers['input_ids']
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
        print('traindata : ', train_dataset)
        # Validation feature 생성
        eval_dataset = self.datasets["validation"]
        eval_dataset = eval_dataset.map(
            self._generate_validation_prompt,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            remove_columns=self.column_names,
            load_from_cache_file=not self.data_args.overwrite_cache,
        )
        print('preprocess')
        print(train_dataset.column_names)
        print(eval_dataset.column_names)
        return train_dataset, eval_dataset

    def _post_processing_function(self, predictions, examples):
        #print('examples : ', examples[0])
        print('examples shape : ', examples.shape)
        # print('predictions : ', predictions[0])
        print('predictions shape : ', predictions.shape)
        # Metric을 구할 수 있도록 Format을 맞춰줍니다.
        formatted_predictions = torch.argmax(predictions, axis=-1)
        
        if self.training_args.do_predict:
            return formatted_predictions
        elif self.training_args.do_eval:
            references = examples
            return EvalPrediction(predictions=formatted_predictions, label_ids=references)