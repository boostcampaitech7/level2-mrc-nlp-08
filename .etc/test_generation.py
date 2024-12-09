from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer, BartForConditionalGeneration
import torch


def inference(context, question, answer):
    # 입력 텍스트를 토크나이즈
    inputs = tokenizer(
            text=question,
            text_pair=context,
            padding="max_length",
            truncation=True,    # context만 truncate하기
            max_length=384,
            stride=128, # 이전 chunk와 overlap되는 부분을 두어 긴 문서를 처리할 때 유용. 모델이 더 나은 임베딩을 하도록 도와 QA에 유용.
            return_tensors='pt'
        )
    del inputs['token_type_ids']

    # GPU 사용 시 입력을 GPU로 이동
    inputs = {k: v.to('cuda') for k, v in inputs.items()}

    outputs = model.generate(**inputs, max_new_tokens=30, num_beams=5)

    # 결과 처리
    #predictions = outputs.argmax(-1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"pred : {generated_text}")
    print(f"answer : {answer}")

if __name__=="__main__":
    datasets = load_from_disk('./resources/data/train_dataset')
    val_dataset = datasets['validation']
    val_dataset = val_dataset.shuffle(seed=104)
    answers = [answer['text'] for answer in val_dataset['answers']]
    test = Dataset.from_dict({'context':val_dataset['context'][:10], 'question':val_dataset['question'][:10], 'answers':val_dataset['answers'][:10]})

    # pretrained model 과 tokenizer를 불러오기

    tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")
    model = BartForConditionalGeneration.from_pretrained("./resources/checkpoint/generation/BART/checkpoint-26000").to('cuda')
    assert model.config.vocab_size == len(tokenizer), "Model and tokenizer vocab sizes do not match!"
    for data in test:
        inference(data['context'], data['question'], data['answers'])