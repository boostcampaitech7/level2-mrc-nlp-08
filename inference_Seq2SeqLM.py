import torch
import random
import numpy as np
from transformers import set_seed, AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorWithPadding
from utils.metric import compute_generation_metrics
from tqdm import tqdm
from collections import OrderedDict
from datasets import load_from_disk
import json

seed = 104
random.seed(seed) # python random seed 고정
np.random.seed(seed) # numpy random seed 고정
torch.manual_seed(seed) # torch random seed 고정
torch.cuda.manual_seed_all(seed)

tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")
model = AutoModelForSeq2SeqLM.from_pretrained("resources/checkpoint/generation/lastBART/checkpoint-400").to('cuda')
datasets = load_from_disk("resources/data/final_clean_top40_bm25_retrieval_dataset")['validation']

# retrieval top-1, top-5 성능을 미루어 볼 때 정답이 포함된 passage가 첫 번째(top-1) 나올 확률이 45%
# 1, 2, 3, 4, 5번째(top-5) 나올 확률이 약 85%이므로 1~5위에서 앞쪽일수록 나올 확률이 높을 것이라 판단하여
# 100개 실험 중 top-5에 정답이 포함된 passage가 들어가있는 실험의 개수가 85개라고 할 때 (85%기 때문에)
# 1~5위에 정답 passage가 들어간 실험의 경우를 45개, 25개, 10개, 4개 1개 (총 85개)로 어림짐작하여
# 각 chunk가 몇 번째 passage에서 나왔느냐에 따라 weight을 곱해 확률합하였음.
weights = [1, 0.58, 0.24, 0.11, 0.02]
def main():
    # 모델을 초기화하기 전에 난수를 고정
    set_seed(104)

    num_return_sequences = 3
    num_beams = 3
    
    # 모델 학습 및 평가 진행
    torch.cuda.empty_cache()
    get_rank(num_return_sequences, num_beams)
            
def get_rank(num_return_sequences, num_beams):
    with open("inference.csv", 'w', encoding='utf8') as f:
        answers = {}    # 최종 json처럼 만들기
        n_best = {}
        
        for i in tqdm(range(600)):
            input = datasets[i]
            f.write(f"question : {input['question']}\n")
            inputs = tokenizer(
            text=input['question'],
            text_pair= input['context'],
            padding="max_length",
            truncation="only_second",    # context만 truncate하기
            max_length=512,
            stride=200, # 이전 chunk와 overlap되는 부분을 두어 긴 문서를 처리할 때 유용. 모델이 더 나은 임베딩을 하도록 도와 QA에 유용.
            return_overflowing_tokens=True,
            return_token_type_ids=False,
            )

            inputs = {k: torch.tensor(v).to(model.device) for k, v in inputs.items()}
            inputs.pop('overflow_to_sample_mapping')
            # 앞 chunk만 냅두기
            # 20개로 자른 근거는 한 context가 1024라고 가정, 512를 256씩 겹쳐서 잘랐을 때 나오는 총 chunk는 3*chunk 개수+(chunk개수-1)
            # 그래서 top-5로 5개의 context만 보고 싶으니까 3*5+4=19.
            inputs['input_ids'] = inputs['input_ids'][:20]
            inputs['attention_mask'] = inputs['attention_mask'][:20]
            
            id = input['id']
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=30,
                                         num_beams=num_beams, return_dict_in_generate=True, # 예측 seq과 확률을 dict 형태로 출력
                                         num_return_sequences=num_return_sequences, output_scores=True) # num_return_seq : 상위 n개 예측을 뽑음, output_scores : 확률 받음
                sequences = outputs.sequences
                # 각 예측의 확률을 구함.
                transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False)
                
                output_length = np.sum(transition_scores.cpu().numpy() < 0, axis=1)
                length_penalty = model.generation_config.length_penalty
                reconstructed_scores = transition_scores.cpu().sum(axis=1) / (output_length**length_penalty)
                
                # 모든 청크의 예측 담기
                chunks = {} 
                for j in range(inputs['input_ids'].shape[0]):   # 각 chunk의 예측값 하나씩 가져오기
                    pred = [tokenizer.decode(sequences[j*num_return_sequences+k], skip_special_tokens=True) for k in range(num_return_sequences)]
                    prev = []
                    for token, score in zip(pred, reconstructed_scores[j*num_return_sequences:(j+1)*num_return_sequences]):
                        # weight 정하기
                        if j//4==0:
                            weight = weights[0]
                        elif j//4==1:
                            weight = weights[1]
                        elif j//4==2:
                            weight = weights[2]
                        elif j//4==3:
                            weight = weights[3]
                        else:
                            weight = weights[4]
                        score = np.exp(score.numpy())*weight
                        token = " ".join(OrderedDict.fromkeys(token.split()))
                    
                        # bos거나 얼토당토 않은 것 예측하면 버려
                        if token=="" or token.replace(" ", "") not in input['context'].replace(" ", ""):
                            continue
                        
                        if chunks.get(token):   # 딕셔너리에 존재하면
                            chunks[token] += score
                        else:
                            chunks[token] = score
                        prev.append(token)

                '''if chunks=={}:
                    print('하아')
                    print(input['question'])
                    print(pred)'''
                # 가장 확률 높은 예측값을 정답으로
                ranking = sorted(chunks.items(), key=lambda x:-x[1])
                target = ranking[0]
                f.write(f'정답은 {target[0]} / {target[1]}\n')
                
                json_ranking = [{"text":k, "probability":v} for k, v in chunks.items()]
                n_best[id] = json_ranking

            answers[id] = target[0] # text 뱉기
            f.write("=================================================================================================\n")

        with open('predictions.json', 'w', encoding='utf-8') as fj: 
            json.dump(answers, fj, ensure_ascii=False, indent=4)
    
        with open('n_best.json', 'w', encoding='utf-8') as fk: 
            json.dump(n_best, fk, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()