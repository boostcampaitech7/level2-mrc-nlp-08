import os
import json
import logging
import datetime
import collections
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from datasets import Dataset
from datasets import load_metric
from transformers import EvalPrediction
from typing import Any, Optional, Tuple
from datasets import DatasetDict, Features, Sequence, Value
from utils.context_preprocess import WikipediaTextPreprocessor

preprocessor = WikipediaTextPreprocessor()

logger = logging.getLogger("mrc")
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]','%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

metric = load_metric("squad")


def compute_metrics(p: EvalPrediction): #EvalPrediction 구조 | predictions: 모델의 예측값, label_ids: 실제 정답 레이블
    result = metric.compute(predictions=p.predictions, references=p.label_ids)
    result['eval_exact_match'] = result['exact_match']
    del result['exact_match']
    result['eval_f1'] = result['f1']
    del result['f1']
    return result


def compute_generation_metrics(p: EvalPrediction): #EvalPrediction 구조 | predictions: 모델의 예측값, label_ids: 실제 정답 레이블
    predictions = p.predictions
    references = p.label_ids
    
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    if isinstance(references, tuple):
        references = references[0]
    
    if np.array(predictions).ndim == 3:
        predictions = np.argmax(predictions, axis=-1)

    predictions = [{'id': str(i), 'prediction_text': pred} for i, pred in enumerate(predictions)]
    references = [{'answers': {'answer_start': [], 'text': [label]}, 'id': str(i)} for i, label in enumerate(references)]

    result = metric.compute(predictions=predictions, references=references)
    result['eval_exact_match'] = result['exact_match']
    del result['exact_match']
    result['eval_f1'] = result['f1']
    del result['f1']
    return result


def postprocess_qa_predictions( #모델의 예측(start logits, end logits) ===> 실제 text답변
    examples, #원본 형태의 데이터(질문, 컨텍스트, 답변)
    features, #전처리(토큰화)된 데이터셋의 feature (input_ids, attention_mask, token_type_ids)
    predictions: Tuple[np.ndarray, np.ndarray], # (start_logits, end_logits)
    version_2_with_negative: bool = False, #정답이 없는 데이터셋이 포함되어있는지 여부
    n_best_size: int = 20, #답변을 찾을 때 생성할 n-best prediction 총 개수
    max_answer_length: int = 30,#최대 답변 길이
    null_score_diff_threshold: float = 0.0, #null 답변(답 존재 안함)을 선택하는 데 사용되는 threshold
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    is_world_process_zero: bool = True,
):
    """
    Post-processes : qa model의 prediction 값을 후처리하는 함수
    모델은 start logit과 end logit을 반환하기 때문에, 이를 기반으로 original text로 변경하는 후처리가 필요함

    Args:
        examples: 전처리 되지 않은 데이터셋 (see the main script for more information).
        features: 전처리가 진행된 데이터셋 (see the main script for more information).
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            모델의 예측값 :start logits과 the end logits을 나타내는 two arrays              첫번째 차원은 :obj:`features`의 element와 갯수가 맞아야함.
        version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
            정답이 없는 데이터셋이 포함되어있는지 여부를 나타냄
        n_best_size (:obj:`int`, `optional`, defaults to 20):
            답변을 찾을 때 생성할 n-best prediction 총 개수
        max_answer_length (:obj:`int`, `optional`, defaults to 30):
            생성할 수 있는 답변의 최대 길이
        null_score_diff_threshold (:obj:`float`, `optional`, defaults to 0):
            null 답변을 선택하는 데 사용되는 threshold
            : if the best answer has a score that is less than the score of
            the null answer minus this threshold, the null answer is selected for this example (note that the score of
            the null answer for an example giving several features is the minimum of the scores for the null answer on
            each feature: all features must be aligned on the fact they `want` to predict a null answer).
            Only useful when :obj:`version_2_with_negative` is :obj:`True`.
        output_dir (:obj:`str`, `optional`):
            아래의 값이 저장되는 경로
            dictionary : predictions, n_best predictions (with their scores and logits) if:obj:`version_2_with_negative=True`,
            dictionary : the scores differences between best and null answers
        prefix (:obj:`str`, `optional`):
            dictionary에 `prefix`가 포함되어 저장됨
        is_world_process_zero (:obj:`bool`, `optional`, defaults to :obj:`True`):
            이 프로세스가 main process인지 여부(logging/save를 수행해야 하는지 여부를 결정하는 데 사용됨)
    """
    assert (
        len(predictions) == 2
    ), "`predictions` should be a tuple with two elements (start_logits, end_logits)."
    all_start_logits, all_end_logits = predictions
    #prediction에 두 값(시작, 끝 logit)이 들어왔는지 확인

    #len(predictions[0]) : start logit 개수 / len(feature) : 전처리된 feature개수
    #두 개가 같아야함
    assert len(predictions[0]) == len(
        features
    ), f"Got {len(predictions[0])} predictions and {len(features)} features."

    # example과 mapping되는 feature 생성
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # prediction, nbest에 해당하는 OrderedDict 생성합니다.
    all_predictions = collections.OrderedDict() #최종 예측
    all_nbest_json = collections.OrderedDict() # nbest 예측
    if version_2_with_negative: #답변이 없는 dataset의 경우 점수 차이
        scores_diff_json = collections.OrderedDict()

    # Logging.
    logger.setLevel(logging.INFO if is_world_process_zero else logging.WARN)
    logger.info(
        f"Post-processing {len(examples)} example predictions split into {len(features)} features."
    ) #후처리할 example, feature 수

    # 전체 example들에 대한 main Loop
    for example_index, example in enumerate(tqdm(examples)): #모든 예제 하나씩
        # 해당하는 현재 example index
        feature_indices = features_per_example[example_index] #현재 예제의 모든 feature 찾기
        min_null_prediction = None
        prelim_predictions = []

        for feature_index in feature_indices: #현재 예제에 대한 하나의 feature에 대해
            # 각 feature에 대한 모든 prediction을 가져옵니다.
            #start_logits : 현재 example(질문-문맥 쌍)에서 각 토큰 위치가 답변의 시작일 가능성
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index] #답변의 시작과 끝 위치 예측값
            # logit과 original context의 logit을 mapping합니다.
            offset_mapping = features[feature_index]["offset_mapping"] #토큰과 원본 텍스트 위치 연결
            # Optional : `token_is_max_context`, 제공되는 경우 현재 기능에서 사용할 수 있는 max context가 없는 answer를 제거합니다
            token_is_max_context = features[feature_index].get(
                "token_is_max_context", None
            )

            # minimum null prediction을 업데이트 합니다.
            feature_null_score = start_logits[0] + end_logits[0] #답변이 없다([cls]토큰)로 판단할 때 score 계산
            if (
                min_null_prediction is None
                or min_null_prediction["score"] > feature_null_score #새로 계산된 null score가 기존보다 낮으면 새로 갱신
            ):
                min_null_prediction = { #null answer에 대한 정보 저장
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # `n_best_size`보다 큰 start and end logits을 살펴봅니다.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            #만약에 start_logit이 [-1, 2.3, 0.8] ===> [0, 2, 1]반환
            #n best가 2이면 [1, 2]만 슬라이싱됨
            #start_logits 정렬했을 때 가장 높은 점수부터 n개만큼 잘라서 그 index값 반환 -> 리스트로 변환
            
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

            #선정된 시작 index, 끝 index의 모든 조합에 대해서
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # out-of-scope answers는 고려하지 않습니다.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # 길이가 < 0 또는 > max_answer_length인 answer도 고려하지 않습니다.
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue
                    # 최대 context가 없는 answer도 고려하지 않습니다.
                    if (
                        token_is_max_context is not None
                        and not token_is_max_context.get(str(start_index), False)
                    ):
                        continue
                    
                    #위에서 안 걸러진 유효한 조합이면 답변을 목록에 추가
                    #(점수, 시작점수, 끝 점수)
                    prelim_predictions.append(
                        {
                            "offsets": (
                                offset_mapping[start_index][0],
                                offset_mapping[end_index][1],
                            ),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )

        if version_2_with_negative:
            #만약 '답변이없다'도 답변으로 고려하면
            # minimum null prediction을 추가합니다.
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        # 점수를 정렬해서 가장 좋은 `n_best_size` predictions만 유지합니다.
        predictions = sorted(
            prelim_predictions, key=lambda x: x["score"], reverse=True
        )[:n_best_size]

        # 낮은 점수로 인해 제거된 경우 minimum null prediction을 다시 추가합니다.
        if version_2_with_negative and not any(
            p["offsets"] == (0, 0) for p in predictions
        ):
            predictions.append(min_null_prediction)

        # offset을 사용하여 context에서 실제 텍스트를 반환
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]

        # rare edge case에는 예측값이 다 null이거나 비어있으면 failure를 피하기 위해 fake prediction("empty")을 만듭니다.
        if len(predictions) == 0 or (
            len(predictions) == 1 and predictions[0]["text"] == ""
        ):

            predictions.insert(
                0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0}
            )

        # 모든 점수의 소프트맥스를 계산합니다(점수==>확률)
        # #(we do it with numpy to stay independent from torch/tf in this file, using the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # 예측값에 확률을 포함합니다.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # best prediction을 선택합니다.
        #답변없음 포함X면 첫번째 답변 선택
        if not version_2_with_negative:
            all_predictions[example["id"]] = predictions[0]["text"]
        else:
            # else case : 먼저 비어 있지 않은 최상의 예측을 찾아야 합니다
            i = 0
            while predictions[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions[i]

            # threshold를 사용해서 null prediction을 비교합니다.
            score_diff = (
                null_score
                - best_non_null_pred["start_logit"]
                - best_non_null_pred["end_logit"]
            )
            scores_diff_json[example["id"]] = float(score_diff)  # JSON-serializable 가능
            if score_diff > null_score_diff_threshold:
                all_predictions[example["id"]] = ""
            else:
                all_predictions[example["id"]] = best_non_null_pred["text"]

        # np.float를 다시 float로 casting -> `predictions`은 JSON-serializable 가능
        # 각 질문에 대한 top n 답변 저장
        all_nbest_json[example["id"]] = [
            {
                k: (
                    float(v)
                    if isinstance(v, (np.float16, np.float32, np.float64))
                    else v
                )
                for k, v in pred.items()
            }
            for pred in predictions
        ]

    # output_dir이 있으면 모든 dicts를 저장합니다.
    if output_dir is not None: #출력 디렉토리가 지정되어 있다면
        #지정된 디렉토리가 있는지 확인
        assert os.path.isdir(output_dir), f"{output_dir} is not a directory."
        #예측결과 저장할 경로 생성
        prediction_file = os.path.join(
            output_dir,
            "predictions.json" if prefix is None else f"predictions_{prefix}".json,
        )
        #n-best 저장
        nbest_file = os.path.join(
            output_dir,
            "nbest_predictions.json"
            if prefix is None
            else f"nbest_predictions_{prefix}".json,
        )
        #'답변없음' 사용하는 경우 저장
        if version_2_with_negative:
            null_odds_file = os.path.join(
                output_dir,
                "null_odds.json" if prefix is None else f"null_odds_{prefix}".json,
            )

        logger.info(f"Saving predictions to {prediction_file}.") #예측 저장 시작 기록
        #예측 파일 w모드로 열기
        with open(prediction_file, "w", encoding="utf-8") as writer:
            writer.write(
                json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n"
            )
        logger.info(f"Saving nbest_preds to {nbest_file}.")
        with open(nbest_file, "w", encoding="utf-8") as writer:
            writer.write(
                json.dumps(all_nbest_json, indent=4, ensure_ascii=False) + "\n"
            )
        if version_2_with_negative:
            logger.info(f"Saving null_odds to {null_odds_file}.")
            with open(null_odds_file, "w", encoding="utf-8") as writer:
                writer.write(
                    json.dumps(scores_diff_json, indent=4, ensure_ascii=False) + "\n"
                )

    return all_predictions


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed))) # Round to the nearest second.
    return str(datetime.timedelta(seconds=elapsed_rounded)) # Format as hh:mm:ss


def get_topk_accuracy(faiss_index, answer_idx, positive_idx, text=None, title=None): 

    top1_correct = 0
    top5_correct = 0
    top10_correct = 0
    top20_correct = 0
    top30_correct = 0
    top50_correct = 0
    top100_correct = 0
    
    for idx, answer in enumerate(answer_idx):
        
        #  *** faiss index: (question num * k) ***
        #      [[73587  2746 15265 96434 ...]
        #       [98388 13550 93912 92610 ...]
        #                    ...
        #       [97530 93498 16607 98168 ...]
        #       [52308 24908 70869 20824 ...]
        #       [44597 35140  7572  4596 ...]]
         
        retrieved_idx = faiss_index[idx] 
        retrieved_idx = [positive_idx[jdx] for jdx in retrieved_idx]
        if any(ridx in answer for ridx in retrieved_idx[:1]):
            top1_correct += 1
        if any(ridx in answer for ridx in retrieved_idx[:5]):
            top5_correct += 1
        if any(ridx in answer for ridx in retrieved_idx[:10]):
            top10_correct += 1
        if any(ridx in answer for ridx in retrieved_idx[:20]):
            top20_correct += 1
        if any(ridx in answer for ridx in retrieved_idx[:30]):
            top30_correct += 1
        if any(ridx in answer for ridx in retrieved_idx[:50]):
            top50_correct += 1
        if any(ridx in answer for ridx in retrieved_idx[:100]):
            top100_correct += 1
        
    top1_accuracy = top1_correct / len(answer_idx)
    top5_accuracy = top5_correct / len(answer_idx)
    top10_accuracy = top10_correct / len(answer_idx)    
    top20_accuracy = top20_correct / len(answer_idx)
    top30_accuracy = top30_correct / len(answer_idx)
    top50_accuracy = top50_correct / len(answer_idx)
    top100_accuracy = top100_correct / len(answer_idx)

    return {
        'top1_accuracy':top1_accuracy,
        'top5_accuracy':top5_accuracy,
        'top10_accuracy':top10_accuracy,        
        'top20_accuracy':top20_accuracy,
        'top30_accuracy':top30_accuracy,
        'top50_accuracy':top50_accuracy,
        'top100_accuracy':top100_accuracy,
    }
    
def get_topk_document(faiss_index, question, question_idx, positive_idx, text=None, title=None, save_dir=None): 

    result = []    
    for idx, q_idx in enumerate(question_idx):
        retrieved_idx = faiss_index[idx] 
        retrieved_idx = [positive_idx[jdx] for jdx in retrieved_idx]
        candidate_corpus = retrieved_idx[:40]
        
        context = []
        for x in candidate_corpus:
            clean_text = preprocessor.preprocess_pipeline(text[x])
            corpus = title[x] + '. ' + clean_text
            context.append(corpus)
            
        tmp = {
            # Query와 해당 id를 반환합니다.
            "question": question[idx],
            "id": q_idx,
            # Retrieve한 Passage의 id, context를 반환합니다.
            "context": "\n\n".join(context),
        }
        result.append(tmp)
    cqas = pd.DataFrame(result)
    print(cqas)
    f = Features(
        {
            "context": Value(dtype="string", id=None),
            "id": Value(dtype="string", id=None),
            "question": Value(dtype="string", id=None),
        }
    )
    datasets = DatasetDict({"validation": Dataset.from_pandas(cqas, features=f)})
    datasets.save_to_disk(save_dir)