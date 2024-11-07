# 🔥 네이버 AI Tech NLP 8조 The AIluminator 🌟
## Level 2 Project - Open-Domain Question Answering

## 목차
1. [프로젝트 소개](#1-프로젝트-소개)
2. [Installation and Quick Start](#2-installation-and-quick-start)
3. [프로젝트 진행](#3-프로젝트-진행)
4. [리더보드 결과](#4-리더보드-결과)

# 1. 프로젝트 소개
(1) 주제 및 목표
- 부스트캠프 AI Tech NLP 트랙 level 2 MRC
- 주제 : ODQA (Open-Domain Question Answering)    
      ODQA 데이터셋을 활용해 질문에 맞는 정답을 예측  <br>

(2) 평가지표
- 주 평가지표 : Exact Match (모델의 예측과 실제 답이 정확하게 일치할 때만 점수가 주어짐) <br>
- 참고용 : F1 score (모델의 예측과 실제 답에 겹치는 부분이 있으면 부분점수가 주어짐) <br>

(3) 개발 환경 <br>
- GPU : Tesla V100 * 4 <br>

(4) 협업 환경
|**Tool**|**Description**|
|:-:|-|
|**GitHub**|- Task 별 issue 생성<br>- 담당한 issue에 대한 branch 생성 후 PR & main에 merge|
|**Slack**| - GitHub과 연동해서 레포지토리에 업데이트 되는 내용 실시간으로 확인<br>- 허들을 이용한 회의 및 결과 공유 |
|**Notion**| - 타임라인 정리<br>- 칸반보드를 이용한 task 관리 |
|**Zoom**| - 진행상황 공유 |
|**WandB**| - Sweep을 통한 하이퍼 파라미터 최적화 |

(5) 팀원 소개
|김동한|김성훈|김수아|김현욱|송수빈|신수환|
|:--:|:--:|:--:|:--:|:--:|:--:|
|<img src="https://github.com/user-attachments/assets/c7d1807e-ef20-4c82-9a88-bc0eb5a700f4" width="100" height="100" />|<img src="https://github.com/user-attachments/assets/62829d6a-13c9-40dd-807a-116347c1de11" width="100" height="100" />|<img src="https://github.com/user-attachments/assets/5933a9e6-b5b8-41df-b050-c0a89ec19607" width="100" height="100" />|<img src="https://github.com/user-attachments/assets/c90f4226-3bea-41d9-8b28-4d6227c1d254" width="100" height="100" />|<img src="https://github.com/user-attachments/assets/65a7e762-b018-41fc-88f0-45d959c0effa" width="100" height="100" />|<img src="https://github.com/user-attachments/assets/8d806852-764d-499b-a780-018b6cf32b8d" width="100" height="100" />|
|[Github](https://github.com/dongspam0209)|[Github](https://github.com/sunghoon014)|[Github](https://github.com/tndkkim)|[Github](https://github.com/hwk9764)|[Github](https://github.com/suvinn)|[Github](https://github.com/kkobugi)| -->|

|**Member**|**Team**|**Role**|
|:--|--|--|
|**김동한**|Data, Model|- **Extraction Reader Modeling**(학습 및 추론)<br>- **Extraction Reader 아키텍처 수정**(CNN Head)<br>- **Sparse Passage Retrieval**(Retrieval 결과 분석)<br>- **EDA**(데이터 토큰 개수 분포 분석)|
|**김성훈**|Data, Model|- **Code Modularization**, **Sparse/Dense Passage Rrieval**(구현 및 실험), **Generation Reader Modeling**(LLM 학습 및 실험), **ML Pipeline**|
|**김수아**|Model|- **Question augmentation**(KoBART)<br>- **Experimentation**(top-k)|
|**김현욱**|Data, Model|- **Generation Reader Modeling**(학습 및 추론)<br>- **EDA**(데이터 텍스트 퀄리티 분석)|
|**송수빈**|Model|- **Extraction Reader Modeling**(학습 및 추론)<br>- **Experimentation**(실험 모델 목록 구성 및 결과 정리)<br>- **Logging & HyperParameter Tuning**(Wandb Sweep)<br>- **Ensemble**(앙상블 코드 작성, 모델 선정을 위한 상관관계 분석 코드 작성)|
|**신수환**|Data, Model|**Sparse Passage Retrieval**(BM25 성능 개선), **데이터 전처리**(Data Cleaning)|
<br>


# 2. Installation and Quick Start
**Step 1.** 프로젝트에 필요한 모든 dependencies는 `requirements.txt`에 있고, 이에 대한 가상환경을 생성해서 프로젝트를 실행
```sh
# 가상환경 만들기
$ python -m venv .venv

# 가상환경 켜기
$ . .venv/bin/activate

# 제공되는 서버 환경에 따라 선택적 사용
$ export TMPDIR=/data/ephemeral/tmp 
$ mkdir -p $TMPDIR

# 필요 라이브러리 설치
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

**Step 2.** Pre Processing 실행
 ```sh
# 작업환경 변경
$ cd pre_process

# 다음 주피터를 따라가며 KorQuAD 1.0 데이터 증강
$ data_augment_korquadv1.ipynb

# 다음 주피터를 따라가며 AIHub 데이터 증강
$ data_augment_aihub.ipynb

# 다음 주피터를 따라가며 DPR retrieval을 학습하기 위한 데이터 만들기
$ generate_DPR_dataset_korquad.ipynb
```

**Step 3.** DPR 모델 학습

**utils/arguments_dpr.py** 에서 DPR 학습을 위한 파라미터 변경
- model : 원하는 사전 학습된 모델 불러오기
- train_data : generate_DPR_dataset_korquad.ipynb 에서 생성한 데이터 경로
- valid_data : generate_DPR_dataset_korquad.ipynb 에서 생성한 데이터 경로
- q_output_path : Query embedding 모델 저장할 경로
- c_output_path : Context embedding 모델 저장할 경로
```sh
# ./level2-mrc-nlp-08 경로에서 실행
$ python train_dpr.py
```



**Step 4.** Retrieval를 위한 사전처리 진행

**database/python get_embedding_vec.csv** : BM25 모델 및 DPR의 embedding vector 저장
- model : 학습된 context embedding 모델 경로
- wiki_path : Wiki.doc 데이터 경로 
- valid_data : Query-Passage 쌍 데이터의 검증 데이터 경로
- save_path : Embedding vector 저장 경로

**test_retrieval.py**
- model : 학습된 query embedding 모델 경로
- valid_data : Query-Passage 쌍 데이터의 검증 데이터 경로
- faiss_path : **database/python get_embedding_vec.csv** 에서 실행한 save_path 경로
- bm25_path : **database/python get_embedding_vec.csv** 에서 실행한 save_path 경로
- context_path : **database/python get_embedding_vec.csv** 에서 실행한 save_path 경로

**test_retrieval_inference.py**
- model : 학습된 query embedding 모델 경로
- test_dataset : Query-Passage 쌍 데이터의 테스트 데이터 경로
- faiss_path : 위와 동일
- bm25_path : 위와 동일
- context_path : 위와 동일

```sh
# 작업환경 변경
$ cd database

# 다음 코드를 실행하여 embedding vector 추출
$ python get_embedding_vec.csv

# BM25 및 DPR 성능 확인
$ cd ..
$ python test_retrieval.py

# Inference 시 사용할 retireve 된 데이터 생성
$ python test_retrieval_inference.py
```


**Step 5.** Reader 학습

**utils/arguments_extraction_reader.py**에서 extracion based model 학습을 위한 파라미터 변경
- model_name_or_path : 사전 학습된 모델 불러오기
- dataset_name : Query-Passage 쌍 데이터나 증강된 데이터 경로로 변경
- output_dir : 학습된 모델 및 평가 결과 저장 경로

```sh
# 다음 코드를 실행하여 extraction based model 학습
$ python train_extraction_reader.py

# 프로젝트 때는 사용하지 않았지만 generation based model 학습, 파라미터 변경은 위와 동일
$ python train_generation_reader_Seq2SeqLM,.py
$ python train_generation_reader_CausalLM,.py
```



**Step 6.** Inference 실행

**utils/arguments_inference.py**에서 inference 할 extraction based 모델의 파라미터 변경
- model_name_or_path : 학습이 완료된 모델 불러오기
- output_dir : Inference 결과 저장 경로

```sh
# 코드 50번째 줄에서 retireve 된 데이터 불러오는 경로 원하는 것으로 변경하면서 사용
$ python inference.py
```


**Step 7.** 앙상블 실행
```sh
# train_extraction_reader 실행 시 생성되는 predictions.json 값들로 상관분석 분석
$ correlation_exp.ipynb

# 상관분석을 통해 사용할 모델 선택 되었다면 그 모델들로 inference 시 생성된 nbest_predictions.json 파일들로 앙상블 진행 / 두가지 버전 모두 사용 가능
$ ensemble_v1.ipynb
$ ensemble_v2.ipynb

```

# 3. 프로젝트 진행
<img src="https://github.com/user-attachments/assets/daf4c4ea-2463-426c-9964-939b5c793937"/>


| Task | **Task Description** |
| --- | --- |
| **[EDA](https://github.com/boostcampaitech7/level2-mrc-nlp-08/tree/main/.doc/eda)** | 데이터의 특성을 살펴보기 위해 중복 데이터 확인, 토큰 개수 분포, 데이터 퀄리티 체크 등 시각화 및 분석 |
| **[베이스라인 모델](https://github.com/boostcampaitech7/level2-mrc-nlp-08/tree/main/.doc/baseline)** | Reader Model로 사용하기 적합한 pre-trained model 실험 및 선정 |
| **[Retrieval](https://github.com/boostcampaitech7/level2-mrc-nlp-08/tree/main/.doc/retrieval)** | BM25, DPR Retrieval 기법 구현 및 실험 |
| **[Reader Model](https://github.com/boostcampaitech7/level2-mrc-nlp-08/tree/main/.doc/reader)** | Transfer Learning <br> CNN Head <br> Cleaning|
| **Post-Processing** | 후처리 <br> 모델 다양성 체크 <br> 앙상블 |


## Post-Processing
### Inference 후처리
- 통합모델이 최선의 답을 도출할 때, 문서 내에 다른 위치에 있는 같은 단어임에도 start logit과 end logit 값이 달라 각 위치에 대한 확률이 분리되어 계산되는 현상이 발생하여 Inference 후처리 진행
- 텍스트가 동일한 경우 확률을 합산해 총 확률을 기반으로 답변을 선택하는 후처리 과정을 적용함

### 모델 다양성 체크
- 모델의 예측값을 벡터로 치환하여(오답을 1, 정답을 0) **모델 간의 상관관계**를 분석함으로써, 모델들이 상호보완적인 작용을 하도록 함
- model1, model2이 있고 각각 5개를 예측했다고 하면 두 벡터 [1, 0, 0, 1, 1], [0, 1, 1, 0, 1]의 상관관계를 도출

### 앙상블
- 다양하게 훈련된 여러 모델을 결합하여 모델들이 서로를 보완하여 더 좋은 결과를 낼 수 있도록 하기 위해 앙상블을 도입
- 확률합을 통해 soft voting (앞서 이야기 한 후처리 방식과 동일)
    - 앙상블 할 모델들의 답변-확률 값을 불러오고 같은 단어에 대한 확률들을 sum
    - 가장 높은 확률의 답변을 정답으로 채택

- 다수결 (majority voting)
    - 앙상블 할 모델들의 답변-확률 값을 불러오고 가장 빈도수가 높은 답변을 정답으로 채택
    - 만약 동률의 답변이 있다면, 앞선 다수결 결과와 상관없이 확률이 가장 높은 답변을 채택
      <img src="https://github.com/user-attachments/assets/b722741c-2879-49ed-aa75-72d49aadd864"/>

# 4. 리더보드 결과
**Public Leader Board 순위**   

<img src="https://github.com/user-attachments/assets/d2d828ff-e443-4a9a-a111-d8e4b8453cc8"/>


**Private Leader Board 순위**

<img src="https://github.com/user-attachments/assets/3c87d052-9734-4c90-8c81-9c186c877bdf"/>

