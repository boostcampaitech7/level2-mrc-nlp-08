# 🔥 네이버 AI Tech NLP 8조 The AIluminator 🌟
## Level 2 Project - Open-Domain Question Answering

## EDA
데이터는 train_dataset, test_dataset의 2개의 DatasetDict로 되어있으며 각 파일의 구성은 다음과 같다. <br>
| 분류(디렉토리 명)|세부 분류|샘플 수|용도|공개여부|
|:-:|:-:|:-:|:-:|:-:|
|train|train|3,952|학습용|모든 정보 공개(id, question, context, answers, document_id, title)|
|-|valid|240|학습용|모든 정보 공개(id, question, context, answers, document_id, title)|
|test|valid|240 (Public)|제출용|id, question 만 공개|
|-|-|360 (Private)|제출용|id, question 만 공개|
|Wiki, Doc|corpus|60,613|제출용|모든 정보 공개 (text, corpus_source, url, title, document_id)


**title** : context 제목 <br>
**context** : 문단 <br>
**question** : 질문 <br>
**id** : context, question 쌍 고유 id <br>
**answers** : {answer_start: 문단 내 시작위치, text: 정답} <br>
**document_id** : 문단 id <br>

### 중복 데이터 확인
- Query-Passage 쌍 데이터 : context 기준, Train 2,761개(1,191↓) / Valid 230개(10↓)
- Wiki. Doc : 56,737개(3,801↓) <br> <br>

### 토큰 별 분포
- Query-Passage 쌍 데이터의 Text들에 대한 토큰을 세 종류의 Tokenizer (BPE, SentenecPiece, WordPiece)를 통해 확인
<img src="https://github.com/user-attachments/assets/4dfd39a3-d18d-483c-b1f4-9fe0fe3ba02f"/> <br>

- Wiki. Doc 의 Text들에 대한 토큰을 세 종류의 토크나이저 (BPE, SentenecPiece, WordPiece)를 통해 확인
<img src="https://github.com/user-attachments/assets/eb02949a-6a3b-4668-b1db-2c9e9b434702"/> <br>

- Train, Validation set 모두 최대 1,500 이하이며 비슷한 분포를 가짐
- Wiki set : boxplot을 통해 outlier가 존재함을 확인
<img src="https://github.com/user-attachments/assets/a0a0c7e7-6983-4a5d-b73b-2a10779e202d"/> <br>
<br>

### 데이터 퀄리티 체크
- 각 데이터셋에서 한글이 아닌 문자(영어, 한자, url, html 태그, 특수문자 등) 개수 파악, text에 한번이라도 포함되면 count <br>
<img src="https://github.com/user-attachments/assets/27ff50e7-5bc5-495c-8f4f-b2947a39e14c"/>
