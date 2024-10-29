# 🔥 네이버 AI Tech NLP 8조 The AIluminator 🌟
## Level 2 Project - Open-Domain Question Answering

## 베이스라인 모델
### Model Exploration
- 베이스라인으로 주어진 코드를 기반으로 HuggingFace의 model 허브에서 “question answering”으로 필터링한 후 Klue로 사전학습 되지 않은 모델에 대해 기초 성능 평가를 진행. <br>
Reader : Query-Passage 쌍 데이터로 학습 한 Extraction based Reader
<img src="https://github.com/user-attachments/assets/6c82984b-76b0-4e02-964a-6e3b67cc977c"/>


### Model Tuning
Retrieval Tuning : TF-IDF <br>
- Retrieval 단계에서 가져오는 문서의 개수(k)가 모델이 정답을 맞출 확률을 높이는 데 미치는 영향을 분석
<img src="https://github.com/user-attachments/assets/6c82984b-76b0-4e02-964a-6e3b67cc977c"/>
<br>
Parameter Tuning <br>
- 과적합을 방지하기 위해 Loss function에 L2 penalty를 좀 더 많이 적용
- 적절한 답변의 길이를 찾기
<img src="https://github.com/user-attachments/assets/ead0a1f5-0945-4a4b-a361-6c6e626ca63f"/>
