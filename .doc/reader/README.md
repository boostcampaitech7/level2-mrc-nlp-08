# 🔥 네이버 AI Tech NLP 8조 The AIluminator 🌟
## Level 2 Project - Open-Domain Question Answering


## Reader improvement
### Transfer Learning
- **배경**
    - 2009년 ICML에서 발표된 논문의 "Curriculum learning"에 따라 Transfer Learning을 효과적으로 적용하고자 함
- **Curriculum Learning**
    - 사람과 유사하게 기계에 Curriculum을 만들어 주어 더 쉬운 데이터부터 학습시키고 차츰 어려운 데이터를 순차적으로 학습시키는 방식을 의미
    - ex) Shape Recognition Task를 학습시킬 때, 쉬운 데이터로 원, 정사각형 등을 학습시키고 어려운 데이터로 타원, 직사각형 등 학습 <br>

- **오픈 데이터셋 추가**
    -  KoQuAD 1.0 : 한국어 MRC를 위해 만든 데이터셋
        -  해당 데이터셋은 Wikipedia article의 일부 하위 영역에 대한 질문과 답변으로 이루어져 있음
        -  Stanford Question Answering Dataset(SQuAD) v1.0과 동일한 방식으로 구성
        ```
            Context = "덴노지 역 구내에서는 야마토지 선과 평면 교차하여 단선으로 운전하는 한와 선과의 단락선을 복선화하는 공사가 이루어져
         2008년 3월 15일 시간표 개정시부터 사용을 개시하였다. 대낮 시간대의 덴노지 역 ~ 와카야마 역 간의 쾌속과 JR 난바 역
         발착의 간쿠 쾌속이 오사카 순환선과 직통 운전하는 간쿠·기슈지 쾌속에 통합되어 같은 시간대의 간쿠·기슈지 쾌속이 매시
         3편으로 증발되었다. 이 복선화는 야마토지 선의 시간표 혼란이 큰 영향을 끼치는 것을 막는 효과를 가지고 있다.
        또 아침 출근 시간대에 오사카 순환선 내 각역에 정차하는 직통 쾌속도 운전이 개시되었다.
        전반적으로 히네노 역 이북에서는 8량 편성의 쾌속 열차가 대폭 증가되었기 때문에 난카이 전기 철도 본선과 수송력에서 차이가 나게 되었다."
            
            Question = "야마토지 선의 시간표가 더 간단해지게 된 이유는 무엇일까?",  
            Answer = "복선화"
        ```
    - ai-hub 데이터 추가 : 뉴스 기사 기계독해 데이터
      - 해당 데이터셋은 뉴스 기사 context와 Question과 Answer로 구성
        - 추출형 : 지문 내에서 답을 도출할 수 잇는 질문과 그 답변으로 이루어진 질의응답
        - **추론형** : 지문 내 정보를 이용하여 답을 유추하도록 하는 질문과 근거, 답변으로 이루어진 질의응답 셋으로, 답에 대해 지문에 직접적으로 명시되지 않은 내용으로 질문을 제작한다는 점에서 추출형과 구분, 추론형은 일반 상식의 조합이나 연역 추론 등의 방식을 활용하여 제작 <br>

    → 이 중 Curriculum Learning에 반영할 난이도를 생각했을 때, 추론형이 더 학습하기 어려운 데이터이므로 추론형을 선택하여 전이 학습에 사용

- **평가**
    - 증강된 데이터를 활용하였을 때, validation에서는 크게 개선되지 않았지만, **F1-score와 Public Score가 많이 향상된 것을 확인**

    <img src="https://github.com/user-attachments/assets/197ea107-b49b-45e7-b0ff-aab25ae0dff6"/>

### CNN Layer
- **배경**
    - 기존 Pre-trained 모델의 출력은 주로 Linear Layer를 거쳐 처리되지만, 이는 인접한 토큰들 간의 연관된 정보를 충분히 학습하기에 부족
    - 삼성 SDS에서 제안한 KorQuAD 1.0 성능 개선 방안1) 에 따르면, 1D Convolution을 추가하면 Kernel Size에 해당하는 인접 토큰들의 임베딩 벡터 간의 연관성을 학습하여 보다 풍부한 문맥 정보를 얻을 수 있다는 것을 확인
- **1차원 Convolution**
    - 1D Convolution은 2D Convolution 와 달리 커널의 높이가 임베딩 차원과 같고, 너비만 kernel size로 설정
    - 결과적으로, 시퀀스의 각 위치에서 해당 범위 내 단어들의 정보를 종합한 특징 벡터가 생성

      <img src="https://github.com/user-attachments/assets/2d0dc473-19c9-49d3-99eb-a107ed14e974"/> 
- **CNN Layer 적용 구조**
    - Kernel Size 3과 1을 갖는 Conv 레이어와 Residual Connection을 통해 인접 토큰 간의 연관성 학습과 동시에 모델이 깊어짐에 따른 학습 부담을 완화

      <img src="https://github.com/user-attachments/assets/d7016bf4-fc64-48a0-8432-c88a184995d5"/>
- **평가**
    - kpfbert-korquad-1 : Eval_F1 : 4.66 , Eval_EM : 5.00 상승
    - roberta-large-qa-korquad-v1: Eval_F1 : 2.84 , Eval_EM : 2.9 상승, Public_EM : 7.08, public_F1 : 7.5 감소 
    
    → 학습 성능이 높아졌지만 오버피팅 가능성이 높아짐을 확인

    <img src="https://github.com/user-attachments/assets/fdf81d6e-0bd4-4077-8fc6-cb36f77c15e9"/>

### Cleaning
- **배경**
    - 대회에서 주어진 Query-Passage 쌍 데이터의 context가 KQUAD, AI Hub 등 데이터셋에 비해 노이즈가 많이 끼어 있어, 해당 부분에서 노이즈를 줄일 필요성을 느낌
    - 해당 데이터의 context가 증강된 데이터와 유사해질수록, 증강된 데이터로 학습이 더 용이해 질 것이라고 가정
- **방안**
    - Context에서 답이 지워지지 않도록 노이즈를 제거하는 코드를 작성
    - 해당 코드를 대회에서 주어진 train 데이터셋, validation 데이터셋에 적용하고, retriever를 통해 불러오는 context 들에 대해서도 적용함
- **예시**
    - **날짜 정보 제거**:
        - 원본: "낸다. 이러한 방식으로 단원제가 빠지기 쉬운 함정을 미리 방지하는 것이다.날짜=2017-02-05"
        - 처리 후: "낸다. 이러한 방식으로 단원제가 빠지기 쉬운 함정을 미리 방지하는 것이다."
    
    - **인용문 제거**:
        - 원본: "연장시킬 뿐이라고 했다.문학의 숲 고전의 바다 元老의 誤判] 조선일보 2004.03.05</ref>"
        - 처리 후: "연장시킬 뿐이라고 했다."
    
    - **섬네일 정보 제거**:
        - 원본: "섬네일|left|페데리코 다 몬테펠트로의 구비오 스투디올로의 모습.\n\n"
        - 처리 후: "페데리코 다 몬테펠트로의 구비오 스투디올로의 모습."
    
    - **질문-답변 불일치 수정**:
        - 원본 질문: "스타필드 고양점이 개업한 날은?"
        - 원본 답변: "3호점인 스타필드 고양"
        - 수정 후 답변: "2017년 8월 24일" (본문에서 실제 개업일을 찾아 수정)

    - **답변 형식 통일**:
        - 이외에 Answer 형식을 통일하려고 시도하였으나, 규칙성을 찾아내지 못하였음

- **평가**
    - Clean 되기 이전의 데이터를 사용했을 때에 비해 Clean 데이터를 사용하였을 때 모델의 성능이 전반적으로 오른다는 것을 확인
    - Data Aaugmentation 진행한 결과에 따른 4가지 version의 train data와 Model exploration&Modeling을 거쳐 선정된 model에 다양한 조합으로 실험하여 최적의 성능 도출
    - **각 기법마다 best case에 대해서 비교해본 결과 min-max 평균을 취한 case가 가장 높은 92.98의 public pearson 값을 가지는 것을 확인하고 이를 최종 리더보드에 제출**

    <img src="https://github.com/user-attachments/assets/ab4f25b9-5571-477e-9aed-bf0234154905"/>

