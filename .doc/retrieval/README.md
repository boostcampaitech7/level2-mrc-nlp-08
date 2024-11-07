# 🔥 네이버 AI Tech NLP 8조 The AIluminator 🌟
## Level 2 Project - Open-Domain Question Answering

## Retrieval 개선
### Sparse Passage Retrieval
- **BM25 사용**
    - BM25는 문서와 쿼리 간의 관련성을 측정하는 통계적 모델
    - TF-IDF(역문헌빈도) 개념을 기반으로 하며, 아래 3가지 조건을 만족할수록 더 큰 점수 부여
      1. 문서 내용에 단어 출현 빈도가 높을수록
         - f(t, d) : 문서 d에서 단어 t가 등장한 횟수
         - k : saturation parameter, 단어 t 출현 수(f)가 한계치에 도달하면 더 이상 출현 수로 인한 점수 부여는 하지 않기 위한 파라미터
      2. 문서 내용이 짧을수록
      3. 다른 문서에는 단어가 출현하지 않을수록 (IDF 계산하는 것, 단어가 제공하는 정보의 양)
         - df : 단어 t가 출현한 문서 수
  <img src="https://github.com/user-attachments/assets/e0f47280-e584-49cd-8509-3f8f2540ec05"/>

- **BM25 성능 고도화**
  - **가설 설정**
    - BM25는 "키워드"의 출현 빈도가 점수 계산에 큰 영향을 끼치므로 문장에서 "키워드"를 어떻게 추출하느냐에 따라 성능에 달라질 것으로 판단
  - **문제 해결 과정**
    - Subword-Level Tokenization(BPE, WordPiece 등)을 이용하여 문장을 토큰으로 나누어 "키워드" 추출
    - 한국어의 어휘와 문법적인 요소를 반영하여 "키워드"를 추출하기 위해 한국어 형태소 분석기 활용
    - 한국어 형태소 분석기는 벤치마크 실험 결과1)에 따라 OKT (Open Korean Text) 형태소 분석기를 사용
    - 문장을 형태소로 나누었을 때 "키워드"를 담고 있다고 생각되는 형태소만을 남김
   
  -  형태소 분류표
  
    | 대분류       | 형태소  | 설명      |
    |--------------|---------|-----------|
    | 체언 (N)     | NNG     | 일반 명사 |
    |              | NNP     | 고유 명사 |
    |              | NNB     | 의존 명사 |
    |              | NR      | 수사      |
    |              | NP      | 대명사    |

    | 대분류       | 형태소  | 설명      |
    |--------------|---------|-----------|
    | 용언 (V)     | VV      | 동사      |
    |              | VA      | 형용사    |
    
    | 대분류       | 형태소  | 설명      |
    |--------------|---------|-----------|
    | 어근         | XR      | 어근      |
    | 부호         | SN      | 숫자      |
    |              | SH      | 한자      |
    |              | SL      | 알파벳    |

  
### Dense Passage Retrieval
- **DPR 사용**
    - Facebook AI에서 개발한 모델로, 문서와 질문을 밀집 벡터(Dense Vector)로 변환한 후, 이 벡터 간의 유사도를 계산해 관련 문서를 찾는 모델
    - BM25의 다음과 같은 한계를 보완하기 위해 DPR 모델을 활용
        - 단어 기반 매칭 : 쿼리와 문서 간의 일치 여부는 단어가 정확히 일치해야 유사도가 높게 계산되어 동의어 또는 의미상으로 유사한 단어를 처리하는 데 한계가 존재
        - 문맥 정보 부족 : 개별 단어의 빈도와 희소성(Sparse)에 기반하여 점수를 계산하기 때문에, 단어가 문장에서 사용된 의미나 뉘앙스를 잘 반영하지 못함
    - **Model Architecture**
        - Question (query)에 대응되는 Encoder 1
        - Wiki. Document (passage)에 대응되는  Encoder 2
        - 각 sentence를 encoding 했을 때 `[cls]` 토큰에 해당하는 vector 𝒉
        - 각 Encoder의 𝒉 를 dot-product 하여 유사도 계산
      <img src="https://github.com/user-attachments/assets/3bc133c4-5b4e-4751-877f-d705337d61d7"/>


- **DPR 학습과정**
    - **학습 목표**
        - 연관된 question 과 passage dense embedding 간의 거리를 좁히는 것
    - **학습 데이터 구축**
        - 학습을 위해서는 대회에서 제공된 데이터셋과 같은 연관성이 높은 question-passage 쌍 필요
            - Training set : 대회 학습 데이터(3,952개) + KoQuAD 1.0 학습 데이터 (60,407개)
            - Validation set : 대회 검증 데이터(240개)
   - **Object Function**
       - 어떠한 질문 $𝑞_𝑖$ ​, 이와 관련된(=positive) 지문  $𝑝_𝑖^+$  , 관련이 없는(=negative) n개의 지문  $𝑝_(𝑖,𝑗)^−$  로 이루어져 있음
       -  NLL (Negative Log-Likelihood) Loss를 사용해서 최적화

          $$L(q_i, p_i^+, p_{i,1}^-, \dots, p_{i,n}^-) = -\log \frac{e^{sim(q_i, p_i^+)}}{e^{sim(q_i, p_i^+)} + \sum_{j=1}^{n} e^{sim(q_i, p_{i,j}^-)}} $$
   - **Hard negative passages**
     - 가장 높은 BM25 점수를 지니는 n개의 passage를 negative passage로 사용
   - **In-batch negatives**
     - mini-batch를 1개는 positive, n개는 negative로 구성
     - Batch Size가 3일 때, 두 encoder의 embedding으로 (3 x 768) * (768 x 3) = (3 x 3)의 similarity matrix를 만들 수 있음
     - 주대각선이 서로 대응하는 질문과 텍스트의 유사도에 해당
      <img src="https://github.com/user-attachments/assets/c8d6dd18-45f2-49a1-a2b6-88dd81cf141c"/>

### Retrieval 평가 결과
<img src="https://github.com/user-attachments/assets/87f6179f-3cb7-4f2d-959a-4d99e5a66974"/>

