# Azure AI Search 상품 추천 시스템

창주스토어 상품 데이터를 기반으로 한 Azure AI Search 검색 전략 비교 및 최적화 프로젝트입니다.

## 프로젝트 구조

```
├── index.ipynb                        # 인덱스 생성 및 데이터 업로드
├── search.ipynb                       # 검색 전략 테스트 및 비교
├── sample_products.csv                # 상품 데이터
├── fashion_data_generator.py          # AI 기반 패션 데이터 생성기
├── upload_fashion_to_search.py        # 패션 데이터 Azure AI Search 업로드
├── upload_to_search.py                # 일반 상품 데이터 업로드
├── run_fashion_workflow.py            # 패션 데이터 생성 및 업로드 통합 워크플로우
└── semantic_tagging_filtering_with_synthetic_data.ipynb  # 패션 데이터 태깅 및 필터링 예제
```

## 검색 전략

다음과 같은 5가지 검색 전략을 구현하고 비교합니다:

### 1. Vector-only Search
- Ada-002 embedding (1536차원) 또는 Large-3 embedding (3072차원) 단독 사용
- 순수 의미적 유사성 기반 검색

### 2. Multi-Vector Search  
- Ada-002와 Large-3 embedding을 동시에 활용
- 다중 벡터 모델의 장점 결합

### 2.5. Vector + Semantic Reranker
- 벡터 검색 후 Semantic Reranker로 재순위화
- BM25 키워드 검색 없이 순수 의미 검색

### 3. Hybrid Search (BM25 + Vector + Semantic)
- 키워드 검색(BM25) + 벡터 검색 + Semantic Reranker
- 키워드 매칭과 의미적 유사성의 균형

### 4. Query Rewrite + Hybrid + Semantic
- AI 기반 쿼리 개선 + 하이브리드 검색 + 시맨틱 리랭킹
- 최고 성능을 위한 모든 기능 조합

## 환경 설정

### 필수 패키지

```bash
pip install -r requirements.txt
```

### 환경 변수 설정

`.env` 파일을 생성하고 다음 변수들을 설정하세요:

```bash
# Azure Search 설정
AZURE_SEARCH_SERVICE_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_INDEX=your-search-index-name
AZURE_SEARCH_ADMIN_KEY=your-search-admin-key

# Azure OpenAI 임베딩 설정
AZURE_OPENAI_ENDPOINT=https://your-openai-service.openai.azure.com/
AZURE_OPENAI_KEY=your-openai-api-key
AZURE_OPENAI_ADA002_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
AZURE_OPENAI_3_LARGE_EMBEDDING_DEPLOYMENT=text-embedding-3-large
```

## 데이터 구조

### 상품 데이터 (sample_products.csv)

창주스토어 상품 데이터의 주요 필드:
- `id`: 상품 고유 ID
- `title`: 상품명
- `brand`: 브랜드명
- `category`: 카테고리 (유아동, 패션의류, 이미용, 스포츠/레저 등)
- `normal_price`: 정가
- `image_link`: 상품 이미지 링크
- `keyword`: 검색 키워드

### Azure AI Search 인덱스 구조

생성되는 인덱스의 주요 필드:
- `main_text`: 통합 검색 텍스트 (title + brand + category + keyword)
- `main_text_vector`: Ada-002 임베딩 벡터 (1536차원)
- `main_text_vector_3`: Large-3 임베딩 벡터 (3072차원)
- `product_group_code`: 카테고리별 필터링용 코드
- `price_range`: 가격대별 분류 (1만원대, 2만원대 등)
- Semantic Search 구성: `my-semantic-config`

## 실행 방법

### 1. 환경 설정
```bash
# 패키지 설치
pip install -r requirements.txt

# 환경 변수 파일 설정
# .env 파일에서 실제 Azure 서비스 정보로 수정
```

### 2. 인덱스 생성 (index.ipynb)
1. 환경 변수 로드
2. Azure Search 인덱스 스키마 정의
3. 인덱스 생성 (기존 인덱스가 있으면 삭제 후 재생성)
4. 상품 데이터 읽기 및 전처리
5. Ada-002와 Large-3 임베딩 생성
6. 배치 업로드로 인덱스에 데이터 저장

### 3. 패션 데이터 생성 및 업로드 (신규 기능)

#### 방법 1: 통합 워크플로우 (권장)
```bash
# 패션 데이터 생성 + 자동 업로드 + 임베딩 생성
python run_fashion_workflow.py --count 20 --upload --embeddings

# 패션 데이터만 생성 (업로드 없이)
python run_fashion_workflow.py --count 10
```

#### 방법 2: 단계별 실행
```bash
# 1단계: 패션 데이터 생성
python fashion_data_generator.py --count 20 --language ko --output fashion_products.json

# 2단계: Azure AI Search에 업로드 (임베딩 포함)
python upload_fashion_to_search.py --input fashion_products.json --create_embeddings

# 또는 임베딩 없이 빠른 업로드
python upload_fashion_to_search.py --input fashion_products.json
```

### 4. 검색 테스트 (search.ipynb)
1. 검색 클라이언트 초기화
2. 검색어 및 카테고리 필터 설정
3. 5가지 검색 전략 실행 및 결과 비교:
   - Vector-only Search
   - Multi-Vector Search  
   - Vector + Semantic Reranker
   - Hybrid + Semantic Search
   - Query Rewrite + Hybrid + Semantic
4. 검색 결과 분석 및 성능 비교


## 주요 기능

### 1. 기존 상품 데이터 검색
sample_products.csv 기반 창주스토어 상품 검색 시스템

### 2. AI 기반 패션 데이터 생성 (신규)
- **fashion_data_generator.py**: 웹 검색 기반 패션 제품 정보 생성
  - 브랜드별 제품 코드 기반 웹 검색
  - AI를 활용한 패션 메타데이터 자동 생성
  - 스타일, 색상, 소재, 계절, 타겟 고객 등 자동 태깅
  - 한국어/영어 제품 설명 생성

- **upload_fashion_to_search.py**: 패션 데이터 전용 업로드
  - fashion_data_generator.py 출력을 shopping-sample 인덱스 스키마로 변환
  - Ada-002 및 Large-3 임베딩 자동 생성
  - 배치 업로드 및 검증

### 3. 통합 워크플로우
- **run_fashion_workflow.py**: 패션 데이터 생성부터 업로드까지 원스톱 처리

### 검색 전략별 특징
- **Vector-only**: 순수 의미적 유사성 기반, 동의어/유사 표현에 강함
- **Multi-Vector**: 다중 임베딩 모델 조합으로 검색 정확도 향상
- **Vector + Semantic**: 벡터 검색 + AI 기반 재순위화
- **Hybrid**: 키워드 매칭 + 의미 검색의 균형적 조합
- **Query Rewrite**: AI 기반 쿼리 개선으로 최적화된 검색


## 알려진 이슈

### SDK 호환성
- Query Rewrite 기능은 Preview API 버전이 필요함 (`2025-08-01-preview`)
- 현재 SDK 버전에서는 `query_rewrites` 파라미터 호환성 문제가 있을 수 있음

### 인덱스 관리
- 인덱스 스키마 변경 시 기존 인덱스 삭제 후 재생성 필요
- 대용량 데이터 업로드 시 배치 처리로 안정성 확보

