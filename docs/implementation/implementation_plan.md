# 원본 Repo 데이터 학습 + PDF Evaluation 계획

## 전략 개요

### 학습 (Training)
- ✅ **원본 repo 데이터 사용**: `example/` 디렉토리의 기존 데이터
- ✅ **원본 스크립트 활용**: PyTorch 기반 학습 스크립트
- ✅ **3-Stage 파이프라인**: Stage 1 → Stage 2 → Stage 3

### 평가 (Evaluation)  
- ✅ **CLaRa 논문 PDF 전용**: `knowledge_data/2511.18659v2.pdf`
- ✅ **평가 데이터 생성**: PDF에서 QA 쌍 추출
- ✅ **성능 측정**: 학습된 모델의 논문 이해도 평가

---

## Phase 7: 원본 Repo 데이터 확인 및 준비

### 기존 데이터 현황

**발견된 파일**:
```bash
example/end_to_end_data.jsonl  # Stage 3용 데이터
```

**데이터 형식 확인 필요**:
```bash
# Stage 1 데이터
example/stage1_*.jsonl

# Stage 2 데이터  
example/stage2_*.jsonl

# Stage 3 데이터
example/end_to_end_data.jsonl  # ✅ 존재 확인됨
```

### 필요 작업

**1. 원본 데이터 파일 확인**
```bash
ls -la example/
ls -la data/
```

**2. 데이터 형식 검증**
```python
# 각 stage별 데이터 형식 확인
import json

# Stage 1 예상 형식
{
    "data_type": "qa",
    "question": ["Q1", "Q2"],
    "answers": ["A1", "A2"],
    "docs": ["Doc1", "Doc2"]
}

# Stage 2 예상 형식
{
    "question": "What is X?",
    "docs": ["Doc1", "Doc2"],
    "gold_answer": "X is..."
}

# Stage 3 예상 형식 (확인됨)
{
    "question": "...",
    "docs": ["..."],
    "gold_answer": "..."
}
```

---

## Phase 8: 3-Stage Training (원본 Repo 방식)

### Stage 1: Compression Pretraining

**목표**: 문서 압축 능력 학습

**데이터**: 원본 repo의 Stage 1 데이터
- 위치: `example/stage1_*.jsonl` (확인 필요)
- 대안: 기존 QA 데이터셋 활용

**학습 방법**:

**Option A: 원본 PyTorch 스크립트 사용**
```bash
# 원본 repo의 학습 스크립트
bash scripts/train_pretraining.sh
```

**Option B: MLX 버전 작성**
```python
# scripts/train_stage1_mlx.py
# - 원본 데이터 형식 지원
# - MSE + QA loss
# - 가중치 저장
```

**출력**:
- `checkpoints/stage1/compressor.npz`

---

### Stage 2: Compression Instruction Tuning

**목표**: 압축된 벡터로 답변 생성 학습

**데이터**: 원본 repo의 Stage 2 데이터
- 위치: `example/stage2_*.jsonl` (확인 필요)

**핵심 구현**:
```python
# 압축 벡터를 입력으로 사용
def forward_stage2(question, docs):
    # 1. 문서 압축 (Stage 1 compressor, frozen)
    compressed = [compressor.compress(doc) for doc in docs]
    
    # 2. 압축 벡터 + 질문을 입력으로
    input_emb = concat([embed(question), compressed])
    
    # 3. 답변 생성
    answer = generator(input_emb)
    return answer
```

**학습 방법**:
```bash
bash scripts/train_instruction_tuning.sh \
    --stage1_checkpoint checkpoints/stage1/compressor.npz
```

**출력**:
- `checkpoints/stage2/generator.npz`

---

### Stage 3: End-to-End Training

**목표**: 검색 + 압축 + 생성 통합 학습

**데이터**: `example/end_to_end_data.jsonl` ✅

**학습 방법**:
```bash
bash scripts/train_stage_end_to_end.sh \
    --stage2_checkpoint checkpoints/stage2/generator.npz
```

**출력**:
- `checkpoints/stage3/full_system.npz`

---

## Phase 9: CLaRa 논문 PDF 기반 Evaluation

### 평가 데이터 생성

**목표**: PDF에서 평가용 QA 쌍 생성

**방법 1: 수동 생성 (고품질)**
```python
# data/clara_paper_eval.jsonl
[
    {
        "id": 1,
        "question": "What is CLaRa?",
        "reference_answer": "CLaRa is a compression-based RAG system...",
        "category": "factual"
    },
    {
        "id": 2,
        "question": "What are the three training stages?",
        "reference_answer": "Stage 1: Compression Pretraining, Stage 2: Instruction Tuning, Stage 3: End-to-End",
        "category": "detailed"
    },
    # ... 20-30개 질문
]
```

**방법 2: 자동 생성**
```python
# scripts/generate_eval_qa_from_pdf.py
import json
from mlx_lm import load, generate

def extract_sections(pdf_text):
    """PDF를 섹션별로 분할"""
    sections = {
        "introduction": "...",
        "method": "...",
        "experiments": "...",
        "conclusion": "..."
    }
    return sections

def generate_questions(section_text, num_q=5):
    """LLM으로 질문 생성"""
    model, tokenizer = load("model_path")
    
    prompt = f"""Based on this text, generate {num_q} evaluation questions.

Text: {section_text}

Requirements:
- Questions should test understanding of key concepts
- Include factual, technical, and analytical questions
- Provide reference answers

Format as JSON:
[
  {{"question": "...", "answer": "...", "category": "factual"}},
  ...
]

Generate:"""
    
    response = generate(model, tokenizer, prompt, max_tokens=1000)
    return parse_json(response)

# 실행
pdf_text = load_pdf("knowledge_data/2511.18659v2.pdf")
sections = extract_sections(pdf_text)

all_questions = []
for section_name, section_text in sections.items():
    questions = generate_questions(section_text, num_q=5)
    all_questions.extend(questions)

# 저장
with open("data/clara_paper_eval.jsonl", "w") as f:
    for q in all_questions:
        f.write(json.dumps(q) + "\n")
```

---

### 평가 시나리오

**Scenario 1: Zero-shot QA**
```python
# 학습된 모델에 논문 관련 질문
# (논문 내용은 학습 데이터에 없음)

for question in eval_questions:
    # Stage 3 모델 사용
    answer = model.generate(question, context=None)
    
    # 정확도 측정
    score = evaluate(answer, reference_answer)
```

**Scenario 2: RAG with PDF**
```python
# PDF를 문서 풀로 사용

pdf_chunks = split_pdf_into_chunks(pdf_text)

for question in eval_questions:
    # 1. 검색
    retrieved = retriever.retrieve(question, pdf_chunks, k=5)
    
    # 2. 압축
    compressed = [compressor.compress(doc) for doc in retrieved]
    
    # 3. 생성
    answer = generator(question, compressed)
    
    # 평가
    score = evaluate(answer, reference_answer)
```

**Scenario 3: Compression Quality**
```python
# PDF 섹션 압축 품질 평가

for section in pdf_sections:
    # 압축
    compressed = compressor.compress(section)
    
    # 복원 (optional)
    reconstructed = compressor.reconstruct(compressed)
    
    # 품질 측정
    mse = compute_mse(section, reconstructed)
    similarity = compute_similarity(section, reconstructed)
```

---

### 평가 메트릭

**1. QA 정확도**
```python
# Exact Match
em_score = (predicted == reference).mean()

# F1 Score
f1_score = compute_f1(predicted, reference)

# Keyword Match
keyword_match = compute_keyword_overlap(predicted, reference)
```

**2. 압축 품질**
```python
# MSE Loss
mse = nn.losses.mse_loss(compressed, original)

# Semantic Similarity
similarity = cosine_similarity(
    embed(original), 
    embed(reconstructed)
)
```

**3. End-to-End 성능**
```python
# Retrieval Accuracy
retrieval_acc = evaluate_retrieval(retrieved, gold_docs)

# Generation Quality
gen_quality = evaluate_generation(answer, reference)

# Overall Score
overall = alpha * retrieval_acc + beta * gen_quality
```

---

## 실행 로드맵

### Week 1: 데이터 준비 및 Stage 1

**Day 1-2: 데이터 확인**
- [ ] 원본 repo 데이터 파일 확인
- [ ] 각 stage별 데이터 형식 검증
- [ ] 누락된 데이터 생성 (필요시)

**Day 3-4: Stage 1 학습**
- [ ] 원본 스크립트 실행 또는 MLX 버전 작성
- [ ] 학습 실행
- [ ] 가중치 저장 및 검증

**Day 5: Stage 1 평가**
- [ ] 압축 품질 측정
- [ ] 중간 결과 분석

---

### Week 2: Stage 2-3 및 Evaluation

**Day 6-7: Stage 2**
- [ ] 압축 벡터 입력 구현
- [ ] Stage 2 학습
- [ ] 가중치 저장

**Day 8-9: Stage 3**
- [ ] Retriever 구현 (간단한 버전)
- [ ] End-to-End 학습
- [ ] 전체 시스템 통합

**Day 10: PDF Evaluation**
- [ ] CLaRa 논문 QA 데이터 생성
- [ ] 평가 실행
- [ ] 결과 분석 및 리포트

---

## 디렉토리 구조

```
ml-clara/
├── example/                    # 원본 학습 데이터
│   ├── stage1_*.jsonl         # Stage 1 데이터 (확인 필요)
│   ├── stage2_*.jsonl         # Stage 2 데이터 (확인 필요)
│   └── end_to_end_data.jsonl  # Stage 3 데이터 ✅
│
├── knowledge_data/             # Evaluation 전용
│   └── 2511.18659v2.pdf       # CLaRa 논문 (평가용)
│
├── data/                       # 생성된 데이터
│   └── clara_paper_eval.jsonl # PDF 기반 평가 데이터
│
├── checkpoints/                # 학습된 모델
│   ├── stage1/
│   │   └── compressor.npz
│   ├── stage2/
│   │   └── generator.npz
│   └── stage3/
│       └── full_system.npz
│
└── results/                    # 평가 결과
    ├── stage1_eval.json
    ├── stage2_eval.json
    ├── stage3_eval.json
    └── final_report.md
```

---

## 핵심 포인트

### 학습 (Training)
✅ **원본 데이터 사용**
- `example/` 디렉토리의 기존 데이터
- 원본 repo의 검증된 데이터셋

✅ **3-Stage 순차 학습**
- Stage 1: 압축 학습
- Stage 2: 압축 벡터 이해
- Stage 3: End-to-End 통합

### 평가 (Evaluation)
✅ **PDF 전용 사용**
- 학습에 사용 안함 (zero-shot 평가)
- 평가 데이터 생성용
- 모델의 일반화 능력 측정

✅ **공정한 평가**
- 학습 데이터와 분리
- 실제 논문 이해도 측정
- 다양한 메트릭 사용

---

## 다음 단계

**즉시 실행**:
1. 원본 repo 데이터 파일 확인
2. Stage 1 데이터 준비
3. Stage 1 학습 시작

**선택 사항**:
- PDF 평가 데이터 미리 생성
- Evaluation 스크립트 작성

이 계획으로 진행하시겠습니까?
