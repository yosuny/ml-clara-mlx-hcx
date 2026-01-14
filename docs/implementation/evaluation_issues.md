# 평가 테스트 문제점 분석

## 발견된 주요 문제점

사용자의 의심이 정확합니다. 다음과 같은 심각한 문제점들이 확인되었습니다:

---

## 문제 1: 학습된 가중치 미적용 ❌

### 현재 상황
`eval_tuned.py` (Line 35-40):
```python
# Note: In a real scenario, we would load trained weights here:
# trained_params = mx.load("checkpoints/lora_adapters.npz")
# model.update(trained_params)

print("Note: Using freshly initialized LoRA (no trained weights loaded)")
```

**문제**: 튜닝 모델 평가 시 **학습된 가중치를 전혀 로드하지 않음**
- Phase 4에서 20 step 학습 (Loss 4.29 → 0.03)
- 하지만 학습된 LoRA 가중치를 저장하지 않음
- 평가 시 **새로 초기화된 LoRA**만 사용

**결과**: 튜닝 모델이 실제로는 **학습되지 않은 랜덤 가중치**로 평가됨

---

## 문제 2: 학습 과정이 SFT가 아닌 압축만 진행 ✅ (의심 정확)

### Phase 4 학습 내용 확인
`train_mlx.py` (Line 176-179):
```python
def loss_fn(params, input_ids, attention_mask):
    model.update(params)
    _, mse_loss = model.compress(input_ids, attention_mask)
    return mse_loss  # ← MSE Loss만 사용!
```

**확인된 사실**:
- ✅ **압축 학습만 진행** (MSE Loss for compression alignment)
- ❌ **SFT 학습 아님** (QA Loss 없음)
- ❌ **언어 모델링 학습 없음** (Next token prediction 없음)

### 학습 데이터
- PDF 텍스트를 토큰화
- 메모리 토큰 주입: `input_ids[:, -n_mem:] = mem_token_ids[0]`
- **압축 품질만 최적화** (메모리 토큰과 비메모리 토큰 간 MSE)

**결과**: 모델이 **질문 답변 능력을 학습하지 않음**, 단지 문서 압축만 학습

---

## 문제 3: CLaRa 프레임워크 적용 여부 ⚠️

### 현재 구현 상태

**학습 시 (`train_mlx.py`)**:
```python
model = CLaRa(base_model, model_config)  # ✅ CLaRa 래퍼 사용
apply_lora_manual(model, lora_layers)     # ✅ LoRA 적용
_, mse_loss = model.compress(...)         # ✅ compress() 메서드 사용
```

**평가 시 (`eval_tuned.py`)**:
```python
model = CLaRa(base_model, model_config)  # ✅ CLaRa 래퍼 생성
apply_lora_manual(model, lora_layers)     # ✅ LoRA 적용 (새 가중치)
return model.model, tokenizer            # ⚠️ 내부 base_model만 반환!
```

**문제점**:
1. CLaRa 래퍼는 생성되지만 **학습된 가중치가 로드되지 않음**
2. `model.model` (base LLM)만 반환하여 **CLaRa 압축 기능 미사용**
3. 평가 시 일반 LLM처럼 동작 (압축 없이 직접 생성)

---

## 문제 4: 평가 방식의 불일치

### RAG 평가
- ✅ PDF 텍스트에서 TF-IDF 검색
- ✅ 관련 청크 3개 제공
- ✅ Context 기반 답변 생성

### 튜닝 모델 평가
- ❌ Context 없이 직접 질문만 제공
- ❌ 학습된 지식 활용 불가 (가중치 미로드)
- ❌ 압축 메커니즘 미사용

**불공정한 비교**: RAG는 명시적 컨텍스트 제공, 튜닝 모델은 아무것도 없음

---

## 왜 튜닝 모델 성능이 낮은가?

### 예상했던 시나리오 (잘못된 가정)
1. PDF 논문으로 SFT 학습 → 논문 지식 내재화
2. 질문만으로 답변 가능 (지식이 가중치에 저장됨)
3. RAG보다 우수한 성능

### 실제 상황
1. ❌ **압축만 학습** (QA 학습 없음)
2. ❌ **가중치 미로드** (랜덤 초기화 상태)
3. ❌ **지식 내재화 안됨** (언어 모델링 학습 없음)
4. ✅ **당연히 성능 낮음** (학습 안된 모델)

---

## 정확한 비교를 위한 수정 방안

### 방안 1: 학습된 가중치 로드 (최소 수정)

#### Step 1: 학습 후 가중치 저장
`train_mlx.py` 마지막에 추가:
```python
# After training loop
print("Saving trained LoRA adapters...")
os.makedirs("checkpoints", exist_ok=True)
mx.savez("checkpoints/lora_adapters.npz", **model.trainable_parameters())
print("Saved to checkpoints/lora_adapters.npz")
```

#### Step 2: 평가 시 가중치 로드
`eval_tuned.py` 수정:
```python
def load_tuned_model(model_path, lora_layers=8, adapter_path="checkpoints/lora_adapters.npz"):
    # ... (기존 코드)
    
    # Load trained weights
    if os.path.exists(adapter_path):
        print(f"Loading trained LoRA adapters from {adapter_path}")
        trained_params = mx.load(adapter_path)
        model.update(trained_params)
        print("✅ Trained weights loaded successfully")
    else:
        print(f"⚠️ No trained weights found at {adapter_path}")
        print("Using freshly initialized LoRA")
    
    return model.model, tokenizer
```

**예상 결과**: 여전히 성능 낮음 (압축만 학습했으므로)

---

### 방안 2: 실제 SFT 학습 수행 (올바른 방법)

#### Option A: 언어 모델링 학습
```python
# train_mlx.py 수정
def loss_fn(params, input_ids, attention_mask):
    model.update(params)
    
    # Language modeling loss (next token prediction)
    logits = model.model(input_ids)
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    
    lm_loss = nn.losses.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        reduction='mean'
    )
    
    return lm_loss
```

#### Option B: 기존 `train_sft_mlx.py` 사용
- 이미 구현되어 있음
- QA + MSE 결합 Loss
- 하지만 **QA 데이터 필요** (question/answer 쌍)

**문제**: PDF 텍스트만 있고 QA 쌍이 없음

---

### 방안 3: 공정한 비교 설정

#### Option A: 둘 다 Context 제공
```python
# RAG: Context 3개 청크
# Tuned: 동일한 3개 청크를 압축하여 제공
```

#### Option B: 둘 다 Context 없이
```python
# RAG: 검색 없이 직접 답변 (불가능할 것)
# Tuned: 질문만으로 답변 (현재 방식)
```

---

## 결론

### 확인된 문제점
1. ✅ **학습이 SFT가 아닌 압축만 진행** (사용자 의심 정확)
2. ✅ **CLaRa 프레임워크는 적용되었으나 학습된 가중치 미로드**
3. ✅ **평가 방식 불공정** (RAG는 context 제공, tuned는 없음)
4. ✅ **언어 모델링 학습 없음** (지식 내재화 불가)

### 튜닝 모델 성능이 낮은 이유
- **학습된 가중치를 전혀 사용하지 않음** (랜덤 초기화 상태)
- **압축만 학습** (QA 능력 학습 안됨)
- **지식 내재화 안됨** (LM loss 없음)

### 권장 해결 방안
1. **즉시**: 학습된 가중치 저장/로드 구현
2. **단기**: 언어 모델링 loss로 재학습 (PDF 텍스트 활용)
3. **장기**: QA 데이터셋 생성 후 실제 SFT 수행

---

## 다음 단계

### 1단계: 가중치 저장/로드 구현 (30분)
- `train_mlx.py`에 저장 코드 추가
- `eval_tuned.py`에 로드 코드 추가
- 재학습 및 재평가

### 2단계: 언어 모델링 학습 (1시간)
- `train_mlx.py`를 LM loss로 수정
- PDF 텍스트로 재학습
- 평가 및 비교

### 3단계: 공정한 비교 (선택)
- RAG와 동일한 context 제공 방식 구현
- 또는 둘 다 context 없이 비교
