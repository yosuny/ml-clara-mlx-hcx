# Apple MLX 마이그레이션 계획 (Migration Plan)

본 문서는 CLaRa 프로젝트를 Apple의 MLX 프레임워크로 포팅하여 MacBook(Apple Silicon)에서의 학습 성능을 극대화하기 위한 상세 계획입니다.

## 1. 목표 (Goal)
- **성능**: PyTorch MPS 백엔드 대비 2배 이상의 학습 속도 향상.
- **효율성**: 통합 메모리 활용 최적화로 더 큰 배치 사이즈 및 빠른 4-bit LoRA 학습 지원.
- **단순화**: 복잡한 분산 처리(DeepSpeed) 로직을 제거하고 단일 기기 최적화 코드 적용.

## 2. 작업 단계 (Execution Phases)

### Phase 1: 모델 변환 (Model Conversion)
### Phase 1: 모델 변환 (완료/생략)
- **상태**: 기존 `HyperCLOVAX-SEED-Omni-8B-Text-4bit` 모델이 이미 MLX 호환 포맷임을 확인하여 변환 과정을 생략하고 해당 모델을 바로 사용합니다.
- **검증**: `mlx_lm` 로드 및 생성 테스트 완료.

### Phase 2: CLaRa 아키텍처 포팅 (Architecture Porting) & 검증
CLaRa 핵심 로직인 "Compressor(압축기)" 기능을 MLX로 재구현해야 합니다.

1.  **`modeling_clara_mlx.py` 작성**:
    -   `transformers.PreTrainedModel` 대신 `mlx.nn.Module`을 상속받는 `CLaRa` 클래스 정의.
    -   `_compr_decoder` 메서드 포팅: PyTorch의 hidden state 추출 로직을 MLX 문법으로 변환.
2.  **Inference 검증**:
    -   PyTorch 원본 코드와 MLX 포팅 코드의 출력값(Logits) 비교 검증 (허용 오차 범위 내 일치 확인).

### Phase 3: 학습 루프 재구현 (Training Loop Implementation)
`torchrun`과 `DeepSpeed`를 제거하고 MLX 네이티브 학습 코드를 작성합니다.

1.  **`train_sft_mlx.py` 작성**:
    -   **Optimizer**: `mlx.optimizers.AdamW` 사용.
    -   **Loss Function**: `mlx.nn.losses.cross_entropy` 사용 (CLaRa의 QA loss + Compressor loss 구현).
    -   **LoRA**: `mlx.nn.LoRA` 레이어를 활용하여 모델 전체 미세조정 대신 LoRA 어댑터 학습 적용.
    -   **Gradient Checkpointing**: `mlx.core.fast.affine_quant` 등 메모리 최적화 기법 적용.

## 3. 예상 기술적 난관 및 해결책

| 난관 | 해결책 |
| :--- | :--- |
| **커스텀 아키텍처 지원** | `HyperCLOVAX`가 `Llama` 기반이라면 `mlx-lm`의 Llama 모델 정의를 활용하되, `rope_scaling` 등 특이 설정만 오버라이딩하여 해결. |
| **복합 Loss 구현** | CLaRa는 생성(QA) Loss와 압축(Compress) Loss를 동시에 계산함. MLX의 `mx.value_and_grad` 함수를 사용하여 다중 Loss에 대한 기울기 계산 그래프 구성. |
| **데이터 로딩** | Hugging Face `datasets` 라이브러리는 그대로 사용하되, `torch.utils.data.DataLoader` 대신 Python Iterator를 사용하여 MLX Array로 변환하는 경량 로더 구현. |

## 4. 실행 제안 (Action Item)
가장 먼저 **Phase 1 (모델 변환)** 가능 여부를 확인해야 합니다. HCX 모델이 `mlx-lm`에서 표준 Llama로 인식되어 변환되는지 테스트해보는 것을 1차 목표로 삼는 것을 추천합니다.
