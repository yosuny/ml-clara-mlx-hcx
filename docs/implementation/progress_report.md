# HCX Omni 8B 모델 학습 진행 상황 및 수정 내역

## 1. 개요 (Overview)
MacBook(Apple Silicon) 환경에서 CLaRa 프레임워크를 사용하여 HCX Omni 8B 모델의 Pretraining을 수행하기 위한 환경 구축 및 트러블슈팅 기록입니다.

## 2. 환경 설정 (Environment Setup)
- **가상 환경**: `.venv` 생성 (Python 3.9).
- **의존성 설치**:
    - `requirements_mac.txt` 생성: `triton` 등 Mac에서 지원하지 않는 CUDA 관련 패키지 제거.
    - `llama-cpp-python`, `deepspeed` 등의 Native Extension 빌드 및 설치.

## 3. 주요 코드 수정 내역 (Key Code Modifications)

### A. 호환성 패치 (Compatibility Patches)
1.  **`openrlhf/models/ring_attn_utils.py`**
    - **Flash Attention**: Mac에서 지원되지 않는 `flash_attn` 모듈 import를 `try-except` 구문으로 감싸 선택적으로 로드하도록 수정.
    - **Type Hinting**: Python 3.9 환경 호환을 위해 `|` 연산자 대신 `typing.Union` 사용으로 수정.

2.  **`openrlhf/utils/deepspeed/deepspeed.py`**
    - **CUDA 호출 제거**: 코드 전반에 산재된 `torch.cuda.set_device` 등의 호출이 GPU가 없는 환경(MPS/CPU)에서도 에러를 발생시키지 않도록 `if torch.cuda.is_available():` 조건 추가.

### B. 기능 확장 (Feature Extensions)
3.  **`openrlhf/cli/train_sft.py`**
    - **Quantization 지원**: 커스텀 모델 로딩을 지원하기 위해 `--quantization` 인자를 추가 (`int4`, `int8` 옵션). `train_sft` 실행 시 동적으로 양자화 설정을 적용할 수 있도록 개선.

### C. 실행 스크립트 (Execution Script)
4.  **`scripts/train_pretraining_hcx.sh`**
    - **네트워크 설정**: Mac의 분산 학습 네트워크 문제(`Gloo` 타임아웃) 해결을 위해 로컬 루프백(`lo0`) 강제 설정.
    - **프로세스 설정**: `torchrun` 대신 `python -m` 단일 프로세스 실행 방식으로 변경하여 분산 초기화 오류 우회.
    - **메모리 최적화**: 24GB 메모리에 맞춰 `batch_size` 4, `max_len` 1024로 조정.
    - **포트 충돌 방지**: `MASTER_PORT`를 29501로 변경.

## 4. 학습 시도 및 트러블슈팅 (Training Attempts & Troubleshooting)

### 시도 1: 분산 학습 모드
- **증상**: `torchrun` 실행 시 네트워크 타임아웃 및 주소 할당 에러.
- **조치**: 단일 프로세스 모드로 전환 및 명시적 환경 변수(`RANK`, `WORLD_SIZE` 등) 설정.

### 시도 2: 기본 모델 (Base Model) 로딩
- **모델 경로**: `.../HyperCLOVAX-SEED-Omni-8B-Text`
- **결과**: **실패 (Corrupted)**
- **원인**: 모델 디렉토리에 가중치 파일(`model-00001`, `00002` 등) 및 인덱스 파일이 누락됨.

### 시도 3: 4-bit 양자화 모델 로딩
- **모델 경로**: `.../HyperCLOVAX-SEED-Omni-8B-Text-4bit`
- **결과**: **실패 (Incompatible Config)**
- **원인**: `config.json`의 양자화 설정이 Transformers 라이브러리 표준과 호환되지 않음. `auto-gptq` 등을 통한 우회 시도했으나 라이브러리 호환성 문제 발생.

### 시도 4: 8-bit 양자화 모델 로딩
- **모델 경로**: `.../HyperCLOVAX-SEED-Omni-8B-Text-8bit`
- **설정**: `--quantization int8` 플래그 사용.
- **결과**: **실패 (Shape Mismatch)**
- **원인**: 가중치를 강제로 로드했으나, 텐서 크기 불일치 발생 (`1024` vs `4096`). 모델 파일이 손상되었거나 예상과 다른 아키텍처일 가능성 높음.

## 5. 현재 상태 및 해결 방안 (Current Status & Next Steps)
- **현재 상황**: 학습 스크립트와 코드는 Mac 환경에서 정상 작동하도록 준비되었으나, **유효한 모델 파일 부재**로 인해 학습 시작이 차단됨.
- **필요 조치**:
    1.  **정상적인 모델 다운로드**: Base Model의 모든 가중치 파일(shard)이 온전한지 확인 필요.
    2.  **호환 가능한 양자화 모델**: Transformers 라이브러리와 호환되는 표준 GGUF 또는 GPTQ 형식이 있다면 해당 파일 사용 권장.

## 6. Apple Silicon 최적화 (MLX Migration)
Mac 환경에서의 성능 극대화를 위해 Apple **MLX 프레임워크**로의 전환을 결정하고 진행 중입니다.

### Phase 1: 모델 검증 (완료 - 수정됨)
- **발견**: 기존 `HyperCLOVAX-SEED-Omni-8B-Text-4bit` 모델이 이미 MLX 호환 포맷(config에 quantization 설정 포함)임을 확인.
- **조치**: 중복 변환 작업을 중단하고 기존 모델을 사용하여 다음 단계 진행 결정.
- **검증**: `mlx_lm.load` 및 `generate` 테스트 성공.

### Phase 2: 아키텍처 포팅 (완료)
- **작업**: `openrlhf/models/modeling_clara_mlx.py` 작성.
- **내용**:
    - `CLaRa` 클래스 구현 (MLX `nn.Module` 상속).
    - `compress`: Memory Token 추출 및 MSE Loss 계산 로직 (Slicing 방식 적용).
    - `compress_query`: Query Reasoning을 위한 임베딩 압축 로직 구현.
- **검증**: `scripts/test_clara_mlx.py`를 통해 더미 데이터 기준 정상 동작 확인.

### Phase 3: 학습 루프 구현 (완료)
- **작업**: `scripts/train_mlx.py` 작성 및 `CLaRa` 모델용 LoRA 적용 로직 구현.
- **내용**:
    - `mlx_lm`의 `LoRALinear`를 사용하여 4-bit Quantized 모델의 Attention Layer(q_proj, v_proj)에 어댑터 수동 주입.
    - `mem_token_ids`를 모델 파라미터에서 제외하여 Gradient 계산 오류 해결.
    - Functional Update 방식(`model.update(params)`)을 적용하여 `mx.value_and_grad` 호환성 확보.
- **결과**: `training_step` 정상 실행 확인 (Step 1 Loss: 3.37, Step 2 Loss: 2.81).

## 결론
CLaRa 모델의 MLX 포팅이 성공적으로 완료되었습니다.
- **Base Model**: `HyperCLOVAX-SEED-Omni-8B-Text-4bit` (MLX 호환 확인)
- **Architecture**: `openrlhf/models/modeling_clara_mlx.py` (압축 및 추론 로직 구현)
- **Training**: `scripts/train_mlx.py` (LoRA Fine-tuning 루프 구현)

이제 Mac 환경에서 `scripts/train_mlx.py`를 실행하여 본격적인 학습 및 실험이 가능합니다.

