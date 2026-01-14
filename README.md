# ml-clara-mlx-hcx: HyperCLOVA X Seed Experiment

This repository is a **Hard Fork** of the original [CLaRa (Continuous Latent Reasoning)](https://arxiv.org/abs/2511.18659) project, adapted for the **HyperCLOVA X Seed (HCX-Seed)** model and optimized for **Apple Silicon (M-series)** using the [MLX framework](https://github.com/ml-explore/mlx).

## 🍎 Project Goal
The primary objective of this project is to experimentally verify the feasibility of the CLaRa (Retrieval-Augmented Generation with Compression) architecture on the **HCX-Seed 8B** model directly on Mac hardware. By porting the core components from PyTorch to MLX, we aim to:
- **Validate HCX-Seed Capability**: Test the internal knowledge retention and reasoning of the HCX-Seed model using CLaRa's compression mechanism.
- **Native Metal Performance**: Enable efficient training and inference on M1/M2/M3 chips.
- **Unified Memory Architecture**: Leverage Apple's unified memory to handle the 8B model without dedicated VRAM constraints.

## 📂 Repository Structure

- **`models/`**: Contains the ported CLaRa model architecture (`modeling_clara_mlx.py`), completely rewritten for MLX.
- **`scripts/`**: MLX-native training and evaluation scripts.
    - `train_stage1_mlx.py`: Stage 1 (Compression Pretraining)
    - `train_stage2_mlx.py`: Stage 2 (Instruction Tuning)
    - `train_stage3_mlx.py`: Stage 3 (End-to-End Joint Training)
    - `eval_clara.py`: End-to-End Evaluation Pipeline
- **`docs/implementation/`**: Comprehensive documentation of the porting process, implementation plans, and verification logs.
- **`data/` & `checkpoints/`**: Directories for datasets and trained weights.
- **`original_project/`**: The complete original PyTorch implementation (frozen reference).

## 🚀 Getting Started

### Prerequisites
- macOS 13.0+ (Ventura or later)
- Python 3.9+
- `mlx`, `mlx_lm`, `transformers`, `huggingface_hub`

### Installation
```bash
pip install -r requirements_mac.txt
```

### Running the Pipeline
You can run the full 3-stage training pipeline directly on your Mac:

```bash
# Stage 1: Train Compressor
python scripts/train_stage1_mlx.py --data_file example/pretrain_data.jsonl

# Stage 2: Instruction Tuning
python scripts/train_stage2_mlx.py --data_file example/instruction_tuning_data.jsonl

# Stage 3: Joint Training (End-to-End)
python scripts/train_stage3_mlx.py --data_file example/end_to_end_data.jsonl
```

### Evaluation
Evaluate the internal knowledge injection vs RAG performance:
```bash
python scripts/eval_clara.py
```

## 📄 Results & Verification
Detailed verification results and "dry-run" logs can be found in `docs/implementation/walkthrough.md`. The port has been technically verified to produce correct loss convergence and successful file I/O for all three training stages.

---
*Based on: [CLaRa: Continuous Latent Reasoning for Knowledge-Rich Tasks](https://arxiv.org/abs/2511.18659)*
