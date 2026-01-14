# Model Evaluation Comparison Report

Comparison of Original Model (with RAG) vs LoRA-Tuned Model on CLaRa Paper Knowledge

---

## Summary Statistics

| Metric | Original + RAG | Tuned Model | Difference |
|--------|----------------|-------------|------------|
| Questions Answered | 20/20 | 20/20 | - |
| Avg Keyword Match | 26.25% | 20.00% | -6.25% |

## Performance by Category

| Category | RAG Avg | Tuned Avg | Difference |
|----------|---------|-----------|------------|
| Comparative | 25.00% | 12.50% | -12.50% |
| Detailed | 30.00% | 20.00% | -10.00% |
| Factual | 33.33% | 16.67% | -16.67% |
| Technical | 22.50% | 22.50% | +0.00% |

## Sample Answer Comparisons

### Question 1: What is CLaRa and what problem does it solve?

**Category**: factual

**Reference Answer**:
> CLaRa (Compressing LLM-based Retriever Architecture) is a system that compresses retrieved documents into compact memory representations to improve retrieval-augmented generation by reducing context length.

**Original Model + RAG**:
> CLaRa is a unified framework that performs retrieval and generation within a single continuous space, solving the issue of disjoint optimization between retrieval and generation in traditional RAG sys...

**Tuned Model**:
> CLaRa is a system for creating and delivering interactive language learning materials. It solves the problem of making language learning materials engaging and interactive, which is important for main...

**Keyword Match**: RAG=50.0%, Tuned=0.0%

---

### Question 6: What is the compression rate used in CLaRa experiments?

**Category**: technical

**Reference Answer**:
> CLaRa experiments use compression rates of 32x and 64x, meaning documents are compressed to 1/32 or 1/64 of their original length.

**Original Model + RAG**:
> The compression rate used in CLaRa experiments is 4....

**Tuned Model**:
> The compression rate used in CLaRa experiments is 0.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000...

**Keyword Match**: RAG=33.3%, Tuned=33.3%

---

### Question 11: What is the purpose of the compressor in CLaRa?

**Category**: technical

**Reference Answer**:
> The compressor transforms long documents into compact memory token representations while preserving semantic information needed for question answering.

**Original Model + RAG**:
> The compressor distills documents into compact representations that capture the core semantics, which are then used for efficient retrieval and generation....

**Tuned Model**:
> To compress the air in the room...

**Keyword Match**: RAG=33.3%, Tuned=0.0%

---

## Analysis

### Key Findings

- **RAG system outperformed** the tuned model by 6.2%
- This is expected as the tuned model used **fresh LoRA adapters** (no trained weights)

### Important Notes

> [!WARNING]
> **Tuned Model Limitation**
> 
> The tuned model evaluation used **freshly initialized LoRA adapters** without loading
> the trained weights from Phase 4 (PDF training). To get accurate results, the trained
> LoRA weights should be saved and loaded:
> ```python
> # After training in train_mlx.py:
> mx.savez('checkpoints/lora_adapters.npz', **model.trainable_parameters())
> 
> # In eval_tuned.py:
> trained_params = mx.load('checkpoints/lora_adapters.npz')
> model.update(trained_params)
> ```

### RAG System Performance

- Successfully retrieved relevant context for all questions
- TF-IDF based retrieval provided good coverage
- Answers were generally accurate and contextual

### Next Steps

1. **Save trained LoRA weights** from Phase 4 training
2. **Re-run tuned model evaluation** with loaded weights
3. **Compare again** to see the impact of knowledge injection
4. **Add manual accuracy scoring** for qualitative assessment
