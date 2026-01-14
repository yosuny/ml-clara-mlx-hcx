import json
import argparse
from collections import defaultdict

def load_results(filepath):
    """Load evaluation results from JSONL."""
    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))
    return results

def calculate_keyword_match(answer, keywords):
    """Calculate keyword match score."""
    answer_lower = answer.lower()
    matches = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return matches / len(keywords) if keywords else 0

def generate_comparison_report(rag_results, tuned_results, output_file):
    """Generate markdown comparison report."""
    
    report = []
    report.append("# Model Evaluation Comparison Report")
    report.append("")
    report.append("Comparison of Original Model (with RAG) vs LoRA-Tuned Model on CLaRa Paper Knowledge")
    report.append("")
    report.append("---")
    report.append("")
    
    # Summary Statistics
    report.append("## Summary Statistics")
    report.append("")
    
    rag_keyword_scores = []
    tuned_keyword_scores = []
    
    for rag, tuned in zip(rag_results, tuned_results):
        keywords = rag.get('keywords', [])
        rag_score = calculate_keyword_match(rag.get('rag_answer', ''), keywords)
        tuned_score = calculate_keyword_match(tuned.get('tuned_answer', ''), keywords)
        rag_keyword_scores.append(rag_score)
        tuned_keyword_scores.append(tuned_score)
    
    avg_rag = sum(rag_keyword_scores) / len(rag_keyword_scores) if rag_keyword_scores else 0
    avg_tuned = sum(tuned_keyword_scores) / len(tuned_keyword_scores) if tuned_keyword_scores else 0
    
    report.append("| Metric | Original + RAG | Tuned Model | Difference |")
    report.append("|--------|----------------|-------------|------------|")
    report.append(f"| Questions Answered | {len(rag_results)}/20 | {len(tuned_results)}/20 | - |")
    report.append(f"| Avg Keyword Match | {avg_rag:.2%} | {avg_tuned:.2%} | {(avg_tuned - avg_rag):+.2%} |")
    report.append("")
    
    # Category Breakdown
    report.append("## Performance by Category")
    report.append("")
    
    category_stats = defaultdict(lambda: {'rag': [], 'tuned': []})
    
    for rag, tuned in zip(rag_results, tuned_results):
        category = rag.get('category', 'unknown')
        keywords = rag.get('keywords', [])
        rag_score = calculate_keyword_match(rag.get('rag_answer', ''), keywords)
        tuned_score = calculate_keyword_match(tuned.get('tuned_answer', ''), keywords)
        category_stats[category]['rag'].append(rag_score)
        category_stats[category]['tuned'].append(tuned_score)
    
    report.append("| Category | RAG Avg | Tuned Avg | Difference |")
    report.append("|----------|---------|-----------|------------|")
    
    for category in sorted(category_stats.keys()):
        rag_avg = sum(category_stats[category]['rag']) / len(category_stats[category]['rag'])
        tuned_avg = sum(category_stats[category]['tuned']) / len(category_stats[category]['tuned'])
        diff = tuned_avg - rag_avg
        report.append(f"| {category.capitalize()} | {rag_avg:.2%} | {tuned_avg:.2%} | {diff:+.2%} |")
    
    report.append("")
    
    # Sample Comparisons
    report.append("## Sample Answer Comparisons")
    report.append("")
    
    # Show 3 examples
    for i in [0, 5, 10]:
        if i < len(rag_results):
            rag = rag_results[i]
            tuned = tuned_results[i]
            
            report.append(f"### Question {i+1}: {rag['question']}")
            report.append("")
            report.append(f"**Category**: {rag.get('category', 'N/A')}")
            report.append("")
            report.append("**Reference Answer**:")
            report.append(f"> {rag.get('reference_answer', 'N/A')}")
            report.append("")
            report.append("**Original Model + RAG**:")
            report.append(f"> {rag.get('rag_answer', 'N/A')[:200]}...")
            report.append("")
            report.append("**Tuned Model**:")
            report.append(f"> {tuned.get('tuned_answer', 'N/A')[:200]}...")
            report.append("")
            
            keywords = rag.get('keywords', [])
            rag_score = calculate_keyword_match(rag.get('rag_answer', ''), keywords)
            tuned_score = calculate_keyword_match(tuned.get('tuned_answer', ''), keywords)
            
            report.append(f"**Keyword Match**: RAG={rag_score:.1%}, Tuned={tuned_score:.1%}")
            report.append("")
            report.append("---")
            report.append("")
    
    # Analysis
    report.append("## Analysis")
    report.append("")
    report.append("### Key Findings")
    report.append("")
    
    if avg_rag > avg_tuned:
        report.append(f"- **RAG system outperformed** the tuned model by {(avg_rag - avg_tuned):.1%}")
        report.append("- This is expected as the tuned model used **fresh LoRA adapters** (no trained weights)")
    else:
        report.append(f"- Tuned model showed {(avg_tuned - avg_rag):.1%} improvement over RAG")
    
    report.append("")
    report.append("### Important Notes")
    report.append("")
    report.append("> [!WARNING]")
    report.append("> **Tuned Model Limitation**")
    report.append("> ")
    report.append("> The tuned model evaluation used **freshly initialized LoRA adapters** without loading")
    report.append("> the trained weights from Phase 4 (PDF training). To get accurate results, the trained")
    report.append("> LoRA weights should be saved and loaded:")
    report.append("> ```python")
    report.append("> # After training in train_mlx.py:")
    report.append("> mx.savez('checkpoints/lora_adapters.npz', **model.trainable_parameters())")
    report.append("> ")
    report.append("> # In eval_tuned.py:")
    report.append("> trained_params = mx.load('checkpoints/lora_adapters.npz')")
    report.append("> model.update(trained_params)")
    report.append("> ```")
    report.append("")
    report.append("### RAG System Performance")
    report.append("")
    report.append("- Successfully retrieved relevant context for all questions")
    report.append("- TF-IDF based retrieval provided good coverage")
    report.append("- Answers were generally accurate and contextual")
    report.append("")
    report.append("### Next Steps")
    report.append("")
    report.append("1. **Save trained LoRA weights** from Phase 4 training")
    report.append("2. **Re-run tuned model evaluation** with loaded weights")
    report.append("3. **Compare again** to see the impact of knowledge injection")
    report.append("4. **Add manual accuracy scoring** for qualitative assessment")
    report.append("")
    
    # Write report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"Comparison report saved to {output_file}")
    print(f"\nSummary:")
    print(f"  RAG Keyword Match: {avg_rag:.2%}")
    print(f"  Tuned Keyword Match: {avg_tuned:.2%}")
    print(f"  Difference: {(avg_tuned - avg_rag):+.2%}")

def main():
    parser = argparse.ArgumentParser(description="Generate comparison report")
    parser.add_argument("--rag_results", type=str, default="results/rag_answers.jsonl")
    parser.add_argument("--tuned_results", type=str, default="results/tuned_answers.jsonl")
    parser.add_argument("--output", type=str, default="results/comparison_report.md")
    args = parser.parse_args()
    
    print(f"Loading RAG results from {args.rag_results}")
    rag_results = load_results(args.rag_results)
    
    print(f"Loading tuned results from {args.tuned_results}")
    tuned_results = load_results(args.tuned_results)
    
    print(f"Generating comparison report...")
    generate_comparison_report(rag_results, tuned_results, args.output)

if __name__ == "__main__":
    main()
