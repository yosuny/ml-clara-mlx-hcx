import mlx.core as mx
from mlx_lm import load, generate
import json
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openrlhf.models.modeling_clara_mlx import CLaRa, CLaRaConfig
from scripts.train_mlx import apply_lora_manual

def load_tuned_model(model_path, lora_layers=8):
    """Load model and apply LoRA (simulating trained model)."""
    print(f"Loading base model from {model_path}")
    base_model, tokenizer = load(model_path)
    base_model.freeze()
    
    # Setup CLaRa
    eos_id = getattr(tokenizer, "eos_token_id", 2) or 2
    model_config = CLaRaConfig(
        base_model_path=model_path,
        doc_max_length=128,
        compr_rate=4,
        mem_token_ids=[eos_id]
    )
    
    print("Initializing CLaRa architecture...")
    model = CLaRa(base_model, model_config)
    
    # Apply LoRA to simulate trained model
    print(f"Applying LoRA to last {lora_layers} layers...")
    apply_lora_manual(model, lora_layers, rank=8)
    
    # Note: In a real scenario, we would load trained weights here:
    # trained_params = mx.load("checkpoints/lora_adapters.npz")
    # model.update(trained_params)
    
    print("Note: Using freshly initialized LoRA (no trained weights loaded)")
    print("In production, load trained adapters with: mx.load('checkpoints/lora_adapters.npz')")
    
    return model.model, tokenizer

def direct_qa(question, model, tokenizer, max_tokens=128):
    """Direct question answering without context."""
    prompt = f"Question: {question}\nAnswer:"
    
    answer = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False
    )
    
    return answer

def main():
    parser = argparse.ArgumentParser(description="Evaluate tuned model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base model")
    parser.add_argument("--lora_layers", type=int, default=8, help="Number of LoRA layers")
    parser.add_argument("--questions", type=str, default="data/eval_questions.jsonl", help="Path to evaluation questions")
    parser.add_argument("--output", type=str, default="results/tuned_answers.jsonl", help="Output file")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max tokens to generate")
    args = parser.parse_args()
    
    # Load tuned model
    model, tokenizer = load_tuned_model(args.model_path, args.lora_layers)
    
    print(f"Loading questions from {args.questions}")
    questions = []
    with open(args.questions, 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line))
    
    print(f"Evaluating {len(questions)} questions with tuned model...")
    
    results = []
    for i, q_data in enumerate(questions, 1):
        question = q_data['question']
        print(f"\n[{i}/{len(questions)}] {question}")
        
        try:
            answer = direct_qa(
                question,
                model,
                tokenizer,
                max_tokens=args.max_tokens
            )
            
            result = {
                **q_data,
                "tuned_answer": answer,
                "status": "success"
            }
            print(f"Answer: {answer[:100]}...")
            
        except Exception as e:
            print(f"Error: {e}")
            result = {
                **q_data,
                "tuned_answer": "",
                "status": "error",
                "error": str(e)
            }
        
        results.append(result)
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"\nResults saved to {args.output}")
    print(f"Successfully answered {sum(1 for r in results if r['status'] == 'success')}/{len(results)} questions")

if __name__ == "__main__":
    main()
