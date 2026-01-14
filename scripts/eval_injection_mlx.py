import mlx.core as mx
from mlx_lm import load, generate
import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Evaluate CPT+SFT model internal knowledge")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base model")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to SFT adapter directory")
    parser.add_argument("--questions", type=str, default="data/eval_questions.jsonl", help="Path to evaluation questions")
    parser.add_argument("--output", type=str, default="results/injection_answers.jsonl", help="Output file")
    args = parser.parse_args()
    
    print(f"Loading model with adapter from {args.adapter_path}")
    # mlx_lm.load takes the model path and adapter_path (directory containing adapters.safetensors)
    model, tokenizer = load(args.model_path, adapter_path=args.adapter_path)
    
    print(f"Loading questions from {args.questions}")
    questions = []
    with open(args.questions, 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line))
            
    results = []
    for i, q_data in enumerate(questions, 1):
        question = q_data['question']
        print(f"\n[{i}/{len(questions)}] {question}")
        
        # Simple prompt for internal knowledge
        prompt = f"Question: {question}\nAnswer:"
        
        try:
            answer = generate(
                model, 
                tokenizer, 
                prompt=prompt,
                max_tokens=256,
                verbose=False
            )
            
            result = {
                **q_data,
                "injection_answer": answer,
                "status": "success"
            }
            print(f"Answer: {answer[:100]}...")
        except Exception as e:
            print(f"Error: {e}")
            result = {
                **q_data,
                "injection_answer": "",
                "status": "error",
                "error": str(e)
            }
        results.append(result)
        
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
