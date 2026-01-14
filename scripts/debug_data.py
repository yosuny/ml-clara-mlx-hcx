import json
from mlx_lm import load

# Load tokenizer
model_path = "/Users/user/Hands-on/hcx-seed-omni-8b-knowledge-injection/models/HyperCLOVAX-SEED-Omni-8B-Text-4bit"
_, tokenizer = load(model_path)

# Load one sample
with open("example/end_to_end_data.jsonl", 'r') as f:
    sample = json.loads(f.readline())

print("Sample:", json.dumps(sample, indent=2, ensure_ascii=False)[:500])
print("\n---\n")

# Process it
question = sample.get("question", "")
docs = sample.get("docs", [])
answer = sample.get("gold_answer", "") or sample.get("answer", "")

doc_text = " ".join(docs) if isinstance(docs, list) else docs

input_text = f"Question: {question}\nContext: {doc_text}\nAnswer:"
target_text = f" {answer}"

print(f"Input text: {input_text[:200]}")
print(f"Target text: {target_text[:100]}")
print("\n---\n")

# Tokenize
input_ids = tokenizer.encode(input_text, add_special_tokens=False)
target_ids = tokenizer.encode(target_text, add_special_tokens=False)

print(f"Input IDs: {len(input_ids)} tokens")
print(f"Target IDs: {len(target_ids)} tokens")
print(f"Target IDs sample: {target_ids[:20]}")

# Labels
labels = ([-100] * len(input_ids)) + target_ids
print(f"Labels: {len(labels)} total, {sum(1 for x in labels if x != -100)} valid")
print(f"Valid labels sample: {[x for x in labels if x != -100][:20]}")
