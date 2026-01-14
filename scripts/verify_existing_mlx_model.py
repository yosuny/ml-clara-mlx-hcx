import mlx.core as mx
from mlx_lm import load, generate
import sys

# Path provided by user
model_path = "/Users/user/Hands-on/hcx-seed-omni-8b-knowledge-injection/models/HyperCLOVAX-SEED-Omni-8B-Text-4bit"

print(f"Checking model at: {model_path}")

try:
    # Attempt load
    print("Loading model with mlx_lm...")
    model, tokenizer = load(model_path)
    print("Model loaded successfully.")
    
    # Attempt basic generation
    print("Testing generation...")
    response = generate(model, tokenizer, prompt="Hello", max_tokens=10, verbose=True)
    print("\nGeneration Success.")
    
except Exception as e:
    print(f"\nFailed: {e}")
    sys.exit(1)
