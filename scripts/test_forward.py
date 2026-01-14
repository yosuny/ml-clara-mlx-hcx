import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from openrlhf.models.modeling_clara_mlx import CLaRa, CLaRaConfig

# Load model
model_path = "/Users/user/Hands-on/hcx-seed-omni-8b-knowledge-injection/models/HyperCLOVAX-SEED-Omni-8B-Text-4bit"
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

model = CLaRa(base_model, model_config)

# Create dummy data
input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=mx.int32)
attention_mask = mx.ones((1, 10), dtype=mx.int32)
labels = mx.array([[-100, -100, -100, -100, -100, -100, -100, 8, 9, 10]], dtype=mx.int32)

print("Input shape:", input_ids.shape)
print("Labels:", labels)
print("Valid labels:", (labels != -100).sum().item())

# Test forward pass
print("\n=== First forward pass ===")
logits1, qa_loss1, mse_loss1 = model.forward(input_ids, attention_mask, labels)
print(f"Logits shape: {logits1.shape}")
print(f"QA loss: {qa_loss1.item()}")
print(f"MSE loss: {mse_loss1.item()}")

# Test second forward pass (should give same results)
print("\n=== Second forward pass ===")
logits2, qa_loss2, mse_loss2 = model.forward(input_ids, attention_mask, labels)
print(f"QA loss: {qa_loss2.item()}")
print(f"MSE loss: {mse_loss2.item()}")

# Test with model.update
print("\n=== After model.update ===")
params = model.trainable_parameters()
model.update(params)
logits3, qa_loss3, mse_loss3 = model.forward(input_ids, attention_mask, labels)
print(f"QA loss: {qa_loss3.item()}")
print(f"MSE loss: {mse_loss3.item()}")
