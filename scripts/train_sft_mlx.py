import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear

import sys
import os
import argparse
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openrlhf.models.modeling_clara_mlx import CLaRa, CLaRaConfig

def load_sft_data(tokenizer, data_path, max_length=512):
    """
    Load SFT data in the format: {"question": str, "docs": [str], "gold_answer": str}
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file {data_path} not found")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    samples = []
    for line in lines:
        try:
            item = json.loads(line)
            samples.append(item)
        except:
            continue
    
    print(f"Loaded {len(samples)} samples from {data_path}")
    
    # Process each sample
    for sample in samples:
        # Build input: question + docs
        question = sample.get("question", "")
        docs = sample.get("docs", [])
        answer = sample.get("gold_answer", "") or sample.get("answer", "")
        
        # C concatenate docs
        doc_text = " ".join(docs) if isinstance(docs, list) else docs
        
        # Format: Question: {q} Context: {docs} Answer: {a}
        input_text = f"Question: {question}\nContext: {doc_text}\nAnswer:"
        target_text = f" {answer}"
        
        # Tokenize separately
        input_ids = tokenizer.encode(input_text, add_special_tokens=False)
        target_ids = tokenizer.encode(target_text, add_special_tokens=False)
        
        # Combine input + target
        full_ids = input_ids + target_ids
        
        # Create labels: -100 for input portion, actual ids for target
        # Labels must match full_ids length BEFORE truncation
        labels = ([-100] * len(input_ids)) + target_ids
        
        # Truncate if needed - IMPORTANT: truncate from INPUT side to keep target
        if len(full_ids) > max_length:
            # Keep the target tokens, truncate the input
            target_len = len(target_ids)
            max_input_len = max_length - target_len
            
            if max_input_len > 0:
                # Truncate input, keep all target
                input_ids_trunc = input_ids[-max_input_len:]  # Keep last part of input
                full_ids = input_ids_trunc + target_ids
                labels = ([-100] * len(input_ids_trunc)) + target_ids
            else:
                # Target itself is too long, just truncate everything
                full_ids = full_ids[:max_length]
                labels = labels[:max_length]
        
        # Create attention mask
        seq_len = len(full_ids)
        attention_mask = [1] * seq_len
        
        # Pad to max_length
        pad_length = max_length - seq_len
        
        if pad_length > 0:
            full_ids = full_ids + ([tokenizer.pad_token_id or 0] * pad_length)
            attention_mask = attention_mask + ([0] * pad_length)
            labels = labels + ([-100] * pad_length)
        
        # Convert to MLX arrays with explicit dtypes
        yield {
            "input_ids": mx.array(full_ids, dtype=mx.int32),
            "attention_mask": mx.array(attention_mask, dtype=mx.int32),
            "labels": mx.array(labels, dtype=mx.int32)
        }

def apply_lora_manual(clara_model, num_layers, rank=8, dropout=0.05):
    """Apply LoRA to specific layers."""
    layers = clara_model.model.layers
    args = clara_model.model.args
    
    total_layers = len(layers)
    num_layers = min(num_layers, total_layers)
    
    target_layers = layers[-num_layers:]
    print(f"Applying LoRA manually to last {num_layers} layers (total {total_layers})...")
    
    # Calculate dims
    hidden_size = args.hidden_size
    head_dim = args.head_dim or (hidden_size // args.num_attention_heads)
    q_out_dim = args.num_attention_heads * head_dim
    v_out_dim = args.num_key_value_heads * head_dim
    
    print(f"Dims: Hidden={hidden_size}, Q_Out={q_out_dim}, V_Out={v_out_dim}")
    
    count = 0
    for layer in target_layers:
        if hasattr(layer.self_attn, "q_proj"):
            original = layer.self_attn.q_proj
            if "LoRALinear" not in str(type(original)):
                try:
                    lora_layer = LoRALinear(hidden_size, q_out_dim, r=rank, dropout=dropout)
                    lora_layer.linear = original
                    layer.self_attn.q_proj = lora_layer
                    count += 1
                except Exception as e:
                    print(f"Failed to apply LoRA to q_proj: {e}")
        
        if hasattr(layer.self_attn, "v_proj"):
            original = layer.self_attn.v_proj
            if "LoRALinear" not in str(type(original)):
                try:
                    lora_layer = LoRALinear(hidden_size, v_out_dim, r=rank, dropout=dropout)
                    lora_layer.linear = original
                    layer.self_attn.v_proj = lora_layer
                    count += 1
                except Exception as e:
                    print(f"Failed to apply LoRA to v_proj: {e}")
                
    print(f"Replaced {count} modules with LoRALinear.")

def main():
    parser = argparse.ArgumentParser(description="SFT Training for CLaRa on MLX")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base model")
    parser.add_argument("--data_file", type=str, default="example/end_to_end_data.jsonl", help="Path to SFT data")
    parser.add_argument("--lora_layers", type=int, default=4, help="Number of LoRA layers to adapt")
    parser.add_argument("--steps", type=int, default=10, help="Training steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--mse_weight", type=float, default=0.1, help="Weight for MSE loss")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}")
    base_model, tokenizer = load(args.model_path)
    
    # Freeze base model
    base_model.freeze()
    
    # Setup CLaRa Config
    eos_id = getattr(tokenizer, "eos_token_id", 2)
    if eos_id is None: eos_id = 2
    mem_token_ids = [eos_id]
    
    model_config = CLaRaConfig(
        base_model_path=args.model_path,
        doc_max_length=128,
        compr_rate=4,
        mem_token_ids=mem_token_ids
    )
    
    print("Initializing CLaRa architecture...")
    model = CLaRa(base_model, model_config)
    
    # Apply LoRA
    apply_lora_manual(model, args.lora_layers, rank=8)
    
    # Print parameter counts
    parameters = tree_flatten(model.parameters())
    trainable = tree_flatten(model.trainable_parameters())
    total_params = sum(x[1].size for x in parameters)
    trainable_params = sum(x[1].size for x in trainable)
    print(f"Total: {total_params/1e6:.2f}M, Trainable: {trainable_params/1e6:.2f}M")
    
    # Optimizer
    optimizer = optim.AdamW(learning_rate=args.learning_rate)
    
    # Training loop
    def loss_fn(params, input_ids, attention_mask, labels):
        """Loss function for gradient computation."""
        model.update(params)
        logits, qa_loss, mse_loss = model.forward(input_ids, attention_mask, labels)
        total_loss = qa_loss + args.mse_weight * mse_loss
        return total_loss
    
    loss_and_grad_fn = mx.value_and_grad(loss_fn)
    
    # Cache for component losses
    cached_qa_loss = None
    cached_mse_loss = None
    
    def train_step(params, optimizer, input_ids, attention_mask, labels):
        nonlocal cached_qa_loss, cached_mse_loss
        
        # Compute loss and gradients
        loss, grads = loss_and_grad_fn(params, input_ids, attention_mask, labels)
        optimizer.update(model, grads)
        
        # Compute component losses separately for logging (after optimizer update)
        # This ensures model state is consistent
        _, qa_loss, mse_loss = model.forward(input_ids, attention_mask, labels)
        
        return loss, qa_loss, mse_loss
    
    print("Starting SFT training loop...")
    
    # Load data
    data_gen = load_sft_data(tokenizer, args.data_file, max_length=512)
    
    step = 0
    for batch in data_gen:
        if step >= args.steps:
            break
        
        try:
            input_ids = batch["input_ids"].reshape(1, -1)
            attention_mask = batch["attention_mask"].reshape(1, -1)
            labels = batch["labels"].reshape(1, -1)
            
            # Train step
            loss, qa_loss, mse_loss = train_step(
                model.trainable_parameters(),
                optimizer,
                input_ids,
                attention_mask,
                labels
            )
            
            mx.eval(model.parameters(), optimizer.state)
            
            print(f"Step {step+1}: Loss={loss.item():.4f}, QA={qa_loss.item():.4f}, MSE={mse_loss.item():.4f}")
            step += 1
            
        except Exception as e:
            print(f"Training step failed: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("Training finished successfully.")

if __name__ == "__main__":
    main()
