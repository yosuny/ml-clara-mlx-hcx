import argparse
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear

import sys
import os
sys.path.append(os.getcwd())

from openrlhf.models.modeling_clara_mlx import CLaRa, CLaRaConfig

import json

def load_data_generator(tokenizer, data_path, seq_len=2048):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file {data_path} not found")
        
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    full_text = ""
    for line in lines:
        try:
            doc = json.loads(line)
            full_text += doc["text"] + "\n"
        except:
            continue
            
    # Tokenize
    # mlx_lm tokenizer is usually huggingface tokenizer
    tokens = tokenizer.encode(full_text)
    total_tokens = len(tokens)
    print(f"Loaded {total_tokens} tokens from {data_path}")
    
    # Yield batches
    # Simple sliding window
    bs = 2
    for i in range(0, total_tokens - seq_len * bs, seq_len * bs):
        batch_tokens = []
        for b in range(bs):
            start = i + b * seq_len
            end = start + seq_len
            chunk = tokens[start:end]
            if len(chunk) < seq_len:
                # Pad? Or just skip last
                continue
            batch_tokens.append(chunk)
            
        if len(batch_tokens) == bs:
            yield mx.array(batch_tokens)
            
    # Infinite loop for training steps if needed?
    # For now, just one pass or restart
    while True:
        # Loop again
        for i in range(0, total_tokens - seq_len * bs, seq_len * bs):
             batch_tokens = []
             for b in range(bs):
                start = i + b * seq_len
                end = start + seq_len
                chunk = tokens[start:end]
                if len(chunk) < seq_len: continue
                batch_tokens.append(chunk)
             if len(batch_tokens) == bs:
                yield mx.array(batch_tokens)

def loss_fn(model, input_ids, attention_mask):
    # Forward pass through CLaRa compression
    # The first return value is memory embeddings, second is mse_loss
    _, mse_loss = model.compress(input_ids, attention_mask)
    return mse_loss

def apply_lora_manual(clara_model, num_layers, rank=8, dropout=0.05):
    # Access internal layers
    layers = clara_model.model.layers
    # Access model args from mlx_lm model
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
        # Target self_attn.q_proj and v_proj
        if hasattr(layer.self_attn, "q_proj"):
            original = layer.self_attn.q_proj
            if "LoRALinear" not in str(type(original)):
                try:
                    # Input is hidden_size, Output is q_out_dim
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
    parser = argparse.ArgumentParser(description="Train CLaRa on MLX")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base model")
    parser.add_argument("--data_file", type=str, default="data/train_data.jsonl", help="Path to training data")
    parser.add_argument("--lora_layers", type=int, default=16, help="Number of LoRA layers to adapt")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}")
    base_model, tokenizer = load(args.model_path)
    
    # Freeze base model
    base_model.freeze()
    
    # Setup CLaRa Config
    # Use EOS token as dummy memory token if not set
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
    
    print("Model trainable parameters:")
    parameters = tree_flatten(model.parameters())
    trainable = tree_flatten(model.trainable_parameters())
    total_params = sum(x[1].size for x in parameters)
    trainable_params_count = sum(x[1].size for x in trainable)
    print(f"Total: {total_params/1e6:.2f}M, Trainable: {trainable_params_count/1e6:.2f}M")
    
    print("Checking LoRA application:")
    count_lora = 0
    for name, m in model.named_modules():
        if "LoRALinear" in str(type(m)):
            count_lora += 1
    print(f"Found {count_lora} LoRALinear modules.")

    # Optimizer
    optimizer = optim.AdamW(learning_rate=1e-4)
    
    # Define functional loss
    def loss_fn(params, input_ids, attention_mask):
        model.update(params)
        _, mse_loss = model.compress(input_ids, attention_mask)
        return mse_loss

    loss_and_grad_fn = mx.value_and_grad(loss_fn)
    
    # @mx.compile
    def compute_loss_and_grads(params, input_ids, attention_mask):
        return loss_and_grad_fn(params, input_ids, attention_mask)

    def train_step(params, optimizer, input_ids, attention_mask):
        loss, grads = compute_loss_and_grads(params, input_ids, attention_mask)
        optimizer.update(model, grads)
        return loss
    
    print("Starting training loop...")
        
    data_gen = load_data_generator(tokenizer, args.data_file, seq_len=128)
    
    for i in range(args.steps):
        try:
            input_ids = next(data_gen)
            # Inject memory tokens
            n_mem = model_config.doc_max_length // model_config.compr_rate
            import numpy as np
            input_ids_np = np.array(input_ids)
            input_ids_np[:, -n_mem:] = mem_token_ids[0]
            input_ids = mx.array(input_ids_np)
            
            attention_mask = mx.ones(input_ids.shape)
            
            # Pass trainable parameters
            loss = train_step(model.trainable_parameters(), optimizer, input_ids, attention_mask)
            mx.eval(model.parameters(), optimizer.state)
            
            print(f"Step {i+1}: Loss = {loss.item():.4f}")
        except Exception as e:
            print(f"Training step failed: {e}")
            import traceback
            traceback.print_exc()
            break
        
    print("Training finished successfully.")

if __name__ == "__main__":
    main()
