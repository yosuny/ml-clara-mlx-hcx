import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load, generate
from mlx_lm.tuner.lora import LoRALinear
from openrlhf.models.modeling_clara_mlx import CLaRa, CLaRaConfig
import json
import argparse
import os
import time

def load_stage1_data(data_path, tokenizer, max_len=512, enc_max_len=256, n_mem_tokens=4):
    """
    Load Stage 1 pretrain data (jsonl).
    Format: {"data_type": "qa", "question": [...], "answers": [...], "docs": [...]}
    """
    mem_tokens_str = "".join([f"<mem_{i}>" for i in range(n_mem_tokens)])
    
    with open(data_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            docs = item['docs']
            questions = item['question']
            answers = item['answers']
            
            # For each question/answer pair, we use the documents as context
            for q, a in zip(questions, answers):
                # 1. Prepare encoder input (documents)
                # Original doc
                doc_text = " ".join(docs) if isinstance(docs, list) else docs
                enc_tokens = tokenizer.encode(doc_text, add_special_tokens=True)[:enc_max_len]
                # Pad with memory tokens at the end
                enc_mem_ids = [tokenizer.encode(f"<mem_{i}>", add_special_tokens=False)[0] for i in range(n_mem_tokens)]
                enc_input_ids = enc_tokens + enc_mem_ids
                enc_attn_mask = [1] * len(enc_input_ids)
                
                # 2. Prepare decoder input (prompt with memory placeholders + answer)
                # In Stage 1, we use a simple format: "Background: <mem_0>... Question: Q Answer: A"
                prompt = f"Background:\n{mem_tokens_str}\n\nQuestion:{q}\nAnswer:"
                full_text = prompt + a
                
                # Tokenize
                full_tokens = tokenizer.encode(full_text, add_special_tokens=True)
                prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
                prompt_len = len(prompt_tokens)
                
                if len(full_tokens) > max_len:
                    continue # Simple skip for now
                
                # Labels: -100 for prompt, token_id for answer
                labels = ([-100] * prompt_len) + full_tokens[prompt_len:]
                
                yield {
                    "enc_input_ids": mx.array(enc_input_ids)[None, :],
                    "enc_attention_mask": mx.array(enc_attn_mask)[None, :],
                    "dec_input_ids": mx.array(full_tokens)[None, :],
                    "labels": mx.array(labels)[None, :]
                }

def loss_fn(model, batch):
    logits, qa_loss, mse_loss = model.forward(
        enc_input_ids=batch["enc_input_ids"],
        enc_attention_mask=batch["enc_attention_mask"],
        dec_input_ids=batch["dec_input_ids"],
        labels=batch["labels"]
    )
    # Total loss = QA Loss + alpha * MSE Loss
    # Original paper/repo uses alpha around 0.1 for MSE
    alpha = 0.1
    return qa_loss + alpha * mse_loss, (qa_loss, mse_loss)

def main():
    parser = argparse.ArgumentParser(description="CLaRa Stage 1 Training (MLX)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="example/pretrain_data.jsonl")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=1) # Memory limit on local
    parser.add_argument("--lora_layers", type=int, default=8)
    parser.add_argument("--save_path", type=str, default="checkpoints/stage1_compressor.npz")
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    model_mlx, tokenizer = load(args.model_path)
    
    # Freeze base model
    model_mlx.freeze()
    
    # Setup CLaRa Config
    # Based on 256/64
    n_mem_tokens = 4 
    mem_tokens = [f"<mem_{i}>" for i in range(n_mem_tokens)]
    mem_token_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in mem_tokens]
    
    config = CLaRaConfig(
        base_model_path=args.model_path,
        doc_max_length=256,
        compr_rate=64,
        mem_token_ids=mem_token_ids
    )
    
    # Apply LoRA
    print(f"Applying LoRA to last {args.lora_layers} layers...")
    hidden_size = model_mlx.args.hidden_size
    num_heads = model_mlx.args.num_attention_heads
    num_kv_heads = model_mlx.args.num_key_value_heads
    head_dim = model_mlx.args.head_dim or (hidden_size // num_heads)
    
    q_out_dim = num_heads * head_dim
    v_out_dim = num_kv_heads * head_dim
    
    for layer in model_mlx.model.layers[-args.lora_layers:]:
        # Q_proj
        original_q = layer.self_attn.q_proj
        lora_q = LoRALinear(hidden_size, q_out_dim, r=16)
        lora_q.linear = original_q
        layer.self_attn.q_proj = lora_q
        
        # V_proj
        original_v = layer.self_attn.v_proj
        lora_v = LoRALinear(hidden_size, v_out_dim, r=16)
        lora_v.linear = original_v
        layer.self_attn.v_proj = lora_v
    
    model = CLaRa(model_mlx, config)
    
    optimizer = optim.Adam(learning_rate=args.lr)

    def loss_fn(params, model, batch):
        model.update(params)
        logits, qa_loss, mse_loss = model.forward(
            enc_input_ids=batch["enc_input_ids"],
            enc_attention_mask=batch["enc_attention_mask"],
            dec_input_ids=batch["dec_input_ids"],
            labels=batch["labels"]
        )
        alpha = 0.1
        return qa_loss + alpha * mse_loss, (qa_loss, mse_loss)

    loss_and_grad_fn = mx.value_and_grad(loss_fn)

    def train_step(model, optimizer, batch):
        (loss, (qa_loss, mse_loss)), grads = loss_and_grad_fn(model.trainable_parameters(), model, batch)
        optimizer.update(model, grads)
        return loss, qa_loss, mse_loss

    print("Starting Stage 1 training...")
    data_gen = load_stage1_data(args.data_path, tokenizer, n_mem_tokens=n_mem_tokens)
    
    start_time = time.time()
    for i in range(args.steps):
        try:
            batch = next(data_gen)
        except StopIteration:
            break
            
        loss, qa_loss, mse_loss = train_step(model, optimizer, batch)
        mx.eval(model.parameters(), optimizer.state, loss, qa_loss, mse_loss)
        
        if i % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Step {i}: Loss {loss.item():.4f} (QA: {qa_loss.item():.4f}, MSE: {mse_loss.item():.4f}) | {elapsed:.1f}s")

    # Save weights
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # Filter only mx.array for saving (mx.savez doesn't like other types)
    from mlx.utils import tree_flatten
    params = tree_flatten(model.trainable_parameters())
    save_dict = {k: v for k, v in params}
    
    mx.savez(args.save_path, **save_dict)
    print(f"Training complete. Weights saved to {args.save_path}")

if __name__ == "__main__":
    main()
