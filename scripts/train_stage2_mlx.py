import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear
from openrlhf.models.modeling_clara_mlx import CLaRa, CLaRaConfig
import json
import argparse
import os
import time

def load_stage2_data(data_path, tokenizer, max_len=512, enc_max_len=256, n_mem_tokens=4):
    """
    Load Stage 2 Instruction Tuning data.
    Format: {"question": "...", "docs": ["...", ...], "answer": "..."}
    """
    with open(data_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            docs = item['docs']
            q = item['question']
            a = item['answer']
            
            # 1. Prepare encoder inputs (multiple docs)
            # We compress each doc separately
            all_enc_input_ids = []
            all_enc_attn_mask = []
            enc_mem_ids = [tokenizer.encode(f"<mem_{i}>", add_special_tokens=False)[0] for i in range(n_mem_tokens)]
            
            # For efficiency in this dry run/script, we take up to 2 docs
            for doc_text in docs[:2]:
                enc_tokens = tokenizer.encode(doc_text, add_special_tokens=True)[:enc_max_len]
                input_ids = enc_tokens + enc_mem_ids
                all_enc_input_ids.append(input_ids)
                all_enc_attn_mask.append([1] * len(input_ids))
            
            # 2. Prepare decoder input
            # Join memory placeholders for all compressed docs
            mem_placeholders = []
            for d_idx in range(len(all_enc_input_ids)):
                mem_placeholders.append("".join([f"<mem_{i}>" for i in range(n_mem_tokens)]))
            
            all_mem_str = "\n".join(mem_placeholders)
            prompt = f"Background:\n{all_mem_str}\n\nQuestion:{q}\nAnswer:"
            full_text = prompt + a
            
            full_tokens = tokenizer.encode(full_text, add_special_tokens=True)
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
            prompt_len = len(prompt_tokens)
            
            if len(full_tokens) > max_len:
                continue
                
            labels = ([-100] * prompt_len) + full_tokens[prompt_len:]
            
            # Pad enc_input_ids to the same length
            max_enc_len = max(len(x) for x in all_enc_input_ids)
            for i in range(len(all_enc_input_ids)):
                pad_len = max_enc_len - len(all_enc_input_ids[i])
                all_enc_input_ids[i] = all_enc_input_ids[i] + [tokenizer.pad_token_id] * pad_len
                all_enc_attn_mask[i] = all_enc_attn_mask[i] + [0] * pad_len
            
            yield {
                "enc_input_ids": mx.array(all_enc_input_ids), # [N, L]
                "enc_attention_mask": mx.array(all_enc_attn_mask), # [N, L]
                "dec_input_ids": mx.array(full_tokens)[None, :],
                "labels": mx.array(labels)[None, :]
            }

def loss_fn(params, model, batch):
    model.update(params)
    
    # Batch is one sample, but enc_input_ids is [N_docs, L]
    # We need to compress these N_docs as a batch
    mem_embs_batch, mse_loss = model.compress(batch["enc_input_ids"], batch["enc_attention_mask"])
    
    # Join all memory embeddings into one sequence for the decoder
    # mem_embs_batch: [N_docs, K, D] -> [1, N_docs*K, D]
    B = batch["dec_input_ids"].shape[0] # Should be 1
    combined_mem_embs = mem_embs_batch.reshape(B, -1, mem_embs_batch.shape[-1])
    
    # Dec forward
    logits, qa_loss, _ = model.forward(
        enc_input_ids=None, # Already compressed
        dec_input_ids=batch["dec_input_ids"],
        labels=batch["labels"],
        # Custom logic to pass combined_mem_embs? 
        # Need to call _replace_embeddings and then decoder manual loop
    )
    
    # Wait, model.forward with enc_input_ids=None does standard SFT.
    # I need a way to pass the pre-calculated combined_mem_embs.
    # Let's call the internal methods.
    
    inputs_embeds = model._replace_embeddings(combined_mem_embs, batch["dec_input_ids"])
    
    # Decoder manual forward
    x = inputs_embeds
    mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1]).astype(x.dtype)
    for layer in model.model.model.layers:
        x = layer(x, mask)
    x = model.model.model.norm(x)
    logits = model.model.lm_head(x)
    
    qa_loss = model._compute_qa_loss(logits, batch["labels"])
    
    alpha = 0.1
    return qa_loss + alpha * mse_loss, (qa_loss, mse_loss)

def main():
    parser = argparse.ArgumentParser(description="CLaRa Stage 2 Training (MLX)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="example/instruction_tuning_data.jsonl")
    parser.add_argument("--stage1_weights", type=str, default="checkpoints/stage1_dry_run.npz")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lora_layers", type=int, default=8)
    parser.add_argument("--save_path", type=str, default="checkpoints/stage2_adapter.npz")
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    model_mlx, tokenizer = load(args.model_path)
    model_mlx.freeze()
    
    # Configure CLaRa
    n_mem_tokens = 4 
    mem_tokens = [f"<mem_{i}>" for i in range(n_mem_tokens)]
    mem_token_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in mem_tokens]
    config = CLaRaConfig(args.model_path, doc_max_length=256, compr_rate=64, mem_token_ids=mem_token_ids)
    
    # Apply LoRA (last 8 layers)
    hidden_size = model_mlx.args.hidden_size
    num_heads = model_mlx.args.num_attention_heads
    num_kv_heads = model_mlx.args.num_key_value_heads
    head_dim = model_mlx.args.head_dim or (hidden_size // num_heads)
    q_out_dim = num_heads * head_dim
    v_out_dim = num_kv_heads * head_dim
    
    for layer in model_mlx.model.layers[-args.lora_layers:]:
        # Q_proj
        orig_q = layer.self_attn.q_proj
        lora_q = LoRALinear(hidden_size, q_out_dim, r=16)
        lora_q.linear = orig_q
        layer.self_attn.q_proj = lora_q
        
        # V_proj
        orig_v = layer.self_attn.v_proj
        lora_v = LoRALinear(hidden_size, v_out_dim, r=16)
        lora_v.linear = orig_v
        layer.self_attn.v_proj = lora_v

    model = CLaRa(model_mlx, config)

    # Debug: check parameters
    print("Checking parameters...")
    try:
        params = model.trainable_parameters()
        print(f"Found {len(params)} parameter groups.")
    except Exception as e:
        print(f"Failed to get parameters before loading: {e}")

    # Load Stage 1 weights if they exist
    if os.path.exists(args.stage1_weights):
        print(f"Loading Stage 1 weights from {args.stage1_weights}")
        model.load_weights(args.stage1_weights, strict=False)
        
    print("Checking parameters after loading...")
    try:
        params = model.trainable_parameters()
        print(f"Found {len(params)} parameter groups.")
    except Exception as e:
        print(f"Failed to get parameters after loading: {e}")
    
    optimizer = optim.Adam(learning_rate=args.lr)
    loss_and_grad_fn = mx.value_and_grad(loss_fn)

    def train_step(model, optimizer, batch):
        params = model.trainable_parameters()
        (loss, (qa_loss, mse_loss)), grads = loss_and_grad_fn(params, model, batch)
        # Apply gradients to get new parameters
        new_params = optimizer.apply_gradients(grads, params)
        # Update model with new parameters
        model.update(new_params)
        return loss, qa_loss, mse_loss

    print("Starting Stage 2 training...")
    data_gen = load_stage2_data(args.data_path, tokenizer, n_mem_tokens=n_mem_tokens)
    
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
    from mlx.utils import tree_flatten
    params = tree_flatten(model.trainable_parameters())
    mx.savez(args.save_path, **{k: v for k, v in params})
    print(f"Stage 2 complete. Weights saved to {args.save_path}")

if __name__ == "__main__":
    main()
