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

def load_stage3_data(data_path, tokenizer, max_len=512, enc_max_len=256, n_mem_tokens=4):
    with open(data_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            docs = item['docs']
            q = item['question']
            a = item['answer']
            pos_idx = item['pos_index'] # [idx1, idx2]
            
            # Encoder inputs (docs)
            all_enc_input_ids = []
            all_enc_attn_mask = []
            enc_mem_ids = [tokenizer.encode(f"<mem_{i}>", add_special_tokens=False)[0] for i in range(n_mem_tokens)]
            
            for doc_text in docs[:10]: # Limit to 10 docs for memory
                tokens = tokenizer.encode(doc_text, add_special_tokens=True)[:enc_max_len]
                ids = tokens + enc_mem_ids
                all_enc_input_ids.append(ids)
                all_enc_attn_mask.append([1] * len(ids))
            
            # Query input (for retrieval)
            query_tokens = tokenizer.encode(q, add_special_tokens=True)[:enc_max_len]
            query_ids = query_tokens + enc_mem_ids
            
            # Decoder input
            # Placeholder for 1 selected doc? Or all? In Stage 3 it selects.
            # For simplicity in this logic, we use the first positive index if available
            target_idx = pos_idx[0] if pos_idx else 0
            if target_idx >= 10: target_idx = 0
            
            mem_placeholders = "".join([f"<mem_{i}>" for i in range(n_mem_tokens)])
            prompt = f"Background:\n{mem_placeholders}\n\nQuestion:{q}\nAnswer:"
            full_text = prompt + a
            full_tokens = tokenizer.encode(full_text, add_special_tokens=True)
            prompt_len = len(tokenizer.encode(prompt, add_special_tokens=True))
            
            labels = ([-100] * prompt_len) + full_tokens[prompt_len:]
            
            # Pad docs
            max_enc_len = max(len(x) for x in all_enc_input_ids)
            for i in range(len(all_enc_input_ids)):
                pad = max_enc_len - len(all_enc_input_ids[i])
                all_enc_input_ids[i] = all_enc_input_ids[i] + [tokenizer.pad_token_id] * pad
                all_enc_attn_mask[i] = all_enc_attn_mask[i] + [0] * pad

            yield {
                "enc_input_ids": mx.array(all_enc_input_ids),
                "enc_attention_mask": mx.array(all_enc_attn_mask),
                "query_input_ids": mx.array(query_ids)[None, :],
                "query_attention_mask": mx.array([1]*len(query_ids))[None, :],
                "dec_input_ids": mx.array(full_tokens)[None, :],
                "labels": mx.array(labels)[None, :],
                "pos_indices": mx.array([target_idx])
            }

def loss_fn(params, model, batch):
    model.update(params)
    
    # 1. Compress all docs
    # batch["enc_input_ids"] is [N_docs, L]
    doc_mem_embs, doc_mse_loss = model.compress(batch["enc_input_ids"], batch["enc_attention_mask"])
    # doc_mem_embs: [N_docs, K, D]
    
    # Pool doc embs for retrieval [N_docs, D]
    doc_reps = doc_mem_embs.mean(axis=1) 
    
    # 2. Compress query
    query_mem_embs = model.compress_query(batch["query_input_ids"], batch["query_attention_mask"])
    # query_mem_embs is [B, K*D], let's reshape to [B, K, D] and mean or keep flat
    query_reps = query_mem_embs.reshape(1, -1, doc_reps.shape[-1]).mean(axis=1) # [1, D]
    
    # 3. Retrieval Loss
    retr_loss = model.compute_retrieval_loss(query_reps, doc_reps[None, ...], batch["pos_indices"])
    
    # 4. Selection (Differentiable or Top-1)
    # For dry run, we use the ground truth doc or top-1
    # selected_mem_embs = doc_mem_embs[batch["pos_indices"]]
    # In MLX, slicing with array might be restricted, use loop or where if B=1
    selected_idx = batch["pos_indices"][0].item()
    selected_mem_embs = doc_mem_embs[selected_idx : selected_idx+1] # [1, K, D]
    
    # 5. QA Loss
    inputs_embeds = model._replace_embeddings(selected_mem_embs, batch["dec_input_ids"])
    x = inputs_embeds
    mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1]).astype(x.dtype)
    for layer in model.model.model.layers:
        x = layer(x, mask)
    x = model.model.model.norm(x)
    logits = model.model.lm_head(x)
    qa_loss = model._compute_qa_loss(logits, batch["labels"])
    
    # Total
    total_loss = qa_loss + 0.1 * doc_mse_loss + 0.5 * retr_loss
    return total_loss, (qa_loss, doc_mse_loss, retr_loss)

def main():
    parser = argparse.ArgumentParser(description="CLaRa Stage 3 Training (MLX)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="example/end_to_end_data.jsonl")
    parser.add_argument("--stage2_weights", type=str, default="checkpoints/stage2_dry_run.npz")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--save_path", type=str, default="checkpoints/stage3_adapter.npz")
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    model_mlx, tokenizer = load(args.model_path)
    model_mlx.freeze()
    
    # LoRA (last 4 layers for dry run)
    hidden_size = model_mlx.args.hidden_size
    num_heads = model_mlx.args.num_attention_heads
    q_out_dim = num_heads * model_mlx.args.head_dim
    
    for layer in model_mlx.model.layers[-4:]:
        orig_q = layer.self_attn.q_proj
        lora_q = LoRALinear(hidden_size, q_out_dim, r=16)
        lora_q.linear = orig_q
        layer.self_attn.q_proj = lora_q
        
        orig_v = layer.self_attn.v_proj
        lora_v = LoRALinear(hidden_size, num_heads * model_mlx.args.head_dim, r=16)
        lora_v.linear = orig_v
        layer.self_attn.v_proj = lora_v

    n_mem_tokens = 4
    mem_token_ids = [tokenizer.encode(f"<mem_{i}>", add_special_tokens=False)[0] for i in range(n_mem_tokens)]
    config = CLaRaConfig(args.model_path, doc_max_length=256, compr_rate=64, mem_token_ids=mem_token_ids)
    model = CLaRa(model_mlx, config)

    if os.path.exists(args.stage2_weights):
        print(f"Loading Stage 2 weights from {args.stage2_weights}")
        model.load_weights(args.stage2_weights, strict=False)

    optimizer = optim.Adam(learning_rate=args.lr)
    loss_and_grad_fn = mx.value_and_grad(loss_fn)

    def train_step(model, optimizer, batch):
        params = model.trainable_parameters()
        (loss, (qa, mse, retr)), grads = loss_and_grad_fn(params, model, batch)
        new_params = optimizer.apply_gradients(grads, params)
        model.update(new_params)
        return loss, qa, mse, retr

    print("Starting Stage 3 training...")
    data_gen = load_stage3_data(args.data_path, tokenizer, n_mem_tokens=n_mem_tokens)
    
    for i in range(args.steps):
        try:
            batch = next(data_gen)
        except StopIteration:
            break
            
        loss, qa, mse, retr = train_step(model, optimizer, batch)
        mx.eval(loss, qa, mse, retr)
        print(f"Step {i}: Loss {loss.item():.4f} (QA: {qa.item():.4f}, MSE: {mse.item():.4f}, Retr: {retr.item():.4f})")

    # Save
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    from mlx.utils import tree_flatten
    params = tree_flatten(model.trainable_parameters())
    mx.savez(args.save_path, **{k: v for k, v in params})
    print(f"Stage 3 complete. Weights saved to {args.save_path}")

if __name__ == "__main__":
    main()
