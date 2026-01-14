import mlx.core as mx
from mlx_lm import load, generate
from openrlhf.models.modeling_clara_mlx import CLaRa, CLaRaConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import argparse
import os

def load_clara_model(model_path, weights_path):
    model_mlx, tokenizer = load(model_path)
    
    # Configure CLaRa
    n_mem_tokens = 4
    mem_tokens = [f"<mem_{i}>" for i in range(n_mem_tokens)]
    mem_token_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in mem_tokens]
    config = CLaRaConfig(model_path, doc_max_length=256, compr_rate=64, mem_token_ids=mem_token_ids)
    
    model = CLaRa(model_mlx, config)
    
    if os.path.exists(weights_path):
        print(f"Loading Stage 3 weights from {weights_path}")
        model.load_weights(weights_path, strict=False)
    
    return model, tokenizer

def run_evaluation(model, tokenizer, eval_data, knowledge_docs):
    """
    Knowledge Retrieval -> Compression -> Generation
    """
    # Simple TF-IDF for retrieval in this evaluation script
    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(knowledge_docs)
    
    results = []
    for item in eval_data:
        q = item['question']
        print(f"\nQuestion: {q}")
        
        # 1. Retrieve
        query_vec = vectorizer.transform([q])
        scores = cosine_similarity(query_vec, doc_vectors).flatten()
        top_k_indices = scores.argsort()[-5:][::-1]
        retrieved_docs = [knowledge_docs[i] for i in top_k_indices]
        
        # 2. Compress retrieved docs (Batch)
        enc_input_ids = []
        enc_attn_mask = []
        n_mem_tokens = 4
        enc_mem_ids = [tokenizer.encode(f"<mem_{i}>", add_special_tokens=False)[0] for i in range(n_mem_tokens)]
        
        for doc in retrieved_docs:
            tokens = tokenizer.encode(doc, add_special_tokens=True)[:256]
            ids = tokens + enc_mem_ids
            enc_input_ids.append(ids)
            enc_attn_mask.append([1] * len(ids))
            
        # Pad enc_input_ids
        max_len = max(len(x) for x in enc_input_ids)
        for i in range(len(enc_input_ids)):
            pad = max_len - len(enc_input_ids[i])
            enc_input_ids[i] = enc_input_ids[i] + [tokenizer.pad_token_id] * pad
            enc_attn_mask[i] = enc_attn_mask[i] + [0] * pad
            
        # Compress
        mem_embs, _ = model.compress(mx.array(enc_input_ids), mx.array(enc_attn_mask))
        
        # 3. Join all memory embeddings [1, N*K, D]
        combined_mem_embs = mem_embs.reshape(1, -1, mem_embs.shape[-1])
        
        # 4. Generate with combined memory
        mem_placeholders = ""
        for _ in range(len(retrieved_docs)):
             mem_placeholders += "".join([f"<mem_{i}>" for i in range(n_mem_tokens)]) + "\n"
        
        prompt = f"Background:\n{mem_placeholders}\n\nQuestion: {q}\nAnswer:"
        
        # Since 'generate' in mlx_lm doesn't support inputs_embeds, we use the manual loop approach
        # or we just manually call the model's forward/generate logic.
        # But for 'generate' convenience, we usually need the full wrapper.
        
        # Let's implement a simple greedy decoding here for CLaRa
        tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=True))[None, :]
        input_embeds = model._replace_embeddings(combined_mem_embs, tokens)
        
        # Manual greedy generation
        out_tokens = []
        x = input_embeds
        # Need to handle mask correctly in generation
        # This is a bit complex for a script, let's see if we can use model.__call__
        
        # Alternative: We use the prompt markers and hope the model learned to use the memory.
        # But the memory tokens are replaced BY the embeddings.
        
        # Let's do it properly:
        L = x.shape[1]
        mask = mx.triu(mx.full((L, L), -float("inf"), dtype=x.dtype), k=1)
        
        # We need the full KV cache for efficient generation, but for a short answer:
        current_len = L
        for _ in range(50): # Max 50 tokens
            # Forward
            h = x
            mask = mx.triu(mx.full((h.shape[1], h.shape[1]), -float("inf"), dtype=h.dtype), k=1)
            for layer in model.model.model.layers:
                h = layer(h, mask, cache=None)
            h = model.model.model.norm(h)
            logits = model.model.lm_head(h[:, -1, :])
            
            token = mx.argmax(logits, axis=-1)
            token_id = token.item()
            if token_id == tokenizer.eos_token_id:
                break
            out_tokens.append(token_id)
            
            # Append new token embed
            next_emb = model.model.model.embed_tokens(token)
            x = mx.concatenate([x, next_emb[None, :, :]], axis=1)
            
        generated_answer = tokenizer.decode(out_tokens)
        print(f"Generated: {generated_answer}")
        results.append({"question": q, "answer": generated_answer, "gold": item.get('gold_answer', "")})
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Final CLaRa MLX Evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--eval_file", type=str, default="data/eval_questions.jsonl")
    parser.add_argument("--knowledge_file", type=str, default="data/train_data.jsonl")
    args = parser.parse_args()

    # Load knowledge and chunk it
    knowledge_chunks = []
    with open(args.knowledge_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            text = item['text']
            # Chunking
            words = text.split()
            chunk_size = 200
            overlap = 50
            for i in range(0, len(words), chunk_size - overlap):
                chunk = ' '.join(words[i:i + chunk_size])
                if chunk.strip():
                    knowledge_chunks.append(chunk)
            
    # Load eval questions
    eval_data = []
    with open(args.eval_file, 'r') as f:
        for line in f:
            eval_data.append(json.loads(line))
            
    # Load model
    model, tokenizer = load_clara_model(args.model_path, args.weights_path)
    
    # Run
    results = run_evaluation(model, tokenizer, eval_data[:5], knowledge_chunks) 
    
    # Save
    with open("results/final_eval_clara.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
