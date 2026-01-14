import mlx.core as mx
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from openrlhf.models.modeling_clara_mlx import CLaRa, CLaRaConfig
from mlx_lm import load

def test_clara_architecture():
    model_path = "/Users/user/Hands-on/hcx-seed-omni-8b-knowledge-injection/models/HyperCLOVAX-SEED-Omni-8B-Text-4bit"
    
    print("Loading base model...")
    model, tokenizer = load(model_path)
    
    # Define memory tokens (arbitrary for test)
    # Assume <MEM> tokens are added. 
    # For test, we'll pick some arbitrary token IDs as memory tokens.
    mem_token_ids = [100, 101] # Dummy IDs
    
    print("Initializing CLaRa config...")
    # Adjust config for test dimensions
    doc_max_len = 40
    compr_rate = 4
    n_mem = doc_max_len // compr_rate # 10
    
    config = CLaRaConfig(
        base_model_path=model_path,
        mem_token_ids=mem_token_ids,
        doc_max_length=doc_max_len,
        compr_rate=compr_rate
    )
    
    print("Initializing CLaRa model...")
    clara = CLaRa(model, config)
    
    # Create dummy input
    # Batch size 2, Seq len = doc_max_len = 40
    B = 2
    L = doc_max_len
    input_ids = mx.random.randint(0, 1000, (B, L))
    
    # Inject memory tokens at the end (mimic add_memory_tokens_to_inputs)
    # The last n_mem tokens should be memory tokens
    import numpy as np
    input_ids_np = np.array(input_ids)
    # Set last 10 tokens to mem_token_id (e.g. 100)
    input_ids_np[:, -n_mem:] = 100
    input_ids = mx.array(input_ids_np)
    
    attention_mask = mx.ones((B, L))
    
    print("Running compress...")
    try:
        mem_embs, mse_loss = clara.compress(input_ids, attention_mask)
        print(f"Success! Output shapes: Embs {mem_embs.shape}, Loss {mse_loss.shape}")
        # Expect Embs [B, n_mem, D] -> [2, 10, 4096]
        print(f"Loss value: {mse_loss.item()}")
    except Exception as e:
        print(f"Compress failed: {e}")
        raise e

    print("Running compress_query...")
    try:
        query_embs = clara.compress_query(input_ids, attention_mask)
        print(f"Success! Query Embs shape: {query_embs.shape}")
    except Exception as e:
        print(f"Compress query failed: {e}")
        raise e

if __name__ == "__main__":
    test_clara_architecture()
