import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.llama import Model as LlamaModel
from mlx_lm.models.llama import ModelArgs
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union

@dataclass
class CLaRaConfig:
    """Configuration for CLaRa MLX model."""
    base_model_path: str
    padding_side: str = "left"
    doc_max_length: int = 128
    compr_rate: int = 64
    mem_token_ids: List[int] = None
    
    @classmethod
    def from_dict(cls, params):
        # Filter params to match fields? Or just kwargs
        # For simple dataclass, we can use filtering
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in params.items() if k in valid_keys}
        return cls(**filtered)

class CLaRa(nn.Module):
    """
    CLaRa architecture ported to MLX.
    Wraps a LlamaModel and implements compression logic.
    """
    def __init__(self, model: LlamaModel, config: CLaRaConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.mem_token_ids = config.mem_token_ids if config.mem_token_ids else []

    def load_weights(self, path: str, strict: bool = True):
        return self.model.load_weights(path, strict=strict)

    def trainable_parameters(self):
        return self.model.trainable_parameters()
    
    def parameters(self):
        return self.model.parameters()

    def update(self, params):
        return self.model.update(params)

    def __call__(self, inputs: Union[mx.array, Tuple[mx.array, mx.array]], cache=None):
        if isinstance(inputs, tuple):
            # inputs_embeds case
            return self.model.model(inputs_embeds=inputs[0], cache=cache)
        return self.model(inputs, cache)

    def load_weights(self, path: str, strict: bool = True):
        return self.model.load_weights(path, strict=strict)
    
    def _replace_embeddings(self, compressed_embs: mx.array, dec_input_ids: mx.array) -> mx.array:
        """
        Replace memory tokens in decoder input with compressed embeddings.
        compressed_embs: [B, K, D]
        dec_input_ids: [B, L]
        """
        # 1. Get initial embeddings
        inputs_embeds = self.model.model.embed_tokens(dec_input_ids) # [B, L, D]
        
        # 2. Find memory tokens
        mem_ids_arr = mx.array(self.mem_token_ids)
        mask = mx.any(dec_input_ids[..., None] == mem_ids_arr, axis=-1) # [B, L]
        
        # 3. Find start indices
        indices = mx.argmax(mask, axis=1) # [B]
        
        B, L, D = inputs_embeds.shape
        K_per_doc = compressed_embs.shape[1] // (compressed_embs.shape[1] // (self.config.doc_max_length // self.config.compr_rate)) # This is complex if not simple
        # Simpler: K is total memory tokens to replace
        K = compressed_embs.shape[1]
        
        results = []
        for i in range(B):
            start = indices[i].item()
            prefix = inputs_embeds[i, :start]
            mid = compressed_embs[i]
            # Assumes K tokens were present in original dec_input_ids
            suffix = inputs_embeds[i, start+K:]
            
            # Ensure dimensions match
            if prefix.shape[0] + mid.shape[0] + suffix.shape[0] != L:
                # Fallback or clipping if sizes mismatched due to truncation
                combined = mx.concatenate([prefix, mid, suffix], axis=0)
                if combined.shape[0] > L:
                    combined = combined[:L]
                else:
                    # Pad if too short? Should not happen if placeholders are correct
                    pass
                results.append(combined)
            else:
                results.append(mx.concatenate([prefix, mid, suffix], axis=0))
        
        return mx.stack(results)

    def _compute_qa_loss(self, logits: mx.array, labels: mx.array) -> mx.array:
        """Helper to compute masked cross entropy loss."""
        # Shift for causal LM: predict next token
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        
        # Mask out ignore index (-100)
        valid_mask = shift_labels != -100
        
        if valid_mask.sum() > 0:
            vocab_size = shift_logits.shape[-1]
            flat_logits = shift_logits.reshape(-1, vocab_size)
            flat_labels = shift_labels.reshape(-1)
            flat_mask = valid_mask.reshape(-1)
            
            safe_labels = mx.where(flat_mask, flat_labels, 0)
            all_losses = nn.losses.cross_entropy(flat_logits, safe_labels, reduction='none')
            masked_losses = all_losses * flat_mask.astype(mx.float32)
            
            return masked_losses.sum() / flat_mask.sum()
        else:
            return mx.array(0.0)

    def forward(self, 
                input_ids: mx.array = None, 
                attention_mask: mx.array = None, 
                labels: mx.array = None,
                enc_input_ids: mx.array = None,
                enc_attention_mask: mx.array = None,
                dec_input_ids: mx.array = None,
                dec_attention_mask: mx.array = None) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Multi-stage forward pass.
        If enc_input_ids is provided, performs Stage 1/2 logic (compress then decode).
        Else, if dec_input_ids is provided, it assumes pre-compressed or simple decode.
        Else, performs standard SFT forward (unified).
        """
        if enc_input_ids is not None:
            # Stage 1/2 logic: Compress then Decode
            mem_embs, mse_loss = self.compress(enc_input_ids, enc_attention_mask)
            inputs_embeds = self._replace_embeddings(mem_embs, dec_input_ids)
            
            # Decoder forward pass
            x = inputs_embeds
            L = x.shape[1]
            mask = nn.MultiHeadAttention.create_additive_causal_mask(L).astype(x.dtype)
            for layer in self.model.model.layers:
                x = layer(x, mask, cache=None)
            x = self.model.model.norm(x)
            logits = self.model.lm_head(x)
            
            qa_loss = self._compute_qa_loss(logits, labels) if labels is not None else mx.array(0.0)
            return logits, qa_loss, mse_loss
        
        elif dec_input_ids is not None:
            # Simple decoder-only forward (e.g. for Stage 2 when embeddings injected manually outside or as prompt)
            hidden_states = self.model.model(dec_input_ids, cache=None)
            logits = self.model.lm_head(hidden_states)
            qa_loss = self._compute_qa_loss(logits, labels) if labels is not None else mx.array(0.0)
            return logits, qa_loss, mx.array(0.0)
        
        else:
            # Standard SFT forward (Unified)
            if input_ids is None:
                return mx.zeros((1,1,1)), mx.array(0.0), mx.array(0.0)
                
            hidden_states = self.model.model(input_ids, cache=None)
            
            # MSE loss logic
            mse_loss = mx.array(0.0)
            if self.mem_token_ids:
                mem_ids_arr = mx.array(self.mem_token_ids)
                mask = mx.any(input_ids[..., None] == mem_ids_arr, axis=-1)
                attn = (attention_mask if attention_mask is not None else mx.ones_like(input_ids)).astype(mx.bool_)
                mem_mask = mask & attn
                non_mem_mask = (~mask) & attn
                
                mem_len = mem_mask.sum(axis=1)
                non_mem_len = non_mem_mask.sum(axis=1)
                
                if not ((mem_len == 0).any() or (non_mem_len == 0).any()):
                    mem_sum = (hidden_states * mem_mask[..., None]).sum(axis=1)
                    non_mem_sum = (hidden_states * non_mem_mask[..., None]).sum(axis=1)
                    mem_mean = mem_sum / (mem_len[..., None] + 1e-8)
                    non_mem_mean = non_mem_sum / (non_mem_len[..., None] + 1e-8)
                    mse_loss = nn.losses.mse_loss(non_mem_mean, mem_mean)
            
            logits = self.model.lm_head(hidden_states)
            qa_loss = self._compute_qa_loss(logits, labels) if labels is not None else mx.array(0.0)
            
            return logits, qa_loss, mse_loss

    def compress(self, input_ids: mx.array, attention_mask: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Compress input documents: extract hidden states for memory tokens and calculate MSE loss.
        """
        # 1. Forward pass using the internal LlamaModel to get hidden states (last layer normalized)
        # mlx_lm model returns norm(h)
        emb = self.model.model(input_ids, cache=None)
        
        # 2. Create mask for memory tokens
        # Equivalent to torch.isin(input_ids, mem_token_ids)
        # user input_ids: [B, L], mem_token_ids: [M]
        # expand: [B, L, 1] == [M] -> [B, L, M] -> any(-1) -> [B, L]
        # 2. Memory token masking
        # Calculate mask: which tokens are memory tokens?
        # input_ids: [B, L]
        # self.mem_token_ids: List[int]
        if not self.mem_token_ids:
             mask = mx.zeros_like(input_ids, dtype=mx.bool_)
        else:
             mem_ids_arr = mx.array(self.mem_token_ids)
             mask = mx.any(input_ids[..., None] == mem_ids_arr, axis=-1) # [B, L]
        
        # 3. Calculate MSE loss
        # masks types
        attn = attention_mask.astype(mx.bool_)
        mem_mask = mask & attn
        non_mem_mask = (~mask) & attn
        
        # Lengths
        mem_len = mem_mask.sum(axis=1)
        non_mem_len = non_mem_mask.sum(axis=1)
        
        # Validation (mlx arrays computation is lazy, checks happen at eval)
        # We can skip strict checks or implement them if needed. 
        # For optimization, we assume data is correct.
        
        # Sums
        mem_sum = (emb * mem_mask[..., None]).sum(axis=1)
        non_mem_sum = (emb * non_mem_mask[..., None]).sum(axis=1)
        
        # MSE loss
        mem_sum = (emb * mem_mask[..., None]).sum(axis=1)
        non_mem_sum = (emb * non_mem_mask[..., None]).sum(axis=1)
        mem_len = mem_mask.sum(axis=1)
        non_mem_len = non_mem_mask.sum(axis=1)
        
        mem_mean = mem_sum / (mem_len[..., None] + 1e-8)
        non_mem_mean = non_mem_sum / (non_mem_len[..., None] + 1e-8)
        mse_loss = nn.losses.mse_loss(non_mem_mean, mem_mean)

        k = self.config.doc_max_length // self.config.compr_rate
        # Let's extract last k tokens per doc
        mem_embs = emb[:, -k:, :]
        
        return mem_embs, mse_loss

    def compress_query(self, input_ids: mx.array, attention_mask: mx.array) -> mx.array:
        """
        Query reasoning compression for stage 2.
        Equivalent to _compr_query_reasoner_stage2.
        Returns flattened memory embeddings [B, K*D].
        """
        # 1. Forward pass
        emb = self.model.model(input_ids, cache=None)
        
        # 2. Extract using slicing
        n_mem_tokens = self.config.doc_max_length // self.config.compr_rate
        
        mem_embs = emb[:, -n_mem_tokens:, :] # [B, K, D]
        
        # Reshape to [B, -1] -> [B, K*D]
        mem_embs = mem_embs.reshape(input_ids.shape[0], -1)
        
        return mem_embs

    def compute_retrieval_loss(self, query_reps: mx.array, doc_reps: mx.array, pos_indices: mx.array):
        """
        query_reps: [B, D_q]
        doc_reps: [B, N_docs, D_doc]
        pos_indices: [B] (index of positive document)
        """
        # doc_reps are often [B*N_docs, K, D]. We need to pool them.
        # Let's assume passed doc_reps are already pooled or we pool them here.
        if len(doc_reps.shape) == 3:
            # [B, N_docs, D] assumed
            pass
        else:
            # Fallback
            pass
            
        # 1. Normalize
        q_norm = query_reps / (mx.linalg.norm(query_reps, axis=-1, keepdims=True) + 1e-8)
        d_norm = doc_reps / (mx.linalg.norm(doc_reps, axis=-1, keepdims=True) + 1e-8)
        
        # 2. Score
        # scores = q_norm @ d_norm.T
        scores = mx.matmul(q_norm[:, None, :], d_norm.transpose(0, 2, 1)).squeeze(1)
        
        # 3. Cross Entropy
        loss = nn.losses.cross_entropy(scores * 20.0, pos_indices)
        return loss.mean()
