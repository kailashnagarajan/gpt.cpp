## GPT-2 in C++ 

I have followed the famous Karpathy video "Let's build GPT" but did it in C++.  

This repository only does the inference part of the karpathy video, the forward pass is done in C++. 

1. `export_weights.py` - outputs gpt2 weights as a .bin file. [**run once to generate the bin**]
2. load `gpt2_weights.bin` in C++ zero copy Eigen Matrices mmap. 
3. Python -> tiktoken.encode(prompt) -> tokenIDs (list of integers)
4. C++ -> forward_pass(tokenID's) -> logits vector `[seq_len, vocab_size]`
5. Python -> top_k_sample(logits[-1]) -> next token ID 
6. Python -> Append next token ID & loop back to step-3 
7. Python -> tiktoken.decode(all token IDs) -> string 

### Weights order : 
**Header** — 256 int32s. First 8 slots used:
- This is similar to karpathy's llm.c
magic number, version, max_seq_len, vocab_size, padded_vocab_size, n_layer, n_head, n_embd

Rest are zeros

**Weights in this order:**

`wte` — token embedding `[vocab_size, n_embd]`
`wpe` — positional embedding `[max_seq_len, n_embd]`  

(**Explanation for Positional Embedding Matrix** : Token at position 0 looks up row 0, token at position 1 looks up row 1, and so on.   
These get added to the token embeddings so the model knows where in the sequence each token sits. 1024 rows = GPT-2's maximum sequence length.)

Then for each layer (loop 0 to 11):

`ln1.weight`, `ln1.bias` - First Layer Norm
`c_attn.weight`, `c_attn.bias` — the fused QKV projection (Instead of three separate weight matrices for Q, K, and V, GPT-2 fuses them into one.)
`c_proj.weight`, `c_proj.bias` - Attention Output Projection. After all 12 attention heads compute their outputs and get concatenated back to [seq_len, 768] 
`ln2.weight`, `ln2.bias` - Second Layer Norm
`mlp.c_fc.weight`, `mlp.c_fc.bias` - FFN First Linear 
`mlp.c_proj.weight`, `mlp.c_proj.bias` - FFN Second Linear

Then finally:

`ln_f.weight`, `ln_f.bias` — the final layer norm before the output projection

`lm_head` is not written — it's weight-tied to wte, C++ code reuses it.  

The output projection — not in the file.  
To get logits over the vocabulary you do `hidden_states @ wte.T` — *multiply by the transpose of the token embedding matrix*. This is weight tying. The same 50257×768 matrix does double duty — lookup table on the way in, classifier on the way out. Saves 38M parameters and works surprisingly well.

