import torch
from transformers import GPT2LMHeadModel
import numpy as np
import struct
import os

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
sd = model.state_dict()

# Config sizes
max_seq_len = 1024
vocab_size = 50257
padded_vocab_size = 50257
n_layer = 12
n_head = 12
n_embd = 768

# Write Header (this is similar to karpathy's llm.c)
header = np.zeros(256, dtype=np.int32)
header[0] = 20240520 # magic number
header[1] = 1 # version
header[2] = max_seq_len
header[3] = vocab_size
header[4] = padded_vocab_size
header[5] = n_layer
header[6] = n_head
header[7] = n_embd

def write_tensor(f, tensor):
    t = tensor.detach().cpu().numpy().astype(np.float32)
    f.write(t.tobytes())

with open("weights/gpt2_weights.bin", "wb") as f:
    f.write(header.tobytes())

    write_tensor(f, sd["transformer.wte.weight"])
    write_tensor(f, sd["transformer.wpe.weight"])

    for i in range(n_head):
        write_tensor(f, sd[f"transformer.h.{i}.ln_1.weight"])
        write_tensor(f, sd[f"transformer.h.{i}.ln_1.bias"])
        write_tensor(f, sd[f"transformer.h.{i}.attn.c_attn.weight"].t())
        write_tensor(f, sd[f"transformer.h.{i}.attn.c_attn.bias"])
        write_tensor(f, sd[f"transformer.h.{i}.attn.c_proj.weight"].t())
        write_tensor(f, sd[f"transformer.h.{i}.attn.c_proj.bias"])
        write_tensor(f, sd[f"transformer.h.{i}.ln_2.weight"])
        write_tensor(f, sd[f"transformer.h.{i}.ln_2.bias"])
        write_tensor(f, sd[f"transformer.h.{i}.mlp.c_fc.weight"].t())
        write_tensor(f, sd[f"transformer.h.{i}.mlp.c_fc.bias"])
        write_tensor(f, sd[f"transformer.h.{i}.mlp.c_proj.weight"].t())
        write_tensor(f, sd[f"transformer.h.{i}.mlp.c_proj.bias"])

    write_tensor(f, sd["transformer.ln_f.weight"])
    write_tensor(f, sd["transformer.ln_f.bias"])

print(f"Done. File size: {os.path.getsize('weights/gpt2_weights.bin')/1024**2: .1f} MB")

total = 0

for k, v in sd.items():
    if k != "lm_head.weight":
        total += v.numel()
print(f"Total Parameters : {total}")
print(f"Expected file size : {total*4/1024**2:.1f} MB")


print ("================Quick Verification==============")
h = np.fromfile("weights/gpt2_weights.bin", dtype=np.int32, count=256)
print(f"Header : {h[:8]}")
