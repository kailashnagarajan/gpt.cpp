import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "build"))

import gpt_cpp
import tiktoken
import numpy as np

def top_k_sampling(logits, k, temperature=1.0, seed=None):
    """
    Perform top-k sampling on a list/array of logits.

    Args:
        logits (array-like): Raw logits (unnormalized scores).
        k (int): Number of top elements to keep.
        temperature (float): Controls randomness (lower = more deterministic).
        seed (int, optional): Random seed for reproducibility.

    Returns:
        int: Index of the sampled token.
    """
    if seed is not None:
        np.random.seed(seed)

    logits = np.array(logits, dtype=np.float64)

    # Apply temperature
    logits = logits / temperature

    # Get top-k indices
    top_k_indices = np.argpartition(logits, -k)[-k:]

    # Extract top-k logits
    top_k_logits = logits[top_k_indices]

    # Convert to probabilities using softmax
    exp_logits = np.exp(top_k_logits - np.max(top_k_logits))  # for numerical stability
    probs = exp_logits / np.sum(exp_logits)

    # Sample from the top-k distribution
    sampled_index = np.random.choice(len(top_k_indices), p=probs)

    # Map back to original index
    return top_k_indices[sampled_index]

encoding = tiktoken.encoding_for_model("gpt2")
tokens = encoding.encode("What is the meaning of life?")

print(tokens)

model = gpt_cpp.GPT2Inference("weights/gpt2_weights.bin")

for i in range(150):
    logits = model.forward(tokens)
    tokens.append(top_k_sampling(logits, 20))

    decoded_ = encoding.decode(tokens)

    print(decoded_)
