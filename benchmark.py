import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "build"))
import gpt_cpp
import tiktoken

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#0d1117",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "axes.grid":         True,
    "grid.color":        "#21262d",
    "grid.linewidth":    0.6,
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#c9d1d9",
    "font.family":       "monospace",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.titlepad":     14,
})

ACCENT   = "#58a6ff"   # blue
ACCENT2  = "#3fb950"   # green
ORANGE   = "#f78166"   # red-orange
PURPLE   = "#d2a8ff"   # purple
YELLOW   = "#e3b341"   # yellow

Path("graphs").mkdir(exist_ok=True)

# ── Load model ─────────────────────────────────────────────────────────────────
print("Loading model...")
model   = gpt_cpp.GPT2Inference("weights/gpt2_weights.bin")
enc     = tiktoken.encoding_for_model("gpt2")
prompt  = "The meaning of life is"
tokens  = enc.encode(prompt)
print(f"Prompt tokens: {tokens}")

# ══════════════════════════════════════════════════════════════════════════════
# Graph 1 — Per-layer forward pass time
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/5] Per-layer timing...")

# Warm up
for _ in range(3):
    model.forward_timed(tokens)

# Average over 10 runs
runs = 10
all_timings = []
for _ in range(runs):
    _, timings = model.forward_timed(tokens)
    all_timings.append(timings)

avg_timings = np.mean(all_timings, axis=0)
embed_time  = avg_timings[0]
block_times = avg_timings[1:13]
ln_time     = avg_timings[13]

labels = ["Embed"] + [f"Block {i}" for i in range(12)] + ["LN_f"]
values = [embed_time] + list(block_times) + [ln_time]
colors = [ACCENT2] + [ACCENT] * 12 + [ORANGE]

fig, ax = plt.subplots(figsize=(13, 5))
bars = ax.bar(labels, values, color=colors, width=0.65, zorder=3)
ax.set_xlabel("Component")
ax.set_ylabel("Time (ms)")
ax.set_title("GPT-2 Forward Pass Time per Component")
plt.xticks(rotation=35, ha="right")

for bar, val in zip(bars, values):
    if val > 0.05:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=8, color="#c9d1d9")

from matplotlib.patches import Patch
legend = [Patch(color=ACCENT2, label="Embedding"),
          Patch(color=ACCENT,  label="Transformer block"),
          Patch(color=ORANGE,  label="Final LayerNorm")]
ax.legend(handles=legend, loc="upper right",
          facecolor="#161b22", edgecolor="#30363d")

fig.tight_layout()
fig.savefig("graphs/01_per_layer_timing.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Block avg: {np.mean(block_times):.2f}ms | Total: {sum(values):.2f}ms")

# ══════════════════════════════════════════════════════════════════════════════
# Graph 2 — Attention O(N²): time vs sequence length
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/5] O(N²) attention scaling...")

seq_lengths = [8, 16, 32, 64, 128, 256, 512]
base_tokens = enc.encode("The quick brown fox jumps over the lazy dog " * 20)
times_seq   = []

for sl in seq_lengths:
    toks = base_tokens[:sl]
    # Warm up
    for _ in range(2):
        model.forward(toks)
    # Measure
    reps = 5
    t0 = time.perf_counter()
    for _ in range(reps):
        model.forward(toks)
    elapsed = (time.perf_counter() - t0) / reps * 1000
    times_seq.append(elapsed)
    print(f"  seq_len={sl:4d} → {elapsed:.1f}ms")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(seq_lengths, times_seq, color=ACCENT, marker="o",
        linewidth=2.2, markersize=7, zorder=3, label="Measured")

# Fit N² curve for overlay
coeffs = np.polyfit(np.array(seq_lengths) ** 2, times_seq, 1)
fit_x  = np.linspace(8, 512, 300)
fit_y  = coeffs[0] * fit_x ** 2 + coeffs[1]
ax.plot(fit_x, fit_y, color=ORANGE, linewidth=1.4,
        linestyle="--", label="O(N²) fit", zorder=2)

ax.set_xlabel("Sequence length (tokens)")
ax.set_ylabel("Forward pass time (ms)")
ax.set_title("Attention is O(N²): Forward Pass Time vs Sequence Length")
ax.legend(facecolor="#161b22", edgecolor="#30363d")

for sl, t in zip(seq_lengths, times_seq):
    ax.annotate(f"{t:.0f}ms", (sl, t),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=8, color="#8b949e")

fig.tight_layout()
fig.savefig("graphs/02_attention_o_n2.png", dpi=150, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# Graph 3 — Top-k sampling distribution
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/5] Top-k sampling distribution...")

logits = np.array(model.forward(tokens))
k      = 20

# Softmax of all logits
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

probs_all = softmax(logits)
sorted_idx = np.argsort(probs_all)[::-1]

# Top-k mask
topk_idx   = sorted_idx[:k]
probs_topk = np.zeros_like(probs_all)
probs_topk[topk_idx] = probs_all[topk_idx]
probs_topk /= probs_topk.sum()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Full distribution (top 100)
top100 = sorted_idx[:100]
ax1.bar(range(100), probs_all[top100] * 100, color=ACCENT, width=0.8, zorder=3)
ax1.axvline(k - 0.5, color=ORANGE, linewidth=1.5, linestyle="--",
            label=f"top-k cutoff (k={k})")
ax1.set_xlabel("Token rank")
ax1.set_ylabel("Probability (%)")
ax1.set_title("Full Distribution (top 100 tokens)")
ax1.legend(facecolor="#161b22", edgecolor="#30363d")

# Top-k filtered
top_labels = [enc.decode([i]) for i in topk_idx]
top_labels = [repr(t)[1:-1][:8] for t in top_labels]
ax2.barh(range(k), probs_topk[topk_idx] * 100,
         color=[ACCENT2 if i == 0 else ACCENT for i in range(k)],
         zorder=3)
ax2.set_yticks(range(k))
ax2.set_yticklabels(top_labels, fontsize=9)
ax2.invert_yaxis()
ax2.set_xlabel("Probability after top-k filter (%)")
ax2.set_title(f"After Top-k Filtering (k={k})")

fig.suptitle(f'Sampling Distribution — prompt: "{prompt}"',
             fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig("graphs/03_topk_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# Graph 4 — Temperature effect
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/5] Temperature effect...")

temperatures = [0.5, 0.7, 1.0, 1.5, 2.0]
colors_temp  = [ACCENT2, ACCENT, YELLOW, ORANGE, PURPLE]
logits_raw   = np.array(model.forward(tokens))
sorted_top50 = np.argsort(logits_raw)[::-1][:50]

fig, ax = plt.subplots(figsize=(11, 5))

for temp, col in zip(temperatures, colors_temp):
    scaled = logits_raw[sorted_top50] / temp
    probs  = softmax(scaled)
    ax.plot(range(50), probs * 100, color=col, linewidth=1.8,
            label=f"T={temp}", zorder=3)

ax.set_xlabel("Token rank (top 50)")
ax.set_ylabel("Probability (%)")
ax.set_title("Effect of Temperature on Output Distribution")
ax.legend(facecolor="#161b22", edgecolor="#30363d", ncol=5)
ax.annotate("← More deterministic", xy=(2, ax.get_ylim()[1] * 0.92),
            color=ACCENT2, fontsize=9)
ax.annotate("More random →", xy=(30, ax.get_ylim()[1] * 0.92),
            color=PURPLE, fontsize=9)

fig.tight_layout()
fig.savefig("graphs/04_temperature_effect.png", dpi=150, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# Graph 5 — Memory breakdown pie chart
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/5] Memory breakdown...")

MB = 1024 ** 2 / 4  # floats per MB

components = {
    "wte (token embed)":   50257 * 768,
    "wpe (pos embed)":     1024  * 768,
    "Attn QKV weights":    12 * 768 * 2304,
    "Attn proj weights":   12 * 768 * 768,
    "FFN fc weights":      12 * 768 * 3072,
    "FFN proj weights":    12 * 3072 * 768,
    "LayerNorm + biases":  12 * (768 * 4) + 768 * 2,
}

sizes  = np.array(list(components.values())) / MB
labels = list(components.keys())
colors_pie = [ACCENT, ACCENT2, ORANGE, PURPLE, YELLOW, "#79c0ff", "#56d364"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

wedges, texts, autotexts = ax1.pie(
    sizes, labels=None, colors=colors_pie,
    autopct="%1.1f%%", startangle=140,
    wedgeprops={"edgecolor": "#0d1117", "linewidth": 1.5},
    pctdistance=0.78,
)
for at in autotexts:
    at.set_fontsize(9)
    at.set_color("#0d1117")

ax1.set_title("Parameter Memory Distribution")
ax1.legend(wedges, [f"{l} ({s:.1f}MB)" for l, s in zip(labels, sizes)],
           loc="lower center", bbox_to_anchor=(0.5, -0.18),
           fontsize=8, facecolor="#161b22", edgecolor="#30363d", ncol=1)

# Bar chart version
ax2.barh(labels, sizes, color=colors_pie, zorder=3)
ax2.set_xlabel("Memory (MB)")
ax2.set_title("Memory per Component (MB)")
for i, (s, l) in enumerate(zip(sizes, labels)):
    ax2.text(s + 0.5, i, f"{s:.1f}MB", va="center", fontsize=8, color="#c9d1d9")
ax2.invert_yaxis()

total_mb = sizes.sum()
fig.suptitle(f"GPT-2 124M Memory Breakdown — Total: {total_mb:.1f}MB",
             fontsize=12)
fig.tight_layout()
fig.savefig("graphs/05_memory_breakdown.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n✓ All 5 graphs saved to graphs/")
print(f"  01_per_layer_timing.png")
print(f"  02_attention_o_n2.png")
print(f"  03_topk_distribution.png")
print(f"  04_temperature_effect.png")
print(f"  05_memory_breakdown.png")
print(f"\nTotal model memory: {total_mb:.1f}MB")
print(f"Avg block time: {np.mean(block_times):.2f}ms")
print(f"O(N²) confirmed: {times_seq[0]:.1f}ms at seq=8 → {times_seq[-1]:.1f}ms at seq=512")
