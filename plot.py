import torch
import math
import matplotlib.pyplot as plt

from train import TinyLM, d_model, heads, vocab_size, block_size, device
from data import get_batch, val, rv_id

L = 32  # characters shown on each axis (must be <= block_size)


def char_label(i):
    c = rv_id[i]
    return "⏎" if c == "\n" else "·" if c == " " else c


@torch.no_grad()
def get_scores(model, x):
    """Raw q·k / sqrt(d_k) per head, before mask and softmax. Returns list of (L, L)."""
    T = x.size(1)
    emb = model.token_embedding(x) + model.positional_embedding(
        torch.arange(T, device=x.device)
    )
    out = []
    for head in model.attn.multi:
        q = head.Q(emb)
        k = head.K(emb)
        s = q @ k.transpose(-2, -1) / math.sqrt(head.d_k)
        out.append(s[0].cpu())
    return out


def plot_similarity(scores, labels, out, hide_future=False):
    ncols = 2
    nrows = math.ceil(len(scores) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows), squeeze=False)

    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("white")

    for h, s in enumerate(scores):
        ax = axes[h // ncols][h % ncols]
        data = s.clone()
        if hide_future:
            data[torch.triu(torch.ones_like(data), diagonal=1).bool()] = float("nan")
        im = ax.imshow(data.numpy(), cmap=cmap)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("key")
        ax.set_ylabel("query")
        ax.set_title(f"head {h}")

    for j in range(len(scores), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


if __name__ == "__main__":
    model = TinyLM(d_model, heads, vocab_size, block_size).to(device)
    model.load_state_dict(torch.load("model.pt", map_location=device))
    model.eval()

    torch.manual_seed(1)
    x, _ = get_batch(val, 1, block_size)
    x = x[:, :L].to(device)
    labels = [char_label(i) for i in x[0].tolist()]

    scores = get_scores(model, x)
    plot_similarity(scores, labels, "similarity_full.png", hide_future=False)
    plot_similarity(scores, labels, "similarity_causal.png", hide_future=True)
    print("saved similarity_full.png and similarity_causal.png")