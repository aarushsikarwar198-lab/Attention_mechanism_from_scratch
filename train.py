import torch 
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention
from data import get_batch , train , val , rv_id

vocab_size = 65 
block_size = 64
d_model = 128
heads = 4
batch_size = 64
lr = 1e-3
steps = 30000
device = "cuda" if torch.cuda.is_available() else "cpu"

#tiny encoder based language model. It has a token embedding, positional embedding, multihead attention and a linear layer to project the output to vocab size
class TinyLM(nn.Module):
    def __init__(self , d_model , heads , vocab_size , block_size):
        super(TinyLM , self).__init__()
        self.d_model = d_model
        self.heads = heads
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size , d_model)
        self.positional_embedding = nn.Embedding(block_size , d_model)  # learned positions, one vector per slot in the context
        self.attn = MultiHeadAttention(d_model , heads)
        self.output = nn.Linear(d_model , vocab_size)  # d_model -> logits over the vocab

    def forward(self , x , return_weights = False):
        B, T = x.shape
        t_e = self.token_embedding(x)
        p_e = self.positional_embedding(torch.arange(T, device = x.device) )  # broadcasts over the batch
        e_e = t_e + p_e
        if(not return_weights):
            out = self.attn(e_e)
            out_proj = self.output(out)
            return out_proj
        else:
            # also hand back the attention weights for plotting/debugging
            weights , out =  self.attn(e_e , return_weights = True)
            out_proj = self.output(out)
            return out_proj , weights

def compute_loss(model , x , y):
    logits = model(x)
    B , T , V = logits.shape
    # flatten batch and time so every position is one classification example
    return F.cross_entropy(logits.view(B*T , V) , y.view(B*T))

@torch.no_grad()
def estimate_loss(model , n_batches = 20):
    model.eval()
    out = {}
    for name, data in [("train", train), ("val", val)]:
        losses = []
        for _ in range(n_batches):
            x, y = get_batch(data, batch_size, block_size)
            x, y = x.to(device), y.to(device)
            losses.append(compute_loss(model, x, y).item()) # the tensor lives on the GPU. .item() copies the value to the CPU
        out[name] = sum(losses) / len(losses)  # average over random batches, so it's a noisy estimate
    model.train()
    return out 

def Train_model(model , optimizer , steps , device):
    model.train()
    for step in range(steps):
       x , y  = get_batch(train, batch_size, block_size)
       x , y = x.to(device) , y.to(device)
       B , T = x.shape
       loss = compute_loss(model , x , y)
       optimizer.zero_grad()
       loss.backward()
       torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # keep gradients from blowing up
       optimizer.step()

       # log every 200 steps and at the end
       if (step%200 == 0 or step == steps-1):
           l = estimate_loss(model)
           print(f"step {step}: train {l['train']:.4f} | val {l['val']:.4f}")
    torch.save(model.state_dict(), "model.pt")  # weights only, reload with load_state_dict

       
if __name__ == "__main__":
    torch.manual_seed(0)
    model = TinyLM(d_model, heads, vocab_size, block_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # sanity checks before training
    x, y = get_batch(train, 4, block_size)
    x, y = x.to(device), y.to(device)
    logits, weights = model(x, return_weights=True)
    print(logits.shape, weights.shape)        # [4, 64, 65] [4, 4, 64, 64]
    print(compute_loss(model, x, y).item())   # ~4.17, which is ln(65): uniform guessing at init

    Train_model(model, optimizer, steps, device)