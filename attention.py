import torch 
import torch.nn as nn 
import math
# (B, T, d_model)
# d_model = d_k for single head
class SingleHeadAttention(nn.Module):
    def __init__(self , d_model , d_k):
        super(SingleHeadAttention , self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.Q = nn.Linear(d_model , d_k , bias = False)
        self.K = nn.Linear(d_model , d_k , bias = False)
        self.V = nn.Linear(d_model , d_k , bias = False)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self , x , return_weights = False):
        T =x.size(1)
        device = x.device
        mask = torch.tril(torch.ones(T , T , device=x.device))
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)
        scores = q@k.transpose(-2 , -1)/ math.sqrt(self.d_k)
        scores = scores.masked_fill( mask == 0, float("-inf"))
        w0 = self.softmax(scores)@v
        if(return_weights):
            return self.softmax(scores) , w0
        else:
            return w0

class MultiHeadAttention(nn.Module):
    def __init__(self , d_model , h ):
        super(MultiHeadAttention , self).__init__()
        self.d_model = d_model
        self.h = h
        assert d_model %h == 0
        self.d_k = d_model//h
        self.multi = nn.ModuleList([SingleHeadAttention(d_model , self.d_k ) for _ in range(h)])
        self.out_proj = nn.Linear(d_model , d_model , bias = False)

    def forward(self , x , return_weights = False):
        if(return_weights):
            results = [A(x, return_weights) for A in self.multi]
            ws = [r[0] for r in results]
            out = [r[1] for r in results]
            scores = self.out_proj(torch.cat(out , dim = -1))
            return  torch.stack(ws , dim =1) , scores 
        else:
            head_outs = [A(x) for A in self.multi]
            head_outs = torch.cat(head_outs , dim = -1)
            return self.out_proj(head_outs)