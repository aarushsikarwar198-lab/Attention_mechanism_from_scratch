import torch
from attention import SingleHeadAttention , MultiHeadAttention
import torch.nn.functional as F


#test to check the shape of the output and the attention weights
def test_shape_and_rows():
    x = torch.randn(2, 6, 16)
    m = MultiHeadAttention(16, 4)
    out,w = m(x , return_weights = True)
    assert out.shape == (2 , 6 ,16)
    assert w.shape == (2 , 4 , 6 , 6)
    assert torch.allclose(w.sum(-1)  , torch.ones_like(w.sum(-1)) , atol = 1e-5)


#test to check the mask i.e. the current embedding can only attend to the previous embeddings and not the future embeddings
def test_mask():
    x = torch.randn(2, 6, 16)
    m = MultiHeadAttention(16, 4)
    out, w = m(x, return_weights=True)
    assert torch.triu(w , diagonal = 1).sum() == 0


# test for causality: In a causal model, output at position i depends only on inputs at positions 0..i. So changing the last input should change only the last output
def test_causal():
    x = torch.randn(1 , 6 , 16)
    m = MultiHeadAttention(16 , 4).eval()
    y1 = m(x)
    x2 = x.clone()
    x2[: , -1] += 1.0
    y2 = m(x2)
    assert torch.allclose(y1[:, :-1] , y2[: , :-1])
    assert not torch.allclose(y1[: , -1] , y2[: , -1])


#test to check if the output of the attention module matches the output of the torch built-in scaled dot product attention function
def test_matches_torch():
    a = SingleHeadAttention(16 , 16)
    x = torch.randn(2 , 6 , 16)
    ref = F.scaled_dot_product_attention(a.Q(x) , a.K(x) , a.V(x) , is_causal = True)
    assert torch.allclose(a(x) , ref , atol = 1e-5)
    

