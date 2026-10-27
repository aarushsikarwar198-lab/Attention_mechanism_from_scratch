import torch
with open("input.txt" , 'r') as file:
    content = file.read()
    chars = sorted(set(content))
    v_id = {}
    id = 0
    for i in chars:
        v_id[i] = id
        id+= 1
    rv_id = {}
    for j in v_id.items():
        rv_id[j[1]] = j[0]
    ls = []
    for j in content:
        ls.append(v_id[j])
    data = torch.tensor(ls , dtype = torch.long)  

n = int(0.9*len(content))
train = data[:n]
val = data[n:]


def get_batch(split_data , batch_size , block_size):
    ix = torch.randint(0, len(split_data) - block_size - 1, (batch_size,))
    x = torch.stack([split_data[i : i + block_size] for i in ix])
    y = torch.stack([split_data[i + 1 : i + block_size + 1] for i in ix])
    return x, y
# for example x[0] = h e l l and y[0] = e l l o. At each position, y is the next char



vocab_size = len(chars)

def encode(s):
    return [v_id[c] for c in s]

def decode(l):
    return "".join(rv_id[i] for i in l)

if __name__ == "__main__":
    print(len(content), len(chars), data.shape)        # 1115394 65 torch.Size([1115394])
    print(len(train), len(val))                        # 1003854 111540

    x, y = get_batch(train, 4, 8)
    print(x.shape, y.shape, x.dtype)                   # [4, 8] [4, 8] torch.int64

    print("".join(rv_id[i] for i in x[0].tolist()))
    print("".join(rv_id[i] for i in y[0].tolist()))

    print(torch.equal(x[:, 1:], y[:, :-1]))            # True