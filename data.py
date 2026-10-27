import torch
#reading text file
with open("input.txt" , 'r') as file:
    content = file.read()
    chars = sorted(set(content))
    v_id = {}
    #creating vocab_char to id dict
    id = 0
    for i in chars:
        v_id[i] = id
        id+= 1
    #creating reverse id to vocab_char
    rv_id = {}
    for j in v_id.items():
        rv_id[j[1]] = j[0]
    ls = []
    for j in content:
        ls.append(v_id[j])
    #combined tensor of text file using vocab id dict
    data = torch.tensor(ls , dtype = torch.long)  

#train - val split
n = int(0.9*len(content))
train = data[:n]
val = data[n:]


#function to get the batch data 
def get_batch(split_data , batch_size , block_size):
    ix = torch.randint(0, len(split_data) - block_size - 1, (batch_size,))
    x = torch.stack([split_data[i : i + block_size] for i in ix])
    #shifting by one for val check of the next char
    y = torch.stack([split_data[i + 1 : i + block_size + 1] for i in ix])
    return x, y
# for example x[0] = h e l l and y[0] = e l l o. At each position, y is the next char



vocab_size = len(chars)

#getting char id
def encode(s):
    return [v_id[c] for c in s]

#returning string from the ids
def decode(l):
    return "".join(rv_id[i] for i in l)

#test
if __name__ == "__main__":
    print(len(content), len(chars), data.shape)   
    print(len(train), len(val))                        

    x, y = get_batch(train, 4, 8)
    print(x.shape, y.shape, x.dtype)               

    print("".join(rv_id[i] for i in x[0].tolist()))
    print("".join(rv_id[i] for i in y[0].tolist()))

    print(torch.equal(x[:, 1:], y[:, :-1]))           