import torch
import torch, torchaudio, glob
import random
import numpy as np  
def seed_everything(seed):      
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(42)

"""
# Transformer Encoder+Decoder

The transformer encoder is a stack of self-attention and feed-forward layers.
The transformer decoder is a stack of self-attention, cross-attention and feed-forward layers.
"""

class FeedForward(torch.nn.Module):
    def __init__(self, d_model=512, d_ff=1024, dropout=0.1, **kwargs):
        super().__init__()
        self.ff = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, d_ff),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_ff, d_model),
        )
        
    def forward(self, x):
        return self.ff(x)

class SelfAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads=8, d_head=64, dropout=0.1, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.scale = torch.sqrt(torch.tensor(d_head, dtype=torch.float32))
        self.norm = torch.nn.LayerNorm(d_model)
        self.q_linear = torch.nn.Linear(d_model, d_head*n_heads)
        self.v_linear = torch.nn.Linear(d_model, d_head*n_heads)
        self.k_linear = torch.nn.Linear(d_model, d_head*n_heads)
        self.dropout = torch.nn.Dropout(dropout)
        self.out = torch.nn.Linear(d_head*n_heads, d_model)

    def forward(self, x):
        x = self.norm(x)
        b = x.shape[0]
        q = self.q_linear(x).view(b, -1, self.n_heads, self.d_head)
        k = self.k_linear(x).view(b, -1, self.n_heads, self.d_head)
        v = self.v_linear(x).view(b, -1, self.n_heads, self.d_head) 

        scores = torch.einsum('bihd,bjhd->bhij', q, k) / self.scale
     
        att = scores.softmax(dim=-1)
        att = self.dropout(att)

        out = torch.einsum('bhij,bjhd->bihd', att, v).reshape(b, -1, self.n_heads*self.d_head)
        out = self.dropout(out)
        out = self.out(out)
        return out

class Encoder(torch.nn.Module):
    def __init__(self, nb_layers=6, seq_len=400, **kwargs):
        super().__init__()        
        self.pos = torch.nn.Parameter(torch.randn(1, seq_len, kwargs['d_model']))
        self.att = torch.nn.ModuleList([SelfAttention(**kwargs) for _ in range(nb_layers)])
        self.ff = torch.nn.ModuleList([FeedForward(**kwargs) for _ in range(nb_layers)])
        
    def forward(self, x):
        b, t, d = x.shape
        x = x + self.pos[:, :t, :]
        for att, ff in zip(self.att, self.ff):
            x = x + att(x)
            x = x + ff(x)            
        return x


class CausalSelfAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads=8, d_head=64, dropout=0.1, seq_len=400, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.scale = torch.sqrt(torch.tensor(d_head, dtype=torch.float32))
        self.norm = torch.nn.LayerNorm(d_model)
        self.q_linear = torch.nn.Linear(d_model, d_head*n_heads)
        self.v_linear = torch.nn.Linear(d_model, d_head*n_heads)
        self.k_linear = torch.nn.Linear(d_model, d_head*n_heads)
        self.dropout = torch.nn.Dropout(dropout)
        self.out = torch.nn.Linear(d_head*n_heads, d_model)
        
        self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len) == 0)
            

    def forward(self, x):
        x = self.norm(x)
        b, n, d = x.shape
        q = self.q_linear(x).view(b, -1, self.n_heads, self.d_head)
        k = self.k_linear(x).view(b, -1, self.n_heads, self.d_head)
        v = self.v_linear(x).view(b, -1, self.n_heads, self.d_head) 

        scores = torch.einsum('bihd,bjhd->bhij', q, k) / self.scale
        
        scores = scores.masked_fill(self.mask[:,:,:n,:n], float('-inf'))
        att = scores.softmax(dim=-1)
        att = self.dropout(att)

        out = torch.einsum('bhij,bjhd->bihd', att, v).reshape(b, -1, self.n_heads*self.d_head)
        out = self.dropout(out)
        out = self.out(out)
        return out

class CrossAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads=8, d_head=64, dropout=0.1, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.scale = torch.sqrt(torch.tensor(d_head, dtype=torch.float32))
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.q_linear = torch.nn.Linear(d_model, d_head*n_heads)
        self.v_linear = torch.nn.Linear(d_model, d_head*n_heads)
        self.k_linear = torch.nn.Linear(d_model, d_head*n_heads)
        self.dropout = torch.nn.Dropout(dropout)
        self.out = torch.nn.Linear(d_head*n_heads, d_model)
    
    def forward(self, x1, x2):
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)  
        b = x1.shape[0]
        q = self.q_linear(x1).view(b, -1, self.n_heads, self.d_head)
        k = self.k_linear(x2).view(b, -1, self.n_heads, self.d_head)
        v = self.v_linear(x2).view(b, -1, self.n_heads, self.d_head) 

        scores = torch.einsum('bihd,bjhd->bhij', q, k) / self.scale
     
        att = scores.softmax(dim=-1)
        att = self.dropout(att)

        out = torch.einsum('bhij,bjhd->bihd', att, v).reshape(b, -1, self.n_heads*self.d_head)
        out = self.dropout(out)
        out = self.out(out)
        return out
    
class Decoder(torch.nn.Module):
    def __init__(self, nb_layers=6, seq_len=400, **kwargs):
        super().__init__()        
        self.pos = torch.nn.Parameter(torch.randn(1, seq_len, kwargs['d_model']))
        self.att = torch.nn.ModuleList([CausalSelfAttention(**kwargs) for _ in range(nb_layers)])
        self.cross_att = torch.nn.ModuleList([CrossAttention(**kwargs) for _ in range(nb_layers)])
        self.ff = torch.nn.ModuleList([FeedForward(**kwargs) for _ in range(nb_layers)])
        
    def forward(self, x, enc):
        b, t, d = x.shape
        x = x + self.pos[:, :t, :]
        for att, cross_att, ff in zip(self.att, self.cross_att, self.ff):
            x = x + att(x)
            x = x + cross_att(x, enc)
            x = x + ff(x)            
        return x


class Transformer(torch.nn.Module):
    def __init__(self, vocab_size=20, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = kwargs['seq_len']
        self.emb = torch.nn.Embedding(vocab_size, kwargs['d_model'])
        self.enc = Encoder(**kwargs)
        self.dec = Decoder(**kwargs)
        self.out = torch.nn.Linear(kwargs['d_model'], vocab_size)
        
    def encoder(self, x):
        x = self.emb(x)
        return self.enc(x)

    def decoder(self, y, enc):
        y = self.emb(y)
        dec = self.dec(y, enc)
        return self.out(dec)
    
    def forward(self, x, y):
        enc = self.encoder(x)
        return self.decoder(y, enc)
                
    def loss(self, x, y):        
        logits = self(x, y[:,:-1])
        target = y[:,1:]
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, self.vocab_size), 
                                                 target.reshape(-1))
        return loss
    
    def generate(self, x):
        device = next(self.parameters()).device
        self.eval()
        x = torch.nn.functional.pad(torch.tensor(x), (0, self.seq_len-len(x)), value=23)
        
        y = [20,]
        with torch.no_grad():
            enc = self.encoder(x.unsqueeze(0).to(device)) 
            
            while y[-1] != 22 and len(y) < self.seq_len:
                logits = self.decoder(torch.tensor(y).unsqueeze(0).to(device), enc)
                y.append(logits.argmax(-1)[:,-1].item())
                
        return y

"""
# Dataset

The dataset is a list of 4 integers. The input is the 4 digits and the output is the list of digits and the sum.
"""

def num2list(n):
    return [int(d) for d in str(n)]


class Digitsumset(torch.utils.data.Dataset):
    def __init__(self, data, digit_len=4, seq_len=10):
        super().__init__()
        self.digit_len = digit_len
        self.seq_len = seq_len
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = num2list(self.data[idx])
        sum_x = num2list( sum(x) )

        y = [20, ] + x + [21, ] + sum_x + [22]

        x = torch.nn.functional.pad(torch.tensor(x), (0, self.seq_len-len(x)), value=23)
        y = torch.nn.functional.pad(torch.tensor(y), (0, self.seq_len-len(y)), value=23)
        return x, y

data = list(range(10000))
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.1, random_state=42)

trainset = Digitsumset(train)
testset = Digitsumset(test)

"""
# Train the network
"""
model = Transformer(vocab_size=24, d_model=128, nb_layers=8, d_ff=256, n_heads=8, d_head=64, dropout=0.1, seq_len=10)
print( model.generate([1,2,3,4]) )


device = 'cuda'
model.to(device)
# opt = torch.optim.Adam(model.parameters(), lr=1e-3)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

nb_epochs = 15
batch_size = 32
model.train()

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
for e in range(nb_epochs):
    loss_sum = 0
    for x, y in trainloader:
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad()
        loss = model.loss(x, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item()
    print(f'epoch {e} loss {loss_sum/len(trainloader):.2f}')

# torch.save([model, opt], 'model72.pt')

"""
# Test the network
"""

# some train samples
print( model.generate([4, 8, 9, 6]) )
print( model.generate([9, 2, 1]) )


# some test samples
print( model.generate([6, 2, 5, 2]) )
print( model.generate([4, 6, 8, 4]) )

model.eval()
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
err = 0
for i,(x, y) in enumerate(testloader):
    x = x[0].tolist()
    y = y[0].tolist()    
    y_pred = model.generate(x)    
    # print(f'{i}/{len(testloader)}')
    # print(x, ' -> ', y)
    # print(y_pred)
    i = y.index(22)
    if str(y[:i + 1]) != str(y_pred):
        err += 1
print(f'error rate {err/len(testloader):.2%},  ({err}/{len(testloader)})')
