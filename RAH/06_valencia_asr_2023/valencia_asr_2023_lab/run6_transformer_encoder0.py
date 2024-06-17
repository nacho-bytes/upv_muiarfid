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
# Transformer Encoder

The transformer encoder is a stack of self-attention and feed-forward layers.
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
        bs = x.shape[0]
        q = self.q_linear(x).view(bs, -1, self.n_heads, self.d_head)
        k = self.k_linear(x).view(bs, -1, self.n_heads, self.d_head)
        v = self.v_linear(x).view(bs, -1, self.n_heads, self.d_head) 

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 1, 3)
        scores = torch.matmul(q, k) / self.scale

        att = torch.nn.functional.softmax(scores, dim=-1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).reshape(bs, -1, self.n_heads*self.d_head)
        
        out = self.dropout(out)
        out = self.out(out)

        return out

class Encoder(torch.nn.Module):
    def __init__(self, nb_layers=6, seq_len=200, **kwargs):
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

"""
# Feature Extractor

The feature extractor is composed of a log mel-spectrogram and a linear layer.
"""

class AudioFeatures(torch.nn.Module):
    def __init__(self, feat_dim=80, d_model=512, **kwargs):
        super().__init__()
        self.fe = torchaudio.transforms.MelSpectrogram(
                        n_fft=512, 
                        win_length=25*16, 
                        hop_length=10*16, 
                        n_mels=feat_dim)                            # 25ms window, 10ms shift
        self.linear = torch.nn.Linear(feat_dim, d_model)

    def forward(self, x): 
        x = self.fe(x)
        x = (x+1e-6).log().transpose(1, 2)
        x = self.linear(x)
        return x
    
"""
# Classification network

The classification network is composed of an audio feature extractor and a transformer encoder.
The prediction is the mean of the transformer encoder output.
"""

class ClassificationNetwork(torch.nn.Module):
    def __init__(self, output_dim, feat_dim=80, **kwargs):
        super().__init__()
        self.fe = AudioFeatures(**kwargs)
        self.encoder = Encoder(**kwargs)
        self.norm = torch.nn.LayerNorm(kwargs['d_model'])
        self.out = torch.nn.Linear(kwargs['d_model'], output_dim)

    def forward(self, x): 
        x = self.fe(x)
        x = self.encoder(x)
        x = self.norm(x)
        x = x.mean(1)
        x = self.out(x)
        return x

model = ClassificationNetwork(output_dim=10,
                              d_model=256, 
                              n_heads=4, 
                              d_head=32, 
                              dropout=0.1, 
                              d_ff=256, 
                              nb_layers=4)


print( model(torch.randn(10, 16000)).shape )

"""
# Dataset 
"""

def identity(x):
    return x

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir='data1/train', audio_len = 16000, transform=[identity]):        
        self.transform = transform
        self.audio_len = audio_len
        self.files = sorted( glob.glob(data_dir+'/*.wav') )        
        print(len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x, fs = torchaudio.load(self.files[idx])
        if x.shape[1] < self.audio_len:
            x = torch.nn.functional.pad(x, (0, self.audio_len-x.shape[1]), value=0)
        
        x = x[0]
        for t in self.transform:
            x = t(x)

        label = self.files[idx].split('.')[-2].split('_')[-1]
        return x, int(label)
    

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir='data1/test', audio_len = 16000):
        self.audio_len = audio_len       
        self.files = sorted(glob.glob(data_dir+'/*.wav'))        
        print(len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x, fs = torchaudio.load(self.files[idx])
        if x.shape[1] < self.audio_len:
            x = torch.nn.functional.pad(x, (0, self.audio_len-x.shape[1]), value=0)
        
        x = x[0]
        label = self.files[idx].split('.')[-2].split('_')[-1]
        return x, int(label)

trainset = TrainDataset()
testset = TestDataset()

"""
# Train the network
"""
device = 'cuda'
model.to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

nb_epochs = 5
batch_size = 32
model.train()   
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
for e in range(nb_epochs):
    loss_sum = 0    
    for x, y in trainloader:
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad()
        out = model(x)
        # print(out.shape, y.shape)
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item() / len(trainloader)
    print('epoch %d, loss %.4f' % (e, loss_sum))

# torch.save([model, opt], 'model60.pt')
"""
# Test the network
"""
model.eval()

err = 0
for x, y in testset:
    x = x.to(device)
    
    out = model(x[None,...])
    y_pred = out.argmax(dim=1).item()
    # print(y_pred, y)
    if y_pred != y:
        err += 1

print('error rate: %.4f' % (err/len(testset)))


