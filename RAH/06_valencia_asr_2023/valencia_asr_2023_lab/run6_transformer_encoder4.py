import torch
import torch, torchaudio, glob
import scipy.signal
import numpy as np
import random

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


class SpecAug(torch.nn.Module):
    def __init__(self, prob_t_warp=0.5,
                       t_factor=(0.9, 1.1), 
                       f_mask_width = (0, 8), 
                       t_mask_width = (0, 10),
                       nb_f_masks=[1,2], 
                       nb_t_masks=[1,2], 
                       ):
        super().__init__()
        self.t_factor = t_factor
        self.f_mask_width = f_mask_width
        self.t_mask_width = t_mask_width
        self.nb_f_masks = nb_f_masks
        self.nb_t_masks = nb_t_masks
        self.prob_t_warp = prob_t_warp

    def time_warp(self, x):
        x = torch.nn.functional.interpolate(x, size=(int(x.shape[2]*np.random.uniform(*self.t_factor)), ))
        # print('warp', x.shape[2])
        return x
    
    def freq_mask(self, x):
        for _ in range(np.random.randint(*self.nb_f_masks)):
            f = np.random.randint(*self.f_mask_width)
            f0 = np.random.randint(0, x.shape[1]-f)
            # print('f', f0, f0+f)
            x[:,f0:f0+f,:] = 0
        return x

    def time_mask(self, x):
        for _ in range(np.random.randint(*self.nb_t_masks)):
            t = np.random.randint(*self.t_mask_width)
            t0 = np.random.randint(0, x.shape[2]-t)
            # print('t', t0, t0+t)
            x[:,:,t0:t0+t] = 0
        return x

    def forward(self, x):
        if np.random.uniform() < self.prob_t_warp:
            x = self.time_warp(x)
        x = self.freq_mask(x)
        x = self.time_mask(x)
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
        # self.spec_aug = SpecAug()
        self.linear = torch.nn.Linear(feat_dim, d_model)

    def forward(self, x): 
        x = self.fe(x)
        x = (x+1e-6).log()
        # if self.training:
        #     x = self.spec_aug(x)
        x = x.transpose(1, 2)
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

class NoiseAug(object):
    def __init__(self, noise_dir='musan_small/', prob=0.5):
        self.prob = prob
        self.noises = glob.glob(noise_dir+'/*/*.wav')
        
    def __call__(self, x):
        if np.random.uniform() < self.prob:
            n = torchaudio.load( np.random.choice(self.noises) )[0][0]            
            if len(n) < len(x):
                n = torch.nn.functional.pad(n, (0, len(x)-len(n)), value=0)
            elif len(n) > len(x):
                t0 = np.random.randint(0, len(n) - len(x))
                n = n[t0:t0+len(x)]
            n = n.numpy()
            p_x = x.std()**2
            p_n = n.std()**2
            snr = np.random.uniform(5, 15)
            n = n * np.sqrt(p_x/p_n) * np.power(10, -snr/20)
            x = x + n
        return x
    
class RIRAug(object):
    def __init__(self, rir_dir='RIRS_NOISES_small/simulated_rirs_small/', prob=0.5):
        self.prob = prob
        self.rirs = glob.glob(rir_dir+'/*.wav') 

    def __call__(self, x):
        if np.random.uniform() < self.prob:
            n = len(x)
            rir = torchaudio.load( np.random.choice(self.rirs) )[0][0]
            rir = rir.numpy()
            rir = rir / np.max(np.abs(rir))
            x = scipy.signal.convolve(x, rir)
            t0 = np.argmax(np.abs(rir))
            x = x[t0:t0+n]
        return x


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
        
        x = x[0].numpy()
        for t in self.transform:
            x = t(x)

        label = self.files[idx].split('.')[-2].split('_')[-1]
        # print(x.shape, x.dtype)
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
        # print(x.shape, x.dtype)
        return x, int(label)

trainset = TrainDataset(transform=[NoiseAug(), RIRAug()])
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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
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

# torch.save([model, opt], 'model64.pt')
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


