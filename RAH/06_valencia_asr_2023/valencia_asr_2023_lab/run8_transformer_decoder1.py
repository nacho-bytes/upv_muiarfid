import torch
import torch, torchaudio, glob
import random
import scipy.signal
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
            else:
                t0 = np.random.randint(0, len(n) - len(x))
                n = n[t0:t0+len(x)]
            n = n.numpy()
            p_x = x.std()**2
            p_n = n.std()**2
            snr = np.random.uniform(-5, 15)
            n = n * np.sqrt(p_x/p_n) * np.power(10, -snr/20)
            # print(x.shape, n.shape)
            # print(x.dtype, n.dtype)
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
    def __init__(self, data_dir='data3/train', audio_len=4*16000, transform=[identity], seq_len=10):        
        self.transform = transform
        self.audio_len = audio_len
        self.seq_len = seq_len
        self.files = sorted( glob.glob(data_dir+'/*.wav') )        
        print(len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x, fs = torchaudio.load(self.files[idx])
        if x.shape[1] < self.audio_len:
            x = torch.nn.functional.pad(x, (0, self.audio_len-x.shape[1]), value=0)
        else:
            x = x[:, :self.audio_len]

        x = x[0].numpy()
        for t in self.transform:
            x = t(x)

        label = self.files[idx].split('.')[-2].split('_')[-1]
        label = label.replace('o', '0')
        # print(x.shape, x.dtype)
        label = [int(d) for d in str(label)]
        y = [20, ] + label + [22, ]
        y = torch.nn.functional.pad(torch.tensor(y), (0, self.seq_len-len(y)), value=23)
        return x, y
    

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir='data3/test', audio_len=4*16000, seq_len=10):
        self.audio_len = audio_len  
        self.seq_len = seq_len     
        self.files = sorted(glob.glob(data_dir+'/*.wav'))        
        print(len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x, fs = torchaudio.load(self.files[idx])
        if x.shape[1] < self.audio_len:
            x = torch.nn.functional.pad(x, (0, self.audio_len-x.shape[1]), value=0)
        else:
            x = x[:, :self.audio_len]

        x = x[0]
        label = self.files[idx].split('.')[-2].split('_')[-1]
        # print(x.shape, x.dtype)
        label = label.replace('o', '0')
        label = [int(d) for d in str(label)]
        y = [20, ] + label + [22, ]
        y = torch.nn.functional.pad(torch.tensor(y), (0, self.seq_len-len(y)), value=23)
        return x, y

trainset = TrainDataset(transform=[NoiseAug(), RIRAug()])
testset = TestDataset()

x, y = trainset[0]
print(x.shape, y, trainset.files[0])

x, y = trainset[1]
print(x.shape, y, trainset.files[1])

