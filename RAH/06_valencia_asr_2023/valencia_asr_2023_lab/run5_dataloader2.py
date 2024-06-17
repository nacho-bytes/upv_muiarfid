
import torch, torchaudio, glob
import numpy as np
import scipy.signal


"""
# Dataset with augmentation

Sample of trainset and testset using torch.utils.data.Dataset class and augmentation with noise and RIR.
"""
def identity(x):
    return x



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

trainset = TrainDataset(transform=[NoiseAug(prob=1.0)])
x, y = trainset[0]
torchaudio.save('out/trainset_noise.wav', torch.tensor(x)[None,...].float(), 16000)

trainset = TrainDataset(transform=[RIRAug(prob=1.0)])
x, y = trainset[0]
torchaudio.save('out/trainset_rir.wav', torch.tensor(x)[None,...].float(), 16000)
