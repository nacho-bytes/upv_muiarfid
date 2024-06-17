
import torch, torchaudio, glob
"""
# Dataset

Sample of trainset and testset using torch.utils.data.Dataset class.
The task is to recognize isolated digits 0-9 from spoken audio.
The audios have a max length of 1 second and a sampling rate of 16kHz.
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
        return x, label
    

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
        return x, label

trainset = TrainDataset()
testset = TestDataset()

x, y = trainset[0]
print(x.shape, y)
