
import torchaudio
from transformers import WavLMModel
import torch
import glob
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

"""
# Load wavlm model
"""
model = WavLMModel.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")

print(model.feature_extractor)
print(model.feature_projection)
print(model.encoder)

# print number of parameters
print('number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

"""
# Test feature extractor and model output (encoder)
"""
x = torch.randn(1, 16000)

f = model.feature_extractor(x)
print(f.shape)

o = model(x)
print(model.feature_extractor)
y = o.last_hidden_state
print(y.shape)

torch.onnx.export(model.feature_extractor, x, "feature_extractor.onnx")

"""
# Embeddings from feature extractor
"""

def embeddings(pattern):
    files_ = glob.glob(pattern)
    e = []
    for wav in sorted(files_):
        x, fs = torchaudio.load(wav)
        f = model.feature_extractor(x)
        e.append( f.mean(2) )
    return torch.cat(e).detach().numpy()

"""
## Embeddings from feature extractor: gender
"""

e1 = embeddings('data1/test/*m*2.wav')
e2 = embeddings('data1/test/*f*2.wav')
e = np.concatenate([e1,e2])
print(e1.shape, e2.shape)
et = TSNE(n_components=2).fit_transform(e)
plt.clf()
plt.scatter(et[:len(e1),0], et[:len(e1),1], c='b')
plt.scatter(et[len(e1):,0], et[len(e1):,1], c='r')
plt.savefig('out/embeddings11.png')

"""
## Embeddings from feature extractor: speaker
"""

e1 = embeddings('data1/test/*m_51_2.wav')
e2 = embeddings('data1/test/*m_53_2.wav')
e3 = embeddings('data1/test/*m_54_2.wav')
e4 = embeddings('data1/test/*m_55_2.wav')

e = np.concatenate([e1,e2,e3,e4])
l = np.concatenate([np.zeros(len(e1)), np.ones(len(e2)), np.ones(len(e3))*2, np.ones(len(e4))*3])
print(e1.shape, e2.shape, e3.shape, e4.shape)
et = TSNE(n_components=2).fit_transform(e)
plt.clf()
plt.scatter(et[l==0,0], et[l==1,1], c='b')
plt.scatter(et[l==1,0], et[l==1,1], c='r')
plt.scatter(et[l==2,0], et[l==2,1], c='g')
plt.scatter(et[l==3,0], et[l==3,1], c='y')
plt.savefig('out/embeddings12.png')

"""
## Embeddings from feature extractor: word
"""
e1 = embeddings('data1/test/*m_51_1.wav')
e2 = embeddings('data1/test/*m_51_2.wav')
e3 = embeddings('data1/test/*m_51_3.wav')
e4 = embeddings('data1/test/*m_51_4.wav')

e = np.concatenate([e1,e2,e3,e4])
l = np.concatenate([np.zeros(len(e1)), np.ones(len(e2)), np.ones(len(e3))*2, np.ones(len(e4))*3])

print(e1.shape, e2.shape, e3.shape, e4.shape)
et = TSNE(n_components=2).fit_transform(e)
plt.clf()
plt.scatter(et[l==0,0], et[l==1,1], c='b')
plt.scatter(et[l==1,0], et[l==1,1], c='r')
plt.scatter(et[l==2,0], et[l==2,1], c='g')
plt.scatter(et[l==3,0], et[l==3,1], c='y')
plt.savefig('out/embeddings13.png')

"""
# Embeddings from model output (encoder)
"""


def embeddings2(pattern):
    files_ = glob.glob(pattern)
    e = []
    for wav in sorted(files_):
        x, fs = torchaudio.load(wav)
        o = model(x).last_hidden_state
        e.append( o.mean(1) )
    return torch.cat(e).detach().numpy()

"""
# Embeddings from model output (encoder): gender
"""

e1 = embeddings2('data1/test/*m*2.wav')
e2 = embeddings2('data1/test/*f*2.wav')
e = np.concatenate([e1,e2])
print(e1.shape, e2.shape)
et = TSNE(n_components=2).fit_transform(e)
plt.clf()
plt.scatter(et[:len(e1),0], et[:len(e1),1], c='b')
plt.scatter(et[len(e1):,0], et[len(e1):,1], c='r')
plt.savefig('out/embeddings21.png')

"""
# Embeddings from model output (encoder): speaker
"""
e1 = embeddings2('data1/test/*m_51_2.wav')
e2 = embeddings2('data1/test/*m_53_2.wav')
e3 = embeddings2('data1/test/*m_54_2.wav')
e4 = embeddings2('data1/test/*m_55_2.wav')

e = np.concatenate([e1,e2,e3,e4])
l = np.concatenate([np.zeros(len(e1)), np.ones(len(e2)), np.ones(len(e3))*2, np.ones(len(e4))*3])
print(e1.shape, e2.shape, e3.shape, e4.shape)
et = TSNE(n_components=2).fit_transform(e)
plt.clf()
plt.scatter(et[l==0,0], et[l==1,1], c='b')
plt.scatter(et[l==1,0], et[l==1,1], c='r')
plt.scatter(et[l==2,0], et[l==2,1], c='g')
plt.scatter(et[l==3,0], et[l==3,1], c='y')
plt.savefig('out/embeddings22.png')


"""
# Embeddings from model output (encoder): word
"""
e1 = embeddings2('data1/test/*m_51_1.wav')
e2 = embeddings2('data1/test/*m_51_2.wav')
e3 = embeddings2('data1/test/*m_51_3.wav')
e4 = embeddings2('data1/test/*m_51_4.wav')

e = np.concatenate([e1,e2,e3,e4])
l = np.concatenate([np.zeros(len(e1)), np.ones(len(e2)), np.ones(len(e3))*2, np.ones(len(e4))*3])

print(e1.shape, e2.shape, e3.shape, e4.shape)
et = TSNE(n_components=2).fit_transform(e)
plt.clf()
plt.scatter(et[l==0,0], et[l==1,1], c='b')
plt.scatter(et[l==1,0], et[l==1,1], c='r')
plt.scatter(et[l==2,0], et[l==2,1], c='g')
plt.scatter(et[l==3,0], et[l==3,1], c='y')
plt.savefig('out/embeddings23.png')



