import os

"""
# download and unzip data
"""

os.makedirs('out', exist_ok=True)

# url = 'https://openslr.elda.org/resources/17/musan.tar.gz' #'https://www.openslr.org/resources/17/musan.tar.gz'
# # download and unzip
# if not os.path.exists('musan.tar.gz'):
#     os.system('wget ' + url)
#     os.system('tar -xzf musan.tar.gz')
url = 'http://dihana.cps.unizar.es/~cadrete/asr/musan_small.zip'
if not os.path.exists('musan_small.zip'):
    os.system('wget ' + url)
    os.system('unzip musan_small.zip')

# url = 'https://openslr.elda.org/resources/28/rirs_noises.zip'#'http://www.openslr.org/resources/28/rirs_noises.zip'
# if not os.path.exists('rirs_noises.zip'):
#     os.system('wget ' + url)
#     os.system('unzip rirs_noises.zip')
url = 'http://dihana.cps.unizar.es/~cadrete/asr/rirs_noises_small.zip'
if not os.path.exists('rirs_noises_small.zip'):
    os.system('wget ' + url)
    os.system('unzip rirs_noises_small.zip')

url = 'http://dihana.cps.unizar.es/~cadrete/asr/data1.zip'
if not os.path.exists('data1.zip'):
    os.system('wget ' + url)
    os.system('unzip data1.zip')

url = 'http://dihana.cps.unizar.es/~cadrete/asr/data3.zip'
if not os.path.exists('data3.zip'):
    os.system('wget ' + url)
    os.system('unzip data3.zip')

url = 'http://dihana.cps.unizar.es/~cadrete/asr/wav.zip'
if not os.path.exists('wav.zip'):
    os.system('wget ' + url)
    os.system('unzip wav.zip')

