# coding=utf-8
# Copyright 2018 jose.fonollosa@upc.edu
# Adapted by CD Martinez Hinarejos (cmartine@dsic.upv.es) for MIARFID ARF
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data generator for the DIG task using standard Kaldi data folders."""

from __future__ import division

import os.path
import subprocess
import struct
import wave

import numpy as np
import librosa
import torch
import torch.utils.data as data

# Classes for the DIG task (digits in Spanish)
CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# pylint: disable=ungrouped-imports
try:
    from subprocess import DEVNULL # python3
except ImportError:
    DEVNULL = open(os.devnull, 'wb')


def wav_read(pipe):
    if pipe[-1] == '|':
        tpipe = subprocess.Popen(pipe[:-1], shell=True, stderr=DEVNULL, stdout=subprocess.PIPE)
        audio = tpipe.stdout
    else:
        tpipe = None
        audio = pipe
    try:
        wav = wave.open(audio, 'r')
    except EOFError:
        print('EOFError:', pipe)
        exit(-1)
    sfreq = wav.getframerate()
    assert wav.getsampwidth() == 2
    wav_bytes = wav.readframes(-1)
    npts = len(wav_bytes) // wav.getsampwidth()
    wav.close()
    # convert binary chunks
    wav_array = np.array(struct.unpack("%ih" % npts, wav_bytes), dtype=float) / (1 << 15)
    return wav_array, sfreq


def get_classes():
    classes = CLASSES
    weight = None
    class_to_id = {classes[i]: i for i in range(len(classes))}
    return classes, weight, class_to_id


def get_segment(wav, seg_ini, seg_end):
    nwav = None
    if float(seg_end) > float(seg_ini):
        if wav[-1] == '|':
            nwav = wav + ' sox -t wav - -t wav - trim {} ={} |'.format(seg_ini, seg_end)
        else:
            nwav = 'sox {} -t wav - trim {} ={} |'.format(wav, seg_ini, seg_end)
    return nwav


def make_dataset(kaldi_path, class_to_id):
    text_path = os.path.join(kaldi_path, 'text')
    wav_path = os.path.join(kaldi_path, 'wav.scp')
    segments_path = os.path.join(kaldi_path, 'segments')

    with open(text_path, 'rt') as text:
        key_to_word = dict()
        for line in text:
            key, word = line.strip().split(' ', 1)
            key_to_word[key] = word

    with open(wav_path, 'rt') as wav_scp:
        key_to_wav = dict()
        for line in wav_scp:
            key, wav = line.strip().split(' ', 1)
            key_to_wav[key] = wav

    wavs = []
    if os.path.isfile(segments_path):
        with open(segments_path, 'rt') as segments:
            for line in segments:
                key, wav_key, seg_ini, seg_end = line.strip().split()
                wav_command = key_to_wav[wav_key]
                word = key_to_word[key]
                word_id = class_to_id[word]
                wav_item = [key, get_segment(wav_command, seg_ini, seg_end), word_id]
                wavs.append(wav_item)
    else:
        for key, wav_command in key_to_wav.items():
            word = key_to_word[key]
            word_id = class_to_id[word]
            wav_item = [key, wav_command, word_id]
            wavs.append(wav_item)

    return wavs


def param_loader(path, window_size, window_stride, window, normalize, max_len):
    y, sfr = wav_read(path)

    # window length
    win_length = int(sfr * window_size)
    hop_length = int(sfr * window_stride)
    n_fft = 512
    lowfreq = 20
    highfreq = sfr/2 - 400

    # melspectrogram
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=False)
    D = np.abs(S)
    param = librosa.feature.melspectrogram(S=D, sr=sfr, n_mels=40, fmin=lowfreq, fmax=highfreq, norm=None)

    # Add zero padding to make all param with the same dims
    if param.shape[1] < max_len:
        pad = np.zeros((param.shape[0], max_len - param.shape[1]))
        param = np.hstack((pad, param))

    # If exceeds max_len keep last samples
    elif param.shape[1] > max_len:
        param = param[:, -max_len:]

    param = torch.FloatTensor(param)

    # z-score normalization
    if normalize:
        mean = param.mean()
        std = param.std()
        if std != 0:
            param.add_(-mean)
            param.div_(std)

    return param


class Loader(data.Dataset):
    """A DIG task loader using Kaldi data format::
    Args:
        root (string): Kaldi directory path.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        window_size: window size for the stft, default value is .02
        window_stride: window stride for the stft, default value is .01
        window_type: typye of window to extract the stft, default value is 'hamming'
        normalize: boolean, whether or not to normalize the param to have zero mean and one std
        max_len: the maximum length of frames to use
     Attributes:
        classes (list): List of the class names.
        class_to_id (dict): Dict with items (class_name, class_index).
        wavs (list): List of (wavs path, class_index) tuples
        STFT parameters: window_size, window_stride, window_type, normalize
    """

    def __init__(self, root, transform=None, target_transform=None, window_size=.02,
                 window_stride=.01, window_type='hamming', normalize=True, max_len=97):
        classes, weight, class_to_id = get_classes()
        wavs = make_dataset(root, class_to_id)
        if not wavs:
            raise RuntimeError("Found 0 segments in '" + root + "'. Folder should be in standard Kaldi format")  # pylint: disable=line-too-long

        self.root = root
        self.wavs = wavs
        self.classes = classes
        self.weight = torch.FloatTensor(weight) if weight is not None else None
        self.class_to_idx = class_to_id
        self.transform = transform
        self.target_transform = target_transform
        self.loader = param_loader
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_type = window_type
        self.normalize = normalize
        self.max_len = max_len

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (params, target) where target is class_index of the target class.
        """
        key, path, target = self.wavs[index]
        params = self.loader(path, self.window_size, self.window_stride, self.window_type, self.normalize, self.max_len)  # pylint: disable=line-too-long
        if self.transform is not None:
            params = self.transform(params)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return key, params, target

    def __len__(self):
        return len(self.wavs)
