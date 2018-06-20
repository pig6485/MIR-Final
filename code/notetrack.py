
# coding: utf-8

import numpy as np
import librosa.core

#
#
#
def notetrack(y, sr = 22050, window_size = 3000, hop_length = 1500, out_format = 'midi'):
    # out_format: str
    # 'midi' outputs midi note number(s): int
    # 'note' outputs note name in str: str
    max_v = 0
    for i in range(round((len(y) - hop_length)/hop_length)):
        max_temp = np.sum(np.square(y[i*1500:i*1500+window_size]))
        if max_temp > max_v:
            max_v = max_temp
    print(max_v)
    
    mask_ratio = 0.01
    silence_mask = []
    acs = []
    for i in range(round((len(y) - hop_length)/hop_length)):
        acs.append(librosa.core.autocorrelate(y[i*1500:i*1500+window_size]))
        value = np.sum(np.square(y[i*1500:i*1500+window_size]))
        if value <= max_v * 0.01:
            silence_mask.append(0)
        else:
            silence_mask.append(1)
    notes = []       
    for ac, m in zip(acs, silence_mask):
        v = np.argmax(ac[10:]) + 10

        if out_format == 'midi':
            if m != 0:
                notes.append(librosa.core.hz_to_midi(sr/v))
            else:
                notes.append(0)
        else:
            if m != 0:
                notes.append(librosa.core.hz_to_note(sr/v))
            else:
                notes.append('')

    return notes
	
	
def midfilter(y, window_size = 3, zero_padding = False):
    if window_size % 2 == 1: # window_size = odd
        radius = round(( window_size - 1 )/2)
    else:
        radius = round(window_size/2)
    
    if zero_padding:
        tmp = [0] * radius + y + [0] * radius
    else:
        tmp = [y[0]] * radius + y + [y[-1]] * radius
    ret = []
    for i in range(radius, len(tmp)-radius):
        n = sorted(tmp[i-radius:i+radius+1])[1]
        #print(y[i-radius:i+radius+1], sorted(y[i-radius:i+radius+1]), n)
        ret.append(n)

    #print(tmp, ret)
    #print(len(ret), len(y))
    assert(len(ret) == len(y))
    return ret