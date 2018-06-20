
# coding: utf-8

# In[5]:

import librosa.core as lib
import numpy as np
from librosa.display import specshow
from librosa.core import amplitude_to_db
from librosa.feature import chroma_stft
import matplotlib.pyplot as plt


# In[6]:

y, sr = lib.load('./data/13_LeadVox.wav')


# In[18]:

stft = lib.stft(y)
specshow(amplitude_to_db(np.abs(stft),
                         ref = np.max),
         x_axis='time', y_axis='log')
plt.show()


# In[35]:

pitches, magnitudes = lib.piptrack(y=y, sr=sr)


# In[49]:




# In[32]:

import librosa.onset
#odf = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)

def ACF(y, window_size = 3000, hop_length = 1500):
    max_v = 0
    for i in range(round((len(y) - hop_length)/hop_length)):
        max_temp = np.sum(np.square(y[i*1500:i*1500+window_size]))
        if max_temp > max_v:
            max_v = max_temp
    print(max_v)
    
    mask_ratio = 0.01
    silence_mask = []
    for i in range(round((len(y) - hop_length)/hop_length)):
        lib.autocorrelate(y[i*1500:i*1500+window_size])
        value = np.sum(np.square(y[i*1500:i*1500+window_size]))
        if value <= max_v * 0.01:
            silence_mask.append(0)
        else:
            silence_mask.append(1)
    return acs, silence_mask
acs, mask = ACF(y)


# In[51]:

notes = []
for i in range(len(mask)):
    ac = acs[0][i]
    m = mask[i]
    v = np.argmax(ac[10:]) + 10
    print(v, m)
    notes.append(v * m)


# In[52]:

plt.plot(notes)
plt.show()


# In[57]:

notes


# In[66]:

plt.bar(range(len(mask)), [ lib.hz_to_midi(22050/n) if n != 0 else 0 for n in notes])
plt.show()


# In[81]:

ac


# In[32]:

chroma = chroma_stft(y=y, sr=sr)
specshow(chroma, y_axis='chroma', x_axis='time')
plt.show()

'''
# In[ ]:

def ifgram(X, N = 256, W = N, H = W/2, SR = 1)
# [F,D] = ifgram(X, N, W, H, SR)       Instantaneous frequency by phase deriv.
#    X is a 1-D signal.  Process with N-point FFTs applying a W-point 
#    window, stepping by H points; return (N/2)+1 channels with the 
#    instantaneous frequency (as a proportion of the sampling rate) 
#    obtained as the time-derivative of the phase of the complex spectrum
#    as described by Toshihiro Abe et al in ICASSP'95, Eurospeech'97
#    Same arguments and some common code as dpwebox/stft.m.
#    Calculates regular STFT as side effect - returned in D.
# after 1998may02 dpwe@icsi.berkeley.edu
# 2001-03-05 dpwe@ee.columbia.edu  revised version
# 2001-12-13 dpwe@ee.columbia.edu  Fixed to work when N != W
# $Header: $

#   Copyright (c) 2006 Columbia University.
# 
#   This file is part of LabROSA-coversongID
# 
#   LabROSA-coversongID is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License version 2 as
#   published by the Free Software Foundation.
# 
#   LabROSA-coversongID is distributed in the hope that it will be useful, but
#   WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#   General Public License for more details.
# 
#   You should have received a copy of the GNU General Public License
#   along with LabROSA-coversongID; if not, write to the Free Software
#   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
#   02110-1301 USA
# 
#   See the file "COPYING" for the text of the license.
    s = len(X)
    # Make sure it's a single row
    if X.shape[0] > 1:
        X = np.transpose(X)

    pi = np.pi
    #win = [0,hanning(W-1)']
    win = 0.5*(1-np.cos([0:W]/W*2*pi))

    # Window for discrete differentiation
    T = W/SR
    dwin = -pi / T * sin([0:W]/W*2*pi)

    # sum(win) takes out integration due to window, 2 compensates for neg frq
    norm = 2/np.sum(win)

    # How many complete windows?
    nhops = 1 + np.floor((s - W)/H)

    F = np.ros(1 + N/2, nhops)
    D = np.zeros(1 + N/2, nhops)

    nmw1 = np.floor( (N-W)/2 )
    nmw2 = N-W - nmw1

    ww = 2*pi*[0:N]*SR/N

    for h = 1:nhops
        u = X((h-1)*H + [1:W]);
        #  if(h==0)
        #	plot(u)
        #  end
        # Apply windows now, while the length is right
        wu = win.*u;
        du = dwin.*u;

        # Pad or truncate samples if N != W
        if N > W
        wu = [zeros(1,nmw1),wu,zeros(1,nmw2)];
        du = [zeros(1,nmw1),du,zeros(1,nmw2)];
        end
        if N < W
        wu = wu(-nmw1+[1:N]);
        du = du(-nmw1+[1:N]);
        end
        # FFTs of straight samples plus differential-weighted ones
        t1 = fft(fftshift(du));
        t2 = fft(fftshift(wu));
        # Scale down to factor out length & window effects
        D(:,h) = t2(1:(1 + N/2))'*norm;

        # Calculate instantaneous frequency from phase of differential spectrum
        t = t1 + j*(ww.*t2);
        a = real(t2);
        b = imag(t2);
        da = real(t);
        db = imag(t);
        instf = (1/(2*pi))*(a.*db - b.*da)./((a.*a + b.*b)+(abs(t2)==0));
        # 1/2pi converts rad/s into cycles/s
        # sampling rate already factored in as constant in dwin & ww
        # so result is in Hz

        F(:,h) = instf(1:(1 + N/2))';

    return F, D


# In[34]:

np.pi
'''
