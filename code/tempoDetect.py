import numpy as np
import librosa

def tempoDetect(y, sr):
    tempo = np.round(librosa.beat.tempo(y))
    score = np.round(60+(tempo-80)/2, decimals=1)
    return tempo, score

y, sr = librosa.load("LizNelson_Rainfall_MIX.wav")
tempo, score = tempoDetect(y, sr)
print("tempo: ", tempo)
print("tempo score:", score)