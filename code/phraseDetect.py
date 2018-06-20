import numpy as np
import librosa

def phraseScore(y, sr):
    step = 0.1
    f = int(sr*step)
    h = int(0.4*f)
    rmse = librosa.feature.rmse(y, frame_length=f, hop_length=h)
    rmse = np.mean(rmse, axis=0)

    phrase = []
    isStart = False
    m = np.mean(rmse)

    for i in range(len(rmse)):
        p = rmse[i]
    
        if isStart == False and p >= 1.5*m:
            phStart = i*h/sr
            isStart = True
        elif isStart == True and p < 0.1*m:
            phEnd = i*h/sr
            phrase.append((phStart, phEnd))
            isStart = False
        else: continue
    

    phrase_len = [y-x for (x, y) in phrase]
    phrase_len = np.round(phrase_len, decimals=1)

    avg_len = np.round(np.mean(phrase_len), decimals=1)
    print("Avg phrase length: ", avg_len)
    max_len = np.max(phrase_len)

    phScore = [0]*2
    phScore[0] = np.round(avg_len*10, decimals=2)
    song_len = len(y)/sr
    phScore[1] = np.round(100*np.sum(phrase_len)/song_len, decimals=2)
    print("Phrase Score: ", np.mean(phScore))
	return phScore


# y, sr = librosa.load("LizNelson_Rainfall_RAW_01_01.wav")
# y, sr = librosa.load("audioclip-1528452182.wav")
# y, sr = librosa.load("audioclip-1528452183.wav")
# y, sr = librosa.load("audioclip-1528457846.wav")
y, sr = librosa.load("beauty_test.wav")
y, sr = librosa.load("13_LeadVox.wav")
phraseScore(y, sr)
