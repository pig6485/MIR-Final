import numpy as np
import librosa

interval_template = [0, 0.8, 0.3, 0.5, 0.1, 0.3, 1, 0.1, 0.5, 0.3, 0.5, 0.8, 0.3, 1]

def myDiff(a):
    b = np.zeros(shape=len(a)-1)
    for i in range(1, len(a)):
       if a[i] == 0 or a[i-1] == 0:
           continue
       b[i-1] = a[i]-a[i-1]
    return b

# detect the intervals in the melody

# interval from easy to hard
# 3 5 (0.1)
# 2 4 6 8 (0.3)
# 3b 6b 7b (0.5)
# 2b 7 (0.8)
# 5b >8 (1)
def intervalDetect(notes):
    
    interval = myDiff(notes)
    for i in range(len(interval)):
        if np.abs(interval[i]) > 13: interval[i] = 13
    score = 0.0
    for i in interval:
        score += interval_template[int(np.abs(i))]
        
    l = len([x for x in interval if x != 0])
    
    
    return np.round(10*score/l, decimals=1)
       
# detect the vocal range
def vocalRangeDetect(songNotes, userNotes):
    diff = np.diff(songNotes)
    cnt = 0
    for i in diff:
        cnt += 1
        if i == 0:
            songNotes[cnt] = 0
    score = 0.0
    for i in songNotes:
        score += userNotes[int(i)]
    l = len([x for x in songNotes if x != 0])
    hard = 1-score/l
    
    return np.round(10*hard, decimals = 1)

# detect the phrases length
def phraseDetect(y, sr):
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
    max_len = np.max(phrase_len)

    phScore = [0]*2 # 這行沒改，從這行開始
    phScore[0] = avg_len
    song_len = len(y)/sr
    phScore[1] = 10*np.sum(phrase_len)/song_len
    
    return np.round(np.mean(phScore), decimals=1)

# Count the number of notes changes in a period of time
def noteChange(notes, sr=22050, hop_length= 1500):
    isStart = False
    phrase = []
    change = []

    for i in range(len(notes)):
        if isStart == 0 and notes[i] != 0:
            beg = i
            isStart = True
        elif isStart == True and notes[i] == 0:
            end = i
            phrase.append(notes[beg:end])
            isStart = False
        else:
            continue

    for i in phrase:
        l = len(i)
        diff = np.diff(i)
        numChange = len([x for x in diff if x != 0])
        if numChange == 0:
            continue
        l *= hop_length/sr
        change.append(numChange/l)

    score = np.mean(change)
    score += np.mean(np.sort(change)[int(len(change)*0.9):])-np.mean(change) # mean 10% most - mean of all
    
    return np.round(score, decimals=1)
