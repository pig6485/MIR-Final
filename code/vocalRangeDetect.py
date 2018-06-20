import numpy as np
import librosa

def vocalRangeDetect(songNotes, userNotes):
    diff = np.diff(songNotes)
    print(diff)
    cnt = 0
    for i in diff:
        cnt += 1
        if i == 0:
            songNotes[cnt] = 0
    score = 0.0
    for i in songNotes:
        score += userNotes[i]
    l = len([x for x in songNotes if x != 0])
    hard = 1-score/l
    
    return np.round(100*hard, decimals = 1)

songNotes = [30, 35, 35, 35, 43, 41, 41, 42, 39, 38]

userNotes = np.zeros(80)
userNotes[30:45] = 1

score = vocalRangeDetect(songNotes=songNotes, userNotes=userNotes)
print(score)