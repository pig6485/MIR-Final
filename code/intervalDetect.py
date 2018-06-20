import numpy as np
import librosa

# from easy to hard
# 3 5 (0.1)
# 2 4 6 8 (0.3)
# 3b 6b 7b (0.5)
# 2b 7 (0.8)
# 5b >8 (1)

interval_template = [0, 0.8, 0.3, 0.5, 0.1, 0.3, 1, 0.1, 0.5, 0.3, 0.5, 0.8, 0.3, 1]

def intervalDetect(notes):
    interval = np.diff(notes)
    for i in range(len(interval)):
        if np.abs(interval[i]) > 13: interval[i] = 13
    print(interval)
    score = 0.0
    for i in interval:
        score += interval_template[np.abs(i)]
        
    l = len([x for x in interval if x != 0])
    
    
    return np.round(100*score/l, decimals=1)
    
notes = [30, 35, 35, 35, 43, 41, 41, 42, 39, 38]
score = intervalDetect(notes=notes)
print(score)