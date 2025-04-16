import numpy as np
from scipy import stats
data= [ 0.9019504189491272, 0.8485071659088135, 0.713484525680542, 0.7152236700057983, 0.6256791353225708, 0.6102746725082397, 0.6219843626022339, 0.7845290899276733, 0.8134503960609436, 0.8217523097991943, 0.9789109230041504, 0.9890298247337341, 0.9330385327339172, 0.9690139889717102, 0.9543581008911133, 0.9988434314727783, 1.0, 0.9999966621398926, 0.8852779865264893
    ]
c=0
c = sum(1 for i in data if i==1.0)
if c > 1:
    mod = np.mean(data)  # Use mean if multiple 1.0 values exist
    print('mean')
else:
    mode_result = stats.mode(data, keepdims=False)
    mod = mode_result.mode if hasattr(mode_result, "mode") else mode_result[0]
    mod = min(mod + 0.2, 1.0)  # Ensure mod doesn't exceed 1.0
    print('mode')

print("seizure prob:",mod)