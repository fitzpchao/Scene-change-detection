import numpy as np

a=np.load("F:/pc/ChangeDetection/Data/DataAppend/train/bingmaps_mean.npy")
a=np.reshape(a,[3,-1])
mean=np.mean(a,axis=-1)
print(mean)
np.save("F:/pc/ChangeDetection/Data/DataAppend/train/bingmaps_mean.npy",mean)


