import pandas as pd
import numpy as np

table=pd.read_table('F:/pc/changedetectiondata/groundtruth.csv',header=None,sep=',')
mask=table.values
change=[]
unchange=[]
for i in range(mask.shape[0]):
    #change.append(['L19/' + mask[i][0][:-4]])
    if(mask[i][1]==0):
        change.append(['L19/'+ mask[i][0][:-4]])
    else:
        unchange.append(['L19/'+ mask[i][0][:-4]])
change=np.array(change)
change=pd.DataFrame(change)
change.to_csv('F:\pc\changedetectiondata\change.csv',header=None,index=None)
unchange=np.array(unchange)
unchange=pd.DataFrame(unchange)
unchange.to_csv('F:\pc\changedetectiondata/unchange.csv',header=None,index=None)

