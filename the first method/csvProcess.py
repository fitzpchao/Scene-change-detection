import pandas as pd
import numpy as np
# all of the label of pair  which not contains buildings ---> not changed(1)
def changelabel():
    table = pd.read_table("F:/pc/change_detection_tf/csv/Building.csv", header=None, sep=',')
    #table = table.sample(frac=1.0)
    mask = table.values
    change = []
    unchange = []
    for i in range(mask.shape[0]):
        if (mask[i][2] == 100000):
            change.append([mask[i][0], mask[i][1], mask[i][2], mask[i][3]])
        else:
            unchange.append([mask[i][0], 1, mask[i][2], mask[i][3]])

    change = np.array(change)
    print(change.shape)

    change = pd.DataFrame(change)
    change.to_csv('F:/pc/change_detection_tf/csv/Building.csv', header=None, index=None)
    unchange = np.array(unchange)
    print(unchange.shape)
    unchange = pd.DataFrame(unchange)
    unchange.to_csv('F:/pc/change_detection_tf/csv/Not_Building.csv', header=None, index=None)

def spilt_trainAndVal():
    table = pd.read_table("F:/pc/change_detection_tf/csv/3labels.csv", header=None, sep=',')
    table = table.sample(frac=1.0)
    mask = table.values
    change = []
    unchange = []
    count=mask.shape[0]
    print(count)
    for i in range(count):
        if (i <= count*0.8):
            change.append([mask[i][0], mask[i][1], mask[i][2], mask[i][3]])
        else:
            unchange.append([mask[i][0], mask[i][1] , mask[i][2], mask[i][3]])

    change = np.array(change)
    print(change.shape)

    change = pd.DataFrame(change)
    change.to_csv('F:/pc/change_detection_tf/csv/cd_Building_train.csv', header=None, index=None)
    unchange = np.array(unchange)
    print(unchange.shape)
    unchange = pd.DataFrame(unchange)
    unchange.to_csv('F:/pc/change_detection_tf/csv/cd_Building_test.csv', header=None, index=None)

def count():
    table = pd.read_table("F:/pc/change_detection_tf/csv/CD_Building.csv", header=None, sep=',')
    table = table.sample(frac=1.0)
    mask = table.values
    change_count = 0
    unchange_count = 0
    count = mask.shape[0]
    print(count)
    for i in range(count):
        if (mask[i][1]>0):
            change_count +=1
        else:
            unchange_count +=1

    print(change_count,unchange_count)

def unchangelist():
    table = pd.read_table("F:/pc/change_detection_tf/csv/Building.csv", header=None, sep=',')
    #table = table.sample(frac=1.0)
    mask = table.values
    change = []
    unchange = []
    for i in range(mask.shape[0]):
        if (mask[i][1] == 0):
            change.append([mask[i][0], mask[i][1], mask[i][2], mask[i][3]])
        else:
            unchange.append([mask[i][0], 1, mask[i][2], mask[i][3]])

    change = np.array(change)
    print(change.shape)

    change = pd.DataFrame(change)
    change.to_csv('F:/pc/change_detection_tf/csv/Building_change.csv', header=None, index=None)
    unchange = np.array(unchange)
    print(unchange.shape)
    unchange = pd.DataFrame(unchange)
    unchange.to_csv('F:/pc/change_detection_tf/csv/Building_unchange.csv', header=None, index=None)

if __name__=='__main__':
    spilt_trainAndVal()

