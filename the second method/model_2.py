import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from resnet101 import se_resnet101
from Arcface_layer import *
from collections import OrderedDict

def myOwnLoad(model, check):
    modelState = model.state_dict()
    tempState = OrderedDict()
    for i in range(len(check.keys())):
        #print(check[list(check.keys())[i]])
        tempState[list(modelState.keys())[i]] = check[list(check.keys())[i]]
    #temp = [[0.02]*1024 for i in range(200)]  # mean=0, std=0.02
    #tempState['myFc.weight'] = torch.normal(mean=0, std=torch.FloatTensor(temp)).cuda()
    #tempState['myFc.bias']   = torch.normal(mean=0, std=torch.FloatTensor([0]*200)).cuda()
    model.load_state_dict(tempState)
    return model

class resnet(nn.Module):
    def __init__(self,num_classes=21,num_hidden=50):
        super(resnet,self).__init__()
        self.n=num_classes
        self.model1=se_resnet101()
        #self.model1=torch.nn.DataParallel(self.model1).cuda()
        model_weight = torch.load(r"C:\Users\Administrator\Downloads\se_resnet101.tar")['state_dict']
        self.model1=myOwnLoad(self.model1,model_weight)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.fc_1=AddMarginProduct(512*4,num_classes)
        self.fc_2= nn.Linear(num_hidden, num_classes)
        self.prelu_fc1 = nn.PReLU()
    def forward(self,x):
        x=self.model1(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #x = self.prelu_fc1(self.fc_1(x))
        y = self.fc_1(x)
        return y
