import torch
import torch.nn as nn
import torch.nn.functional as F


def AddGaussianNoise(tensor,mean=0.,std=1.):
    return tensor + torch.randn(tensor.size(),device=tensor.device) * std + mean

# 
class SimpleTripletNetwork(nn.Module):

    def __init__(self,input_shape=(),output_size=4):
        super(SimpleTripletNetwork,self).__init__()
        self.CNN_outshape = self._get_output(input_shape)

        self.linear1 = nn.Linear(self.CNN_outshape,512)
        self.linear2 = nn.Linear(512,output_size)

        self.drop = nn.Dropout(p=0.1)

    # 获取输入
    def _get_output(self,shape):
        bs = 1
        dummy_x = torch.empty(bs,*shape)
        output_shape = dummy_x.flatten(1).size(1)
        return output_shape

    def forward(self,x):
        x = torch.relu(self.linear1(x.flatten(1)))
        x = self.drop(x)
        x = self.linear2(AddGaussianNoise(x,mean=0,std=1e-2))
        x = self.drop(x)
        return x
