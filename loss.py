'''
AAMsoftmax loss function copied from voxceleb_trainer: https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
'''

import torch, math
import torch.nn as nn
import torch.nn.functional as F
from tools import *

class AAMsoftmax(nn.Module):

    def __init__(self, n_class, m, s):
        
        super(AAMsoftmax, self).__init__()
        self.m = m
        self.s = s
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, 192), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label=None):
        
        #print(x)
        #print(x.size())
        #print(self.weight.size())


        #print("This is the label in classifier:", label)

        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        if label is not None:
            # If label is a single integer, convert it to a tensor
            if not isinstance(label, torch.Tensor):
                label = torch.tensor([label]).cuda()

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        #print(output)

        loss = self.ce(output, label)
        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]

        return loss, prec1, output
    
    """
    def __init__(self, n_class, m=0.5, s=30.0):
        super(AAMsoftmax, self).__init__()
        
        self.weight = nn.Parameter(torch.randn(n_class, 192))
        self.m = m
        self.s = s

    def forward(self, x, labels):
        # Normalize input embeddings
        x = F.normalize(x, p=2, dim=-1)

        # Normalize weight vectors
        weight = F.normalize(self.weight, p=2, dim=-1)

        # Calculate cosine similarity between input embeddings and weight vectors
        logits = F.linear(x, weight)

        # Calculate the angular margin loss
        m_hot = torch.zeros_like(logits)
        m_hot.scatter(1, labels.view(-1, 1).long(), self.m)
        logits -= m_hot

        # Scale logits for better optimization
        logits *= self.s

        # Calculate cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        prec1 = accuracy(logits.detach(), labels.detach(), topk=(1,))[0]

        return loss, prec1
    """