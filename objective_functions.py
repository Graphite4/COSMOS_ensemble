import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from torch.autograd import Variable


class OVACrossEntropyLoss(_Loss):
    def __init__(self, cls: int):
        super().__init__()
        self.cls = cls

    def forward(self, input, target, **kwargs):
        ova_input = torch.stack(
            [
                input[:, : self.cls].sum(dim=1) + input[:, (self.cls + 1) :].sum(dim=1),
                input[:, self.cls],
            ],
            dim=1,
        )
        ova_target = (target == self.cls).long()

        return F.cross_entropy(ova_input, ova_target)


class MSELoss(_Loss):

    def __init__(self):
        super().__init__()

    def forward(self, input, target, **kwargs):
        # if input.ndim == 2:
        #     input = torch.squeeze(input)
        ohe = OneHotEncoder(sparse=False)
        target = torch.from_numpy(ohe.fit_transform(target.numpy().reshape(-1, 1)).astype("float32"))
        return F.mse_loss(input, target)


class OneClassMSELoss(_Loss):

    def __init__(self, c):
        super().__init__()
        self.cls = c

    def forward(self, input, target, **kwargs):
        # if input.ndim == 2:
        #     input = torch.squeeze(input)
        oc_input = input[:, self.cls]
        oc_target = (target == self.cls).float()
        return F.mse_loss(oc_input, oc_target)


# class CrossEntropyLoss(_Loss):
#
#     def __init__(self, cls=0, cls_number=0):
#         super().__init__()
#         # weights = np.zeros((cls_number))
#         # weights[int(cls)] = 1.0
#         # weights = np.full(cls_number, 0.1/(cls_number-1))
#         # weights[int(cls)] = 0.9
#         # self.weights = torch.from_numpy(weights.astype("float32"))
#
#     def forward(self, input, target, **kwargs):
#         # return F.cross_entropy(input, target, weight=self.weights)
#         return F.cross_entropy(input, target, reduction='mean')


class CrossEntropyLoss(torch.nn.CrossEntropyLoss):

    def __init__(self):
        super().__init__(reduction='mean')

    def __call__(self,outputs, target, **kwargs):
        if outputs.ndim == 2:
            outputs = torch.squeeze(outputs)
        if target.dtype != torch.float:
            target = target.float()
        return super().__call__(outputs, target)


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target, **kwargs):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class OVAFocalLoss(nn.Module):
    def __init__(self, cls, gamma=0, alpha=None, size_average=True):
        super(OVAFocalLoss, self).__init__()
        self.cls = cls
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target, **kwargs):
        ova_input = torch.stack(
            [
                input[:, : self.cls].sum(dim=1) + input[:, (self.cls + 1):].sum(dim=1),
                input[:, self.cls],
            ],
            dim=1,
        )
        ova_target = (target == self.cls).long()
        ova_target = ova_target.view(-1, 1)

        logpt = F.log_softmax(ova_input)
        logpt = logpt.gather(1, ova_target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != ova_input.data.type():
                self.alpha = self.alpha.type_as(ova_input.data)
            at = self.alpha.gather(0, ova_target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class DEOHyperbolicTangentRelaxation():

    # def __init__(self, label_name='labels', logits_name='logits', s_name='sensible_attribute', c=1):
    #     self.label_name = label_name
    #     self.logits_name = logits_name
    #     self.s_name = s_name
    #     self.c = c
    def __init__(self, sensible_argument_index, c=1):
        self.sensible_argument_index = sensible_argument_index
        self.c = c

    # def __call__(self, **kwargs):
    #     logits = kwargs[self.logits_name]
    #     labels = kwargs[self.label_name]
    #     sensible_attribute = kwargs[self.s_name]
    #
    #     n = logits.shape[0]
    #     logits = torch.sigmoid(logits)
    #     s_negative = logits[(sensible_attribute.bool()) & (labels == 1)]
    #     s_positive = logits[(~sensible_attribute.bool()) & (labels == 1)]
    #
    #     return 1/n * torch.abs(torch.sum(torch.tanh(self.c * torch.relu(s_positive))) - torch.sum(torch.tanh(self.c * torch.relu(s_negative))))

    def __call__(self, input, target, **kwargs):
        X = kwargs['X']
        n = input.shape[0]
        logits = torch.sigmoid(input)
        s_negative = logits[(X[:, self.sensible_argument_index] == 0) & (target == 1)]
        s_positive = logits[(X[:, self.sensible_argument_index] == 1) & (target == 1)]

        return 1/n * torch.abs(torch.sum(torch.tanh(self.c * torch.relu(s_positive))) - torch.sum(torch.tanh(self.c * torch.relu(s_negative))))

