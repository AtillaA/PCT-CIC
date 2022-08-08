from torchsummary import summary
import numpy as np
import torch
from pct_partseg_torch import Point_Transformer_partseg 
from data import ShapeNetPart

trainval = ShapeNetPart(2048, 'trainval')
test = ShapeNetPart(2048, 'test')
data, label, seg = trainval[0]

label_one_hot = np.zeros((label.shape[0], 16))
for idx in range(label.shape[0]):
    label_one_hot[idx, label[idx]] = 1
label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))

pct_full = Point_Transformer_partseg()
summary(pct_full, [(data.shape), label_one_hot.shape])