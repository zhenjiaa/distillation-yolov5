import torch
from torch._C import FunctionSchema
import torch.nn as nn


tensor_a = torch.ones((5,6))
tensor_b = torch.zeros((5,6))
loss_f = nn.MSELoss()
print(loss_f(tensor_a,tensor_b))

print((tensor_a-tensor_b)**2)

print(torch.dist(tensor_a,tensor_b,2))