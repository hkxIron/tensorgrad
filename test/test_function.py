import numpy as np
import random
from tensorgrad.tensor import *
from tensorgrad.network import *
from micrograd.graph_utils import draw_dot_tensor
from micrograd.graph_utils import *
from tensorgrad.function import Functional as F
import matplotlib.pyplot as plt
import torch
import os
from torch import nn
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def test_cross_entropy_loss():
   # Example of target with class indices
   loss = nn.CrossEntropyLoss()
   input = torch.randn(3, 5, requires_grad=True)
   target = torch.empty(3, dtype=torch.long).random_(5)
   output = loss(input, target)
   output.backward()
   # Example of target with class probabilities
   input = torch.randn(3, 5, requires_grad=True)
   target = torch.randn(3, 5).softmax(dim=1)
   output = loss(input, target)
   output.backward()

def test_sigmoid_loss():
   np.random.seed(0)
   x = Tensor(np.random.random((4,1))+0.5, name='x')
   y = Tensor(np.random.randint(0,2, (4,1)))
   loss = F.SigmoidCrossEntropyWithLogitLoss.forward(x.data, y.data)
   pred = F.Sigmoid.forward(x.data)
   x_grad = F.SigmoidCrossEntropyWithLogitLoss.backward(pred, y.data)
   print("pred shape:", pred.shape, "pred", pred)
   print("loss shape:", loss.shape, "loss", loss)
   print("grad shape:", x_grad.shape, "grad", x_grad)
   a_g_tensor = x_grad


   print("="*50)
   x= torch.from_numpy(x.data).double()
   x.requires_grad = True;x.retain_grad()
   pred = nn.Sigmoid().forward(x)
   y= torch.from_numpy(y.data)

   loss_fn = nn.BCEWithLogitsLoss(reduction='none')
   loss = loss_fn(x, y)
   loss.backward(torch.ones_like(loss))
   x_grad = x.grad
   print("pred shape:", pred.shape, "pred", pred)
   print("loss shape:", loss.shape, "loss", loss)
   print("grad shape:", x_grad.shape, "grad", x_grad)

   a_g_torch = x.grad.numpy()

   print("a_g_tensor shape:", a_g_tensor.shape)
   print("a_g_torch shape:", a_g_torch.shape)
   print("a_g_tensor:", a_g_tensor)
   print("a_g_torch:", a_g_torch)
   assert np.allclose(a=a_g_tensor,b=a_g_torch, atol=1e-6)

def test_sigmoid_loss_layer():
   np.random.seed(0)
   x = Tensor(np.random.random((4,1))+0.5, name='x')
   y = Tensor(np.random.randint(0,2, (4,1)))
   loss_layer = SigmoidCrossEntropyWithLogitLossLayer()

   loss, pred = loss_layer(x, y)
   loss.backward()

   a_g_tensor = x.grad

   print("loss.grad",loss.grad)
   print("pred shape:", pred.shape, "pred", pred)
   print("loss shape:", loss.shape, "loss", loss)
   print("grad shape:", x.grad.shape, "grad", x.grad)


   print("="*50)
   x= torch.from_numpy(x.data).double()
   x.requires_grad = True;x.retain_grad()
   loss_layer = nn.BCEWithLogitsLoss(reduction='none')
   pred = nn.Sigmoid().forward(x)
   y= torch.from_numpy(y.data)

   loss = loss_layer(x, y)
   loss.retain_grad()
   loss.backward(torch.ones_like(loss))
   a_g_torch = x.grad.numpy()
   print("loss.grad",loss.grad)
   print("pred shape:", pred.shape, "pred", pred)
   print("loss shape:", loss.shape, "loss", loss)
   print("grad shape:", a_g_torch.shape, "grad", a_g_torch)


   print("a_g_tensor:", a_g_tensor)
   print("a_g_torch:", x.grad)
   assert np.allclose(a=a_g_tensor,b=a_g_torch, atol=1e-6)

if __name__ == "__main__":
   if True:
       #test_sigmoid_loss()
       test_sigmoid_loss_layer()
       #test_cross_entropy_loss()
