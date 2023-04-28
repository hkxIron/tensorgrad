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
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def test_dot():
   a = np.arange(2*3).reshape((2,3))
   b = np.arange(3*4).reshape((3,4))
   c = np.dot(a,b)
   d = a@b
   print(c.shape)
   print(d.shape)
   print(c)
   print(d)

def test_boardcast_add():
   np.random.seed(0)
   # tensor
   a = Tensor(np.random.random((2,3)), name='a')
   b = Tensor(np.random.random((3)), name='b')
   c = a + b
   c.name='c'
   c.backward()
   print(c.shape)
   print(a.grad)
   print(b.grad)
   print(c.grad)
   a_g_tensor = a.grad
   b_g_tensor = b.grad
   c_dot = draw_dot_tensor(c)
   c_dot.view()


   # torch
   a = torch.from_numpy(a.data).double()
   b = torch.from_numpy(b.data).double()
   a.requires_grad = True;a.retain_grad()
   b.requires_grad = True;b.retain_grad()
   c = a + b;c.retain_grad()
   c.backward(torch.ones_like(c))
   #c.backward()
   a_g_torch = a.grad.numpy()
   b_g_torch = b.grad.numpy()
   print(c.shape)
   print(a.grad)
   print(b.grad)
   print(c.grad)

   assert np.all(a_g_tensor - a_g_torch == 0)
   assert np.all(b_g_tensor - b_g_torch == 0)


def test_boardcast_multiply():
   np.random.seed(0)
   a = Tensor(np.ones((2,3)), name='a')
   b = Tensor(np.ones((3)), name='b')
   # a:[2,3]
   # b:[3]
   # c:[2,3]
   # da:[2,3]
   # db:[3]
   c = a * b
   c.name='c'

   c.backward()

   a_g_tensor = a.grad
   b_g_tensor = b.grad
   print("c:", c.shape)
   print("a.grad:", a.grad)
   print("b.grad:", b.grad)
   print("c.grad:", c.grad)

   # torch
   a = torch.from_numpy(a.data).double()
   b = torch.from_numpy(b.data).double()
   a.requires_grad = True;a.retain_grad()
   b.requires_grad = True;b.retain_grad()
   c = a * b
   c.retain_grad()
   c.backward(torch.ones_like(c))
   #c.backward()

   a_g_torch = a.grad.numpy()
   b_g_torch = b.grad.numpy()
   print(c.shape)
   print(a.grad)
   print(b.grad)
   print(c.grad)

   assert np.all(a_g_tensor - a_g_torch == 0)
   assert np.all(b_g_tensor - b_g_torch == 0)

def test_draw_dot():
   np.random.seed(0)
   a = Tensor(np.random.random((2,3)))
   b = Tensor(np.random.random((2,3)))
   c = a + b
   d = a * b + b ** 3
   d_dot = draw_dot_tensor(d)
   d_dot.view()


def test_more_ops():
   np.random.seed(0)
   a = Tensor(np.random.random((2,3)))
   b = Tensor(np.random.random((2,3)))
   c = a + b
   d = a * b + b ** 3
   e = c + d
   f = e**2/4
   #f = e*e/4
   g = Tensor(np.random.random((3,4)))
   h = f@g
   i = Tensor(np.random.random(4))
   j = h + i.relu() # broadcast, h:[2,4], i:[4], j:[2,4]
   k = j.tanh()
   L = k.sigmoid()
   a_g_tensor = a.grad
   b_g_tensor = b.grad

   print("a shape:", a.shape)
   print("b shape:", b.shape)
   print("f shape:", f.shape)
   print("g shape:", g.shape)
   print("h shape:", h.shape)
   print("i shape:", i.shape)
   print("j shape:", j.shape)
   print("k shape:", k.shape)
   print("L shape:", L.shape)

   L.backward()
   print("a:",a.grad)
   print("b:",b.grad)
   print("c:",c.grad)
   print("e:",e.grad)

   L_dot = draw_dot_tensor(L)
   L_dot.view()


   print("torch:")
   # torch
   a = torch.from_numpy(a.data).float()
   b = torch.from_numpy(b.data).float()
   a.requires_grad = True;a.retain_grad()
   b.requires_grad = True;b.retain_grad()

   c = a + b
   d = a * b + b ** 3
   e = c + d
   f = e**2/4
   g = torch.Tensor(g.data).float()
   h = f@g
   i = torch.Tensor(i.data).float()
   #j = h + i # broadcast, h:[2,4], i:[4], j:[2,4]
   j = h + i.relu()# broadcast, h:[2,4], i:[4], j:[2,4]
   k = j.tanh()
   L = k.sigmoid()
   L.retain_grad()
   #L.backward()
   L.backward(torch.ones_like(L))
   a_g_torch = a.grad.numpy()
   b_g_torch = b.grad.numpy()

   print("a:", a.grad)
   assert np.allclose(a=a_g_tensor ,b= a_g_torch, atol=1e-6)
   assert np.allclose(a=b_g_tensor ,b= b_g_torch, atol = 1e-6)

def test_relu():
   x = Tensor(np.array([0.7, 0, -2]), name='x')
   #w = np.array([-8, 1, 0.2])
   w = Tensor(np.array([-3, 1, 0.2]))

   # 怀颖python3.7有bug
   a = w*x
   y = (1+a).relu()
   #print(type(w*x))
   y.name = 'y'
   y.backward()
   print("x:", x, "y:", y)

def test_function():
   x = Tensor(np.array([1, 0, -2]))
   y = F.Relu.forward(x.data)
   print("x:", x, "y:", y)

   x = Tensor(np.array([1, 0, -2]))
   y = F.Sigmoid.forward(x.data)
   print("x:", x, "y:", y)

   x = Tensor(np.array([1, 0, -2]))
   def get_fun(fn:F.Func):
      return fn

   y = get_fun(F.Sigmoid()).forward(x.data)
   print("x:", x, "y:", y)

def test_mean():
   np.random.seed(0)
   a = Tensor(np.random.random((2,3,4,5)), name='x')
   dims=(1,3)
   b = a.mean(axis=dims)
   b.backward()
   a_g_tensor = a.grad

   a= torch.from_numpy(a.data).double()
   a.requires_grad = True;a.retain_grad()
   b = a.mean(dim=dims)
   b.backward(torch.ones_like(b))
   a_g_torch = a.grad.numpy()

   print("a_g_tensor shape:", a_g_tensor.shape)
   print("a_g_torch shape:", a_g_torch.shape)
   print("a_g_tensor:", a_g_tensor)
   print("a_g_torch:", a_g_torch)
   assert np.allclose(a=a_g_tensor,b=a_g_torch, atol=1e-6)

def test_mean2():
   np.random.seed(0)
   a = Tensor(np.random.random((2,3,4,5)), name='x')
   dims=None
   b = a.mean(axis=dims)
   b.backward()
   a_g_tensor = a.grad

   a= torch.from_numpy(a.data).double()
   a.requires_grad = True;a.retain_grad()
   b = a.mean()
   b.backward(torch.ones_like(b))
   a_g_torch = a.grad.numpy()

   print("a_g_tensor shape:", a_g_tensor.shape)
   print("a_g_torch shape:", a_g_torch.shape)
   print("a_g_tensor:", a_g_tensor)
   print("a_g_torch:", a_g_torch)
   assert np.allclose(a=a_g_tensor,b=a_g_torch, atol=1e-6)

def test_sum():
   np.random.seed(0)
   a = Tensor(np.random.random((2,3,4,5)), name='x')
   dims=(1,3)
   b = a.sum(axis=dims)
   b.backward()
   a_g_tensor = a.grad

   a= torch.from_numpy(a.data).double()
   a.requires_grad = True;a.retain_grad()
   b = a.sum(dim=dims)
   b.backward(torch.ones_like(b))
   a_g_torch = a.grad.numpy()

   print("a_g_tensor shape:", a_g_tensor.shape)
   print("a_g_torch shape:", a_g_torch.shape)
   print("a_g_tensor:", a_g_tensor)
   print("a_g_torch:", a_g_torch)
   assert np.allclose(a=a_g_tensor,b=a_g_torch, atol=1e-6)

def test_softmax():
   np.random.seed(0)
   a = Tensor([[1.0,1.2, 1.3],
               [0.2, 0.6, 0.7]
               ], name='x')
   #a = Tensor(np.random.normal(0, 1, (2,2)), name='x')
   dims=(1)
   #b = Function.Softmax().forward(a, axis=1)
   b = a.softmax(axis=1)
   y_g = np.array([[1.0, 1.0, 1.0],
                   [1.0, 0.0, 0.0]], dtype=np.float64)
   b.backward(grad=y_g)
   a_g_tensor = a.grad
   b_d_tensor = b.data
   print("b tensor:", b.data)

   #torch
   a= torch.from_numpy(a.data).double()
   a.requires_grad = True;a.retain_grad()
   b = torch.softmax(a, dim=dims)
   print("b torch:", b.data)
   #b.backward(torch.ones_like(b))
   b.backward(torch.from_numpy(y_g))
   a_g_torch = a.grad.numpy()
   b_d_torch = b.data.numpy()

   print("a_g_tensor shape:", a_g_tensor.shape)
   print("a_g_torch shape:", a_g_torch.shape)
   print("a_g_tensor:", a_g_tensor)
   print("a_g_torch:", a_g_torch)
   assert np.allclose(a=b_d_tensor,b=b_d_torch, atol=1e-6)
   assert np.allclose(a=a_g_tensor,b=a_g_torch, atol=1e-6)

class SoftMax():
   def __init__(self):
      pass

   def _softmax(self, x):
      x = x - np.max(x, axis=1, keepdims=True)
      y = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
      return y

   def forward(self, input):
      return self._softmax(input)

   def backward(self, input, grad_output):
      out = self.forward(input)
      ret = []
      for i in range(grad_output.shape[0]):
         softmax_grad = np.diag(out[i]) - np.outer(out[i], out[i])
         ret.append(np.dot(softmax_grad, grad_output[i].T))
      ret = np.array(ret)
      return ret

def test_softmax2():
   print("test_softmax2")
   fn = SoftMax()
   x = np.array([[1.0, 1.2, 1.3], [0.2,0.6, 0.7]])
   y = fn.forward(x)
   print("y:", y)
   grad = fn.backward(x, np.array([[0,1,0], [1,0,0]]))
   print("grad:", grad)

def test_sigmoid():
   np.random.seed(0)
   a = Tensor(np.random.random((2,3)), name='x')
   b = a.sigmoid()
   b.backward()
   a_g_tensor = a.grad

   a= torch.from_numpy(a.data).double()
   a.requires_grad = True;a.retain_grad()
   b = a.sigmoid()
   b.backward(torch.ones_like(b))
   a_g_torch = a.grad.numpy()

   print("a_g_tensor shape:", a_g_tensor.shape)
   print("a_g_torch shape:", a_g_torch.shape)
   print("a_g_tensor:", a_g_tensor)
   print("a_g_torch:", a_g_torch)
   assert np.allclose(a=a_g_tensor,b=a_g_torch, atol=1e-6)

def test_tanh():
   np.random.seed(0)
   a = Tensor(np.random.random((2,3)), name='x')
   b = a.tanh()
   b.backward()
   a_g_tensor = a.grad

   a= torch.from_numpy(a.data).double()
   a.requires_grad = True;a.retain_grad()
   b = a.tanh()
   b.backward(torch.ones_like(b))
   a_g_torch = a.grad.numpy()

   print("a_g_tensor shape:", a_g_tensor.shape)
   print("a_g_torch shape:", a_g_torch.shape)
   print("a_g_tensor:", a_g_tensor)
   print("a_g_torch:", a_g_torch)
   assert np.allclose(a=a_g_tensor,b=a_g_torch, atol=1e-6)

def test_abs():
   np.random.seed(0)
   a = Tensor(np.random.random((2,3))-0.5, name='x')
   b = a.abs()
   b.backward()
   a_g_tensor = a.grad

   a= torch.from_numpy(a.data).double()
   a.requires_grad = True;a.retain_grad()
   b = a.abs()
   b.backward(torch.ones_like(b))
   a_g_torch = a.grad.numpy()

   print("a_g_tensor shape:", a_g_tensor.shape)
   print("a_g_torch shape:", a_g_torch.shape)
   print("a_g_tensor:", a_g_tensor)
   print("a_g_torch:", a_g_torch)
   assert np.allclose(a=a_g_tensor,b=a_g_torch, atol=1e-6)

def test_exp():
   np.random.seed(0)
   a = Tensor(np.random.random((2,3))-0.5, name='x')
   b = a.exp()
   b.backward()
   a_g_tensor = a.grad

   a= torch.from_numpy(a.data).double()
   a.requires_grad = True;a.retain_grad()
   b = a.exp()
   b.backward(torch.ones_like(b))
   a_g_torch = a.grad.numpy()

   print("a_g_tensor shape:", a_g_tensor.shape)
   print("a_g_torch shape:", a_g_torch.shape)
   print("a_g_tensor:", a_g_tensor)
   print("a_g_torch:", a_g_torch)
   assert np.allclose(a=a_g_tensor,b=a_g_torch, atol=1e-6)

def test_log():
   np.random.seed(0)
   a = Tensor(np.random.random((2,3))+0.5, name='x')
   b = a.log()
   b.backward()
   a_g_tensor = a.grad

   a= torch.from_numpy(a.data).double()
   a.requires_grad = True;a.retain_grad()
   b = a.log()
   b.backward(torch.ones_like(b))
   a_g_torch = a.grad.numpy()

   print("a_g_tensor shape:", a_g_tensor.shape)
   print("a_g_torch shape:", a_g_torch.shape)
   print("a_g_tensor:", a_g_tensor)
   print("a_g_torch:", a_g_torch)
   assert np.allclose(a=a_g_tensor,b=a_g_torch, atol=1e-6)

if __name__ == "__main__":
   test_draw_dot()
   if False:
      test_more_ops()
      test_softmax()
      test_log()
      test_exp()
      test_abs()
      test_tanh()
      #test_dot()
      test_sigmoid()
      test_boardcast_multiply()
      #test_boardcast_add()
      test_function()
      test_relu()
      test_mean()
      test_mean2()
      test_sum()
      test_softmax2()
