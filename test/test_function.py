import numpy as np
import random

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

def softmax_cross_entropy_loss_numerical_vs_analytical_gradient():
   def softmax(x):
      exp_x = np.exp(x - np.max(x))  # 数值稳定性
      return exp_x / np.sum(exp_x)

   def cross_entropy_loss(z, p):
      return -np.sum(z * np.log(p + 1e-8))  # 加小值防止log(0)

   # 数值梯度验证
   def numerical_gradient(z, # z:[C]
                          x, # x:[C]
                          epsilon=1e-6):
      grad = np.zeros_like(x)
      class_num = len(x)
      # 对每一维进行数值梯度计算
      for c in range(class_num):
         x_plus = x.copy()
         x_plus[c] += epsilon
         p_plus = softmax(x_plus)
         L_plus = cross_entropy_loss(z, p_plus)

         x_minus = x.copy()
         x_minus[c] -= epsilon
         p_minus = softmax(x_minus)
         L_minus = cross_entropy_loss(z, p_minus)

         # NOTE: 由微积份知识可得：
         #   dL/dx = (L(x+epsilon) - L(x-epsilon)) / (2*epsilon)
         grad[c] = (L_plus - L_minus) / (2 * epsilon)
      return grad

   # 解析梯度
   def analytical_gradient(z, x):
      p = softmax(x)
      return p - z

   # 测试
   logits = np.array([2.0, 1.0, 0.1])
   z = np.array([1, 0, 0])  # one-hot 标签

   p = softmax(logits)
   print("Softmax输出 p:", p)
   print("解析梯度:", analytical_gradient(z, logits))
   print("数值梯度:", numerical_gradient(z, logits))
   print("梯度差异:", np.abs(analytical_gradient(z, logits) - numerical_gradient(z, logits)))

def sigmoid_cross_entropy_loss_numerical_vs_analytical_gradient():
   def sigmoid(x):
      """sigmoid 函数"""
      return 1 / (1 + np.exp(-x))

   def sigmoid_cross_entropy_loss(x, z):
      """SigmoidCrossEntropyWithLogitLoss"""
      p = sigmoid(x)
      # 数值稳定的实现，避免log(0)
      loss = -np.sum(z * np.log(p + 1e-8) + (1 - z) * np.log(1 - p + 1e-8))
      return loss


   def sigmoid_ce_gradient_analytical(x, z):
      """解析梯度"""
      p = sigmoid(x)
      return p - z


   def numerical_gradient_sigmoid_ce(x, z, epsilon=1e-6):
      """数值梯度"""
      grad = np.zeros_like(x)
      for i in range(len(x)):
         x_plus = x.copy()
         x_plus[i] += epsilon
         loss_plus = sigmoid_cross_entropy_loss(x_plus, z)

         x_minus = x.copy()
         x_minus[i] -= epsilon
         loss_minus = sigmoid_cross_entropy_loss(x_minus, z)

         grad[i] = (loss_plus - loss_minus) / (2 * epsilon)
      return grad

   def sigmoid_cross_entropy_forward(x, z):
       """前向传播"""
       p = sigmoid(x)
       loss = -np.sum(z * np.log(p + 1e-8) + (1 - z) * np.log(1 - p + 1e-8))
       return loss, p

   def sigmoid_cross_entropy_backward(x, z, dout=1.0):
      """反向传播
      x: 输入logits
      z: 真实标签
      dout: 上游梯度（通常为1）
      """
      p = sigmoid(x)
      local_grad = p - z  # 本地梯度
      return dout * local_grad  # 链式法则

   # 示例
   x = np.array([1.0, -2.0, 0.5])
   z = np.array([1.0, 0.0, 1.0])

   loss, p = sigmoid_cross_entropy_forward(x, z)
   dx = sigmoid_cross_entropy_backward(x, z)

   print("前向传播:")
   print(f"x: {x}, z: {z}")
   print(f"p: {p}, loss: {loss}")
   print(f"梯度 dx: {dx}")

      # 测试

   x = np.array([2.0, 1.0, -1.0, 0.5])
   z = np.array([1.0, 0.0, 1.0, 0.0])  # 二进制标签

   p = sigmoid(x)
   loss = sigmoid_cross_entropy_loss(x, z)
   grad_analytical = sigmoid_ce_gradient_analytical(x, z)
   grad_numerical = numerical_gradient_sigmoid_ce(x, z)

   print("输入 x:", x)
   print("标签 z:", z)
   print("sigmoid输出 p:", p)
   print("损失值 L:", loss)
   print("\n解析梯度 dL/dx:", grad_analytical)
   print("数值梯度 dL/dx:", grad_numerical)
   print("梯度差异:", np.abs(grad_analytical - grad_numerical))

   # 验证极端情况
   print("\n=== 极端情况验证 ===")

   # 情况1: 完全正确预测
   x1 = np.array([10.0, -10.0])  # 很大的正数和负数
   z1 = np.array([1.0, 0.0])  # 对应正确标签
   p1 = sigmoid(x1)
   grad1 = sigmoid_ce_gradient_analytical(x1, z1)
   print(f"完全正确: x={x1}, z={z1}, p={p1}, grad={grad1}")

   # 情况2: 完全错误预测
   x2 = np.array([-10.0, 10.0])  # 很大的负数和正数
   z2 = np.array([1.0, 0.0])  # 对应错误标签
   p2 = sigmoid(x2)
   grad2 = sigmoid_ce_gradient_analytical(x2, z2)
   print(f"完全错误: x={x2}, z={z2}, p={p2}, grad={grad2}")

   # 情况3: 不确定预测
   x3 = np.array([0.0, 0.0])  # 接近决策边界
   z3 = np.array([1.0, 0.0])
   p3 = sigmoid(x3)
   grad3 = sigmoid_ce_gradient_analytical(x3, z3)
   print(f"不确定: x={x3}, z={z3}, p={p3}, grad={grad3}")

if __name__ == "__main__":
   softmax_cross_entropy_loss_numerical_vs_analytical_gradient()
   sigmoid_cross_entropy_loss_numerical_vs_analytical_gradient()
   if True:
       test_sigmoid_loss()
       test_sigmoid_loss_layer()
       test_cross_entropy_loss()
