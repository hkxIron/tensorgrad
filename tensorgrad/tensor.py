from typing import Union
from tensorgrad.util import *
from tensorgrad.function import Functional as F

import numpy as np

"""
利用ndarray实现自动微分的Tensor, 参考：https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
author: kexinhu
"""

class Tensor:
    """ stores a single scalar value and its gradient """
    data: np.array
    grad: np.array
    shape = property(fget=lambda self: self.data.shape, fset=lambda self, v: None)

    # 注意：每个值都带有操作符,以及他的子结点（即c=a+b,那么c的子结点就是a,b）
    def __init__(self, data, dtype=np.float64, _prev_nodes=(), _op='', name=''):
        if isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, Tensor):  # data是一个tensor
            self.data = data.data.copy()
        else:
            self.data = np.array(data, dtype=np.float32)

        if dtype:
            self.data = self.data.astype(dtype)  # 注意：不是self.data.dtype=dtype

        self.grad = np.zeros_like(self.data)
        # internal variables used for autograd graph construction
        self._backward_grad_fn = lambda: None
        self._prev = set(_prev_nodes)  # 需要要用set去重，因为c=a+a，他的子结点相同，均为a
        self._op = _op  # the op that produced this node, for graphviz / debugging / etc
        self.name = name
        #self.requires_grad = requires_grad
        # 标量的name直接为元素本身
        if isinstance(data, (int, float)) and len(name) == 0:
            self.name = '%.2f'%data

    # 清空梯度
    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data)

    def flatten(self):
        return self.reshape(-1)

    def reshape(self, newshape):
        # numpy中的reshape返回的的数组是原来的视图
        out = Tensor(self.data.copy().reshape(newshape), _prev_nodes=(self,), _op='reshape')
        out.grad = self.grad.copy().reshape(newshape)

        # 计算梯度
        def _backward_grad():
            self.grad += out.grad.copy().reshape(self.grad.shape)

        out._backward_grad_fn = _backward_grad

        return out

    def numpy(self) -> np.ndarray:
        return self.data.copy()

    # @property
    # def shape(self):
    #     return self.data.shape

    def set_backward_func(self, fn: callable):
        self._backward_grad_fn = fn

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, _prev_nodes=(self, other), _op='+')

        # is_boradcast = self.data.shape != other.data.shape

        # 计算梯度
        def _backward_grad():
            self_not_repeat, self_repeat_axis = get_repeat_axis(self.grad.shape, out.grad.shape)
            other_not_repeat, other_repeat_axis = get_repeat_axis(other.grad.shape, out.grad.shape)
            # if self_shape_minus >=0:
            #    self.grad += out.grad # 注意：之所以这里用+=而不是=，是因为b=a+a里会引用两次a，梯度需要累加
            # else: # 需要进行sum
            #    self.grad += out.grad.sum(axis=self_axis, keepdims=False)
            accumulative_add_by_shape(self_not_repeat, self_repeat_axis, self.grad, out.grad)
            accumulative_add_by_shape(other_not_repeat, other_repeat_axis, other.grad, out.grad)

            # self.grad += out.grad # 注意：之所以这里用+=而不是=，是因为b=a+a里会引用两次a，梯度需要累加
            # other.grad += out.grad

        out._backward_grad_fn = _backward_grad
        return out

    # 逐元素相乘
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data * other.data, _prev_nodes=(self, other), _op='*', name=self.name + "*" + other.name)

        def _backward_grad():
            self_delta = other.data * out.grad
            other_delta = self.data * out.grad

            self_not_repeat, self_repeat_axis = get_repeat_axis(self.grad.shape, self_delta.shape)
            other_not_repeat, other_repeat_axis = get_repeat_axis(other.grad.shape, other_delta.shape)

            accumulative_add_by_shape(self_not_repeat, self_repeat_axis, self.grad, self_delta)
            accumulative_add_by_shape(other_not_repeat, other_repeat_axis, other.grad, other_delta)

            # self.grad += other.data * out.grad
            # other.grad += self.data * out.grad

        out._backward_grad_fn = _backward_grad

        return out

    # 矩阵相乘
    # X:[B,D], W:[D, C], Y: [B, C]
    # Y = X @ W
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        assert self.data.shape[-1] == other.data.shape[0], 'self data last dim should equals other data first dim'
        out = Tensor(self.data @ other.data, _prev_nodes=(self, other), _op='@', name=self.name + "@" + other.name)

        def _backward_grad():
            # self.grad: [B, D]
            # out.grad:[B, C]
            # other.data:[D, C]

            # dL/dX = dL/dY*dY/dx
            # dL/dY = out.grad
            # dy/dx= other.data
            # dL/dw = dL/dY * dY/dw
            # dy/dw = X
            self.grad += out.grad @ other.data.T  # 或np.dot

            # other.grad:[D, C]
            # out.grad:[B, C]
            # self.data:[B, D]
            other.grad += self.data.T @ out.grad

        out._backward_grad_fn = _backward_grad
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data ** other, _prev_nodes=(self,), _op=f'^{other}', name=f'{self.name}^{other}')

        """
        y = x**n
        dy/dx = n*(x**(n-1))
        """

        def _backward_grad():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward_grad_fn = _backward_grad

        return out

    def log(self):
        # y = log_e(x)
        # dy/dx = 1/x
        out = Tensor(np.log(self.data), _prev_nodes=(self,), _op='log', name=f'log({self.name})')

        def _backward_grad():
            self.grad += (self.data ** -1) * out.grad

        out._backward_grad_fn = _backward_grad
        return out

    def exp(self):
        # y = exp(x)
        # dy/dx = exp(x)
        out = Tensor(np.exp(self.data), _prev_nodes=(self,), _op='exp', name=f'exp({self.name})')

        def _backward_grad():
            self.grad += out.data * out.grad
        out._backward_grad_fn = _backward_grad
        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), _prev_nodes=(self,), _op='Mean')

        def _backward_grad():
            factor = np.prod(self.data.shape) / np.prod(out.data.shape)
            out_grad_expand = np.expand_dims(out.grad, axis=axis if axis else ())  # axis=()不作任何改变
            self.grad += (1 / factor) * out_grad_expand

        out._backward_grad_fn = _backward_grad
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), _prev_nodes=(self,), _op='Sum')

        def _backward_grad():
            out_grad_expand = np.expand_dims(out.grad, axis=axis if axis else ())
            self.grad += out_grad_expand
        out._backward_grad_fn = _backward_grad
        return out

    def abs(self):
        y_data = np.abs(self.data)
        y = Tensor(data=y_data, _prev_nodes=(self,), _op='Abs')

        # x>0: dl/dx=1
        # x<0: dL/dx=-1
        # x=0: dL/dx=0
        def _backward_grad():
            self.grad += (self.data>0)*1+(self.data<0)*(-1)
        y._backward_grad_fn = _backward_grad
        return y

    def relu(self):
        # old coding
        # out_data = self.data.copy()
        # out_data[out_data<0]=0
        # out = Tensor(out_data, _children=(self,), _op='ReLU')
        #
        y_data = F.Relu.forward(self.data)
        y = Tensor(data=y_data, _prev_nodes=(self,), _op='ReLU')

        def _backward_grad():
            self.grad += F.Relu.backward(x=self.data, y=y.data, y_grad=y.grad)

        y._backward_grad_fn = _backward_grad

        return y

    def identify(self):
        # out_data = self.data.copy()
        # out = Tensor(out_data, _children=(self,), _op='Identify')
        #
        # def _backward_grad():
        #     self.grad += out.grad
        # out._backward_grad_fn = _backward_grad
        y_data = F.Identity.forward(self.data)
        y = Tensor(y_data, _prev_nodes=(self,), _op='Identity')

        def _backward_grad():
            self.grad += F.Identity.backward(x=self.data, y=y.data, y_grad=y.grad)

        y._backward_grad_fn = _backward_grad

        return y

    def sigmoid(self):
        # y = sigmoid(x) = 1/(1+exp(-x))
        # dy/dx = y(1-y)
        # out = Tensor(1/(1+ np.exp(-self.data)), _children=(self,), _op='sigmoid')
        # def _backward_grad():
        #     self.grad += (out.data*(1-out.data)) * out.grad
        # out._backward_grad_fn = _backward_grad

        y_data = F.Sigmoid.forward(x=self.data)
        y = Tensor(y_data, _prev_nodes=(self,), _op='Sigmoid')

        def _backward_grad():
            self.grad += F.Sigmoid.backward(x=self.data, y=y.data, y_grad=y.grad)

        y._backward_grad_fn = _backward_grad

        return y

    def softmax(self, axis=None, batch_axis=0):
        prob = F.Softmax.forward(self.data, axis=axis)
        y = Tensor(data=prob, _prev_nodes=(self,), _op='Softmax')

        def _backward_grad():
            self.grad += F.Softmax.backward(x=self.data, y=y.data, y_grad=y.grad, batch_axis=batch_axis)
        y._backward_grad_fn = _backward_grad

        return y

    def log_softmax(self, axis=None, batch_axis=0):
        prob = F.LogSoftmax.forward(self.data, axis=axis)
        y = Tensor(data=prob, _prev_nodes=(self,), _op='log_softmax')

        def _backward_grad():
            self.grad += F.LogSoftmax.backward(x=self.data, y=y.data, y_grad=y.grad, batch_axis=batch_axis)
        y._backward_grad_fn = _backward_grad

        return y

    def tanh(self):
        # y = tanh(x)
        # dy/dx = 1-y*y
        # out = Tensor(np.tanh(self.data), _children=(self,), _op='tanh')
        # def _backward_grad():
        #     self.grad += (1-out.data*out.data) * out.grad
        #
        # out._backward_grad_fn = _backward_grad
        y_data = F.Tanh.forward(self.data)
        y = Tensor(y_data, _prev_nodes=(self,), _op='Tanh')

        def _backward_grad():
            self.grad += F.Tanh.backward(x=self.data, y=y.data, y_grad=y.grad)
        y._backward_grad_fn = _backward_grad

        return y

    # 选遍历获取所有子节点，然后再计算梯度
    def backward(self, grad=None):
        # topological order all of the children in the graph
        # 后序遍历获取所有子节点（只有最后一个节点需要这样遍历全部一次，其它的节点均不用遍历）
        topo_nodes = []
        visited = set()

        def build_nodes(v):
            # 当前节点没被访问(由于在图中，可能会有相同的子结点，因此需要添加访问标记)
            if v not in visited:
                visited.add(v)
                # 后序遍历(左右根)它的所有孩子
                for prev in v._prev:
                    build_nodes(prev)
                # 访问当前点，每个点只访问一次
                topo_nodes.append(v)

        build_nodes(self)

        # print("\n".join([str(t) for t in topo_nodes]))

        # go one variable at a time and apply the chain rule to get its gradient
        if grad is not None:
            self.grad = grad
        else:
            self.grad = np.ones_like(self.data)
        # 从后向前传播梯度
        for v in reversed(topo_nodes):
            # 每个节点都更新其前驱节点的梯度
            v._backward_grad_fn()

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other: Union[float, int]):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        assert isinstance(other, (int, float)), "only support int or float"
        return self * other

    def __truediv__(self, other):  # self / other
        if isinstance(other, (int)):
            return self * ((float(other)) ** -1)
        if isinstance(other, (np.int32, np.int64)):
            return self * (other.astype(np.float32) ** -1)
        else:
            return self * (other ** -1)

    def __rtruediv__(self, other):  # other / self
        return other * (self ** -1) # 因为已经实现过__pow__

    def __repr__(self):
        return f"Tensor(name:{self.name},data:{self.data},grad:{self.grad})"
