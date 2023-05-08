from typing import List
from tensorgrad.tensor import *
import numpy as np

# 基类
class Module:
    # 每次迭代时，都需要将梯度清空
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    # 返回所有的参数
    def parameters(self)->List[Tensor]:
        return []

class LinearLayer(Module):
    def __init__(self, n_input:int,
                 n_output:int,
                 bias=False,
                 activate_func:F.Func = None,
                 layer_index=None):
        self.weight = Tensor(np.random.uniform(low=-1, high=1, size=(n_input, n_output)),
                             name="L"+str(layer_index)+"_w" if layer_index is not None else "w")
        self.bias = Tensor(np.zeros(n_output),
                           name="L"+str(layer_index) +'_b' if layer_index is not None else 'b') if bias else None
        self.act_func = activate_func
        self.layer_index = layer_index

    def __call__(self, x:Tensor)->Tensor:
        xw = x @ self.weight
        xw.name = "L"+str(self.layer_index) +'_w@x'

        xw_b = xw
        if self.bias:
            xw_b = xw + self.bias
            xw_b.name = "L"+str(self.layer_index) +'_w@x+b'

        xw_b_act = xw_b
        if self.act_func:
            out2_data = self.act_func.forward(xw_b.data)
            xw_b_act = Tensor(out2_data, _prev_nodes=(xw_b,), name="L"+str(self.layer_index)+'_activate')

            # 如果有激活函数，那么需要显式定义梯度的回传,如果没有激活函数，则依靠xw+b中tensor的自动微分机制即可
            def _backward_grad():
                xw_b.grad += self.act_func.backward(x=xw_b.data, y=out2_data, y_grad=xw_b_act.grad)
            xw_b_act.set_backward_func(_backward_grad)


        return xw_b_act

    def parameters(self)->List[Tensor]:
        return [self.weight, self.bias] if self.bias else [self.weight]

    def __repr__(self):
        return "Layer:{} w:{} b:{}".format(self.layer_index, self.weight.shape, self.bias.shape)

class MLP(Module):
    def __init__(self,
                 n_input:int,
                 out_size_of_each_layer:List[int],
                 bias=False,
                 activate_func:F.Func=None):
        size_list = [n_input] + out_size_of_each_layer
        self.layers = [LinearLayer(n_input=size_list[layer_index],
                                   n_output=size_list[layer_index + 1],
                                   bias=bias,
                                   layer_index=layer_index,
                                   activate_func = activate_func if layer_index != len(out_size_of_each_layer) - 1 else None)  # 即最后一层没有激活函数
                       for layer_index in range(len(out_size_of_each_layer))]

    def __call__(self, x:Tensor)->Tensor:
        # forward pass
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self)->List[Tensor]:
        return [p for layer in self.layers
                    for p in layer.parameters()]

    def calculate_total_param_num(self)->int:
        return sum([np.prod(p.data.shape) for p in self.parameters()])

    def __repr__(self):
        return "\n".join(["layer:{} params:[w:{} b:{}] act:{}".format(i,
                                                                      layer.weight.shape,
                                                                      layer.bias.shape if layer.bias else None,
                                                                      layer.act_func.__class__.__name__) for i, layer in enumerate(self.layers)])

# loss layers
class SigmoidCrossEntropyWithLogitLossLayer(Module):
    def __init__(self):
        self.func = F.SigmoidCrossEntropyWithLogitLoss()

    # loss = -(y*log(p) + (1-y)*log(1-p))
    def __call__(self, x: Tensor, y: Tensor) -> (Tensor, Tensor):
        # y in {0, 1}
        y_pred = F.Sigmoid.forward(x.data)
        loss = self.func.forward(x=x.data, y=y.data)
        loss_tensor = Tensor(data=loss, _prev_nodes=(x, y), name='SigmoidCrossEntropyWithLogitLoss')

        def _backward_grad():
            x.grad += self.func.backward(y_pred=y_pred, y=y.data) * loss_tensor.grad

        loss_tensor.set_backward_func(_backward_grad)
        y_pred_tensor = Tensor(y_pred, name='pred', _prev_nodes=(x,))
        return loss_tensor, y_pred_tensor

class SoftmaxCrossEntropyWithLogitLossLayer(Module):
    def __init__(self, reduction='sum', axis=1):
        """
        axis:which axis to calculate softmax probility
        """
        self.func = F.SoftmaxCrossEntropyWithLogitLoss()
        self.reduction= reduction
        self.axis = axis

    # sum_k{y_k*log(p_k)}
    def __call__(self, x: Tensor, y: Tensor) -> (Tensor,Tensor):
        # y in {0, 1}
        loss = self.func.forward(x=x.data, y=y.data, axis=self.axis)
        if self.reduction == 'sum':
            loss = np.sum(loss)
        elif self.reduction == 'mean':
            loss = np.mean(loss)

        loss_tensor = Tensor(data=loss, _prev_nodes=(x, y), name='SoftmaxCrossEntropyWithLogitLossLayer')
        y_pred = F.Softmax().forward(x.data, axis=self.axis)
        def _backward_grad():
            # y没有loss
            x.grad += self.func.backward(y_pred=y_pred, y_label_index=y.data) * loss_tensor.grad

        loss_tensor.set_backward_func(_backward_grad)
        y_pred_tensor = Tensor(y_pred, name='pred', _prev_nodes=(x,))
        return loss_tensor, y_pred_tensor

class MeanSquareErrorLossLayer(Module):
    def __init__(self):
        pass

    # loss = sum_i{ (yi-pi)**2 }
    def __call__(self, y_pred: Tensor, y: Tensor) -> (Tensor, Tensor):
        # y in {0, 1}
        assert len(y_pred.shape)==2
        assert y_pred.shape == y.shape
        N = y_pred.data.shape[0]
        loss = np.sum((y_pred.data - y.data) ** 2)/N

        loss_tensor = Tensor(data=loss, _prev_nodes=(y_pred, y), name='MeanSquareLoss')

        def _backward_grad():
            y_pred.grad += 2/N*(y_pred.data - y.data) * loss_tensor.grad
        loss_tensor.set_backward_func(_backward_grad)

        return loss_tensor
