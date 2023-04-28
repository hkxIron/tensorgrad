import random
from typing import List
from micrograd.engine import Value

# 基类
class Module:
    # 每次迭代时，都需要将梯度清空
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    # 返回所有的参数
    def parameters(self):
        return []

# 单个神经元
class Neuron(Module):
    def __init__(self, n_input:int, non_linear=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(n_input)]
        self.b = Value(0)
        self.non_linear = non_linear

    def __call__(self, x):
        # act = wx + b
        activation = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return activation.relu() if self.non_linear else activation

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.non_linear else 'Linear'} Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, n_input:int, n_output:int, **kwargs):
        self.neurons = [Neuron(n_input, **kwargs) for _ in range(n_output)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons
                    for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, n_input:int, out_size_of_each_layer:List[int]):
        size_list = [n_input] + out_size_of_each_layer
        self.layers = [Layer(n_input=size_list[layer_index],
                             n_output=size_list[layer_index + 1],
                             non_linear=layer_index != len(out_size_of_each_layer) - 1)  # 即最后一层没有激活函数
                       for layer_index in range(len(out_size_of_each_layer))]

    def __call__(self, x):
        # forward pass
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers
                    for p in layer.parameters()]

    def __repr__(self):
        #return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
        return "\n".join(["layer:{} non_linear:{} params:[{},{}]".format(i, layer.neurons[0].non_linear, len(layer.neurons[0].w), len(layer.neurons))
                                for i, layer in enumerate(self.layers)])
