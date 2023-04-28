import torch
import os
import numpy as np
import random
from micrograd import nn
from tensorgrad.function import Functional as F

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from micrograd.engine import Value
from micrograd.graph_utils import *
import matplotlib.pyplot as plt

def test_sanity_check():

    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0],).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.data == ypt.data.item()
    # backward pass went well
    assert xmg.grad == xpt.grad.item()

def test_more_ops():

    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.retain_grad()
    g.backward()

    print("g grad:", g.grad)
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol

def test_graph2():
    # a very simple example
    a = Value(1.0, name='a')
    b = a *2; b.name ='b'
    c = b + 1; c.name ='c'
    d = c.relu(); d.name ='d'
    d.backward()
    g = draw_dot(d)
    g.view()

def test_graph():
    # a very simple example
    a = Value(1.0, name='a')
    b = Value(2.0, name='b')
    c = a * b; c.name ='c'
    e = c+Value(1, name='d'); e.name ='e'
    y = e.relu();y.name ='y'
    y.backward()
    g = draw_dot(y)
    g.view()

def test_nn():
    # a simple 2D neuron
    import random
    from micrograd import nn

    random.seed(1337)
    n = nn.Neuron(2)
    x = [Value(1.0), Value(-2.0)]
    y = n(x)
    print(n)
    y.backward()
    #g = draw_dot(y)
    #g.view()

def test_mlp():
    # a simple 2D neuron

    random.seed(1337)
    net = nn.MLP(n_input=2, out_size_of_each_layer=[3, 4, 1])
    x = [Value(1.0), Value(-2.0)]
    y = net(x)
    print(net)
    y.backward()
    g = draw_dot(y)
    g.view()

def test_demo_data_mlp():
    np.random.seed(1337)
    random.seed(1337)
    # make up a dataset
    from sklearn.datasets import make_moons, make_blobs
    # X:[100, 2]
    # y:[100]
    X, y = make_moons(n_samples=100, noise=0.1)

    y = y * 2 - 1  # make y be -1 or 1
    # visualize in 2D
    plt.figure(figsize=(5, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap='jet')
    #plt.show()

    # initialize a model
    model = nn.MLP(n_input=2, out_size_of_each_layer=[16, 16, 1])  # 2-layer neural network
    print(model)
    print("number of parameters:", len(model.parameters()))

    # loss function
    def loss(batch_size=None):
        # inline DataLoader :)
        if batch_size is None:
            Xb, yb = X, y
        else:
            ri = np.random.permutation(X.shape[0])[:batch_size] # 随便取一个排列，取前batch_size个元素
            Xb, yb = X[ri], y[ri]
        # 生成input values
        inputs = [list(map(Value, xrow)) for xrow in Xb]

        # forward the model to get scores
        scores = list(map(model, inputs))

        # svm "max-margin" loss
        # loss = relu(1 - yi*score_i), where y in{-1,1}
        losses = [(1 - yi * scorei).relu() for yi, scorei in zip(yb, scores)]
        data_loss = sum(losses) / len(losses)
        # L2 regularization
        alpha = 1e-4
        reg_loss = alpha * sum((p * p for p in model.parameters()))
        total_loss = data_loss + reg_loss

        # also get accuracy
        accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)]
        return total_loss, sum(accuracy) / len(accuracy)

    # loss
    total_loss, acc = loss()
    print("loss:{} acc:{}".format(total_loss.data, acc))

    # optimization
    for k in range(100):
        # forward
        total_loss, acc = loss()

        # backward
        model.zero_grad()
        total_loss.backward()

        # update (sgd)
        learning_rate = 1.0 - 0.9 * k / 100 # k 越大，lr越小
        for p in model.parameters():
            p.data -= learning_rate * p.grad

        if k % 10 == 0:
            print(f"step:{k:03d} loss:{total_loss.data:.6f}, accuracy:{acc * 100:.1f}%")

    # visualize decision boundary
    h = 0.25
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Xmesh = np.c_[xx.ravel(), yy.ravel()]
    inputs = [list(map(Value, xrow)) for xrow in Xmesh]
    scores = list(map(model, inputs)) # 输出预测的分数
    Z = np.array([s.data > 0 for s in scores]) # >0为正样本，<0为负样本
    Z = Z.reshape(xx.shape)

    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

def test_tanh():
    x = np.arange(-10, 10, 0.001)
    sigmoid = 1/(1+np.exp(-x))
    dsigmoid = sigmoid*(1-sigmoid)
    y = np.tanh(x)
    dydx = 1-y**2
    plt.plot(x,y)
    plt.plot(x,dydx)
    plt.plot(x, sigmoid)
    plt.plot(x, dsigmoid)
    plt.legend(['tanh', 'dtanh','sigmoid', 'dsigmoid'])
    plt.show()

if __name__ == "__main__":
    #test_graph2()
    #test_graph()
    test_more_ops()
    #test_nn()
    #test_mlp()
    #test_sanity_check()
    #test_demo_data_mlp()
    #test_tanh()
