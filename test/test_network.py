import numpy as np
import sys

from sklearn.decomposition import PCA

sys.path.append("../")

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import random
from sklearn.datasets import make_moons, make_blobs, make_multilabel_classification, load_iris
from tensorgrad.network import *
from micrograd.graph_utils import *
import matplotlib.pyplot as plt
import torch

"""
先要安装graphviz
sudo apt-get install graphviz
"""
def test_simple_mlp():
    Xb = np.array([[0.2, 0.3],
                   [-0.4, 0.8],
                   [-0.3, 0.9],
                   [0.5, 0.3]
                   ])
    yb = np.array([1, -1, -1, 1])

    model = MLP(n_input=2,
                out_size_of_each_layer=[1],
                activate_func=F.Tanh(),
                bias=True)  # 2-layer neural network
    print(model)
    print("number of parameters:", len(model.parameters()))

    # 生成input values
    inputs = Tensor(Xb, name='x')
    model_pred = model(inputs)
    model_pred.name = 'model_pred'
    scores = model_pred.reshape(-1)
    scores.name = 'scores'

    diff = scores - yb
    losses = diff ** 2
    losses.name = 'loss'
    mean_loss = losses.mean()
    mean_loss.name = 'mean_loss'

    model.zero_grad()
    mean_loss.backward()
    print("mean_loss:", mean_loss)


def test_demo_data_mlp():
    np.random.seed(1337)
    random.seed(1337)
    # X:[100, 2]
    # y:[100]
    X, y = make_moons(n_samples=100, noise=0.1)

    y = y * 2 - 1  # make y be -1 or 1
    # visualize in 2D
    plt.figure(figsize=(5, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap='jet')
    # plt.show()
    print("X:", X.shape)
    print("y:", y.shape)

    # initialize a model
    model = MLP(n_input=2,
                out_size_of_each_layer=[16, 16, 1],
                activate_func=F.Tanh(),
                bias=True)  # 2-layer neural network
    print(model)
    print("number of parameters:", len(model.parameters()))
    print("total parameters:", model.calculate_total_param_num())

    # loss function
    def calculate_loss(batch_size=None):
        # inline DataLoader :)
        if batch_size is None:
            Xb, yb = X, y
        else:
            ri = np.random.permutation(X.shape[0])[:batch_size]  # 随便取一个排列，取前batch_size个元素
            Xb, yb = X[ri], y[ri]
        # 生成input values
        inputs = Tensor(Xb, name='x')

        # forward the model to get scores
        model_pred = model(inputs)
        model_pred.name = 'model_pred'
        scores = model_pred.reshape(-1)
        scores.name = 'scores'

        # svm "max-margin" loss
        # loss = relu(1 - yi*score_i), where y in{-1,1}
        # losses = [(1 - yi * scorei).relu() for yi, scorei in zip(yb, scores)]
        yb_tensor = Tensor(yb, dtype=np.float32, name='yb')

        losses = (1 - yb_tensor * scores).relu()
        mean_loss = losses.mean()
        mean_loss.name = 'mean_loss'

        # L2 regularization
        alpha = 1e-4
        reg_loss = 0
        for p in model.parameters():
            reg_loss += (p * p).sum()

        total_loss = mean_loss + reg_loss * alpha

        # also get accuracy
        score = scores.numpy()

        is_pos = score > 0
        score[is_pos] = 1
        score[~is_pos] = -1
        accuracy = (score == yb).mean()
        return total_loss, accuracy

    # loss
    total_loss, acc = calculate_loss()
    print("loss:{} acc:{}".format(total_loss.data, acc))

    # 打印tensor依赖图
    print("draw tensor依赖图")
    L_dot = draw_dot_tensor(total_loss, show_data=False)
    L_dot.view(filename="net.svg")

    # optimization
    for k in range(100):
        # forward
        total_loss, acc = calculate_loss()

        # backward, 计算梯度
        model.zero_grad()
        total_loss.backward()

        # update (sgd), 更新模型参数, 类似于pytorch的optimizer.step()
        # 学习率随着迭代次数增加而减小
        learning_rate = 1.0 - 0.9 * k / 100  # k 越大，lr越小
        for p in model.parameters():
            p.data -= learning_rate * p.grad

        if k % 10 == 0:
            print(f"step:{k:03d} loss:{total_loss.data:.6f}, accuracy:{acc * 100:.1f}%")

    print("train done!")
    # visualize decision boundary
    h = 0.25
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Xmesh = np.c_[xx.ravel(), yy.ravel()]

    inputs = Tensor(Xmesh)
    scores = model(inputs).numpy()  # 输出预测的分数
    Z = scores > 0  # >0为正样本，<0为负样本
    Z = Z.reshape(xx.shape)

    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()
    plt.savefig("demo.png")


def test_mlp_sigmoid_with_cross_entropy_loss():
    batch_size = 10
    N = 100
    iter = 300
    alpha = 1e-4

    class_num = 2
    input_dim = 2

    np.random.seed(1337)
    random.seed(1337)
    X, Y = make_moons(n_samples=100, noise=0.1)
    #X, Y = make_blobs(n_samples=N, centers=class_num, n_features=input_dim, random_state=0)
    # X, Y = make_multilabel_classification(n_samples=N, n_classes=class_num, n_labels=1, n_features=input_dim, random_state = 0)
    print("X:", X.shape, X[:5,:])
    print("y:", Y.shape, Y[:5])
    # Y in {0,1}
    # visualize in 2D
    #plt.figure(figsize=(5, 5))
    #plt.scatter(X[:, 0], X[:, 1], c=Y, s=20, cmap='jet')
    #plt.show()

    # initialize a model
    model = MLP(n_input=2,
                out_size_of_each_layer=[16, 16,  1],
                activate_func=F.Tanh(),
                bias=True)  # 2-layer neural network
    loss_layer = SigmoidCrossEntropyWithLogitLossLayer()

    print(model)
    print("number of parameters:", len(model.parameters()))
    print("total parameters:", model.calculate_total_param_num())

    # optimization
    for k in range(iter):
        ri = np.random.permutation(N)[:batch_size]  # 随便取一个排列，取前batch_size个元素
        Xb, Yb = Tensor(X[ri], name='x'), Tensor(Y[ri].reshape(-1,1), name='y')
        # data loss
        model_out = model(Xb)
        data_loss, pred = loss_layer(model_out, Yb)

        #reg loss
        reg_loss = 1e-4
        for p in model.parameters():
            reg_loss += (p * p).sum()

        total_loss = data_loss.mean() + reg_loss * alpha
        #total_loss = data_loss.mean()

        # backward
        model.zero_grad()
        total_loss.backward()

        if k==0:
            # 打印tensor依赖图
            print("draw tensor依赖图")
            L_dot = draw_dot_tensor(total_loss, show_data=False)
            L_dot.view()

        # def print_debug_info():
        #     print(f"\n\nstep:{k:03d} xb:{Xb.data} yb:{Yb.data} pred:{pred.data} loss:{total_loss.data:.6f}")
        #     for p in model.parameters():
        #         print(f"name:{p.name} data:{p.data} grad:{p.grad}", end=' ')
        #
        #print_debug_info()

        # update (sgd), 更新模型参数, 类似于pytorch的optimizer.step()
        # 学习率随着迭代次数增加而减小
        learning_rate = 1.0 - 0.9 * k / iter  # k 越大，lr越小
        #learning_rate = 1E-4
        for p in model.parameters():
            p.data -= learning_rate * p.grad

        # also get accuracy
        score = pred.numpy()
        #print("score:", score, " y:",Yb.data)
        is_pos = score >= 0.5
        score[is_pos] = 1
        score[~is_pos] = 0
        accuracy = np.mean(score == Yb.numpy())

        hint = True
        if hint:
            if k % 100 == 0:
                print(f"step:{k:03d} loss:{total_loss.data:.6f}, accuracy:{accuracy * 100:.1f}%")

    print("train done!")
    # visualize decision boundary
    h = 0.25
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Xmesh = np.c_[xx.ravel(), yy.ravel()]

    inputs = Tensor(Xmesh)
    scores = F.Sigmoid.forward(model(inputs).data)  # 输出预测的分数
    Z = scores > 0.5  # >0为正样本，<0为负样本
    Z = Z.reshape(xx.shape)

    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

def test_mlp_softmax_with_cross_entropy_loss():
    N = 1000
    batch_size = 20
    iter = 10000

    class_num = 4
    input_dim = 2

    np.random.seed(1337)
    random.seed(1337)
    X, Y = make_blobs(n_samples=N, centers=class_num, n_features=input_dim, random_state=1)
    print("X:", X.shape, X[:5,:])
    print("y:", Y.shape, Y[:5])
    # Y in {0,1}

    # visualize in 2D
    #plt.figure(figsize=(5, 5))
    #plt.scatter(X[:, 0], X[:, 1], c=Y, s=20, cmap='jet')
    #plt.show(); return

    # initialize a model
    model = MLP(n_input=input_dim,
                out_size_of_each_layer=[16, 32, 8, class_num],
                activate_func=F.Tanh(),
                bias=True)  # 2-layer neural network
    loss_layer = SoftmaxCrossEntropyWithLogitLossLayer(reduction='mean')

    print(model)
    print("number of parameters:", len(model.parameters()))
    print("total parameters:", model.calculate_total_param_num())

    alpha = 1e-4
    # optimization
    for k in range(iter):
        ri = np.random.permutation(N)[:batch_size]  # 随便取一个排列，取前batch_size个元素
        Xb, Yb = Tensor(X[ri], name='x'), Tensor(Y[ri], name='y')

        def get_loss_pred(x_tensor, y_tensor):
            # data loss
            model_out = model(x_tensor)
            data_loss, pred = loss_layer(model_out, y_tensor)

            # L2 - reg loss
            reg_loss = 1e-4
            for p in model.parameters():
                reg_loss += (p * p).sum()

            total_loss = data_loss.mean() + reg_loss * alpha
            return total_loss, pred

        total_loss, pred = get_loss_pred(Xb, Yb)
        # backward
        model.zero_grad()
        total_loss.backward()

        if k==0:
            # 打印tensor依赖图
            print("draw tensor依赖图")
            L_dot = draw_dot_tensor(total_loss, show_data=True)
            L_dot.view()

        # update (sgd)
        #learning_rate = 1.0 - 0.9 * k / iter  # k 越大，lr越小
        learning_rate = 1E-4
        for p in model.parameters():
            p.data -= learning_rate * p.grad

        hint = True
        if hint:
            if k % 100 == 0:
                #data_loss, pred = loss_layer(model(Tensor(X)), Tensor(Y))
                total_loss, pred = get_loss_pred(Tensor(X), Tensor(Y))
                # also get all accuracy
                label_index = np.argmax(pred.numpy(), axis=1).flatten()
                # print("score:", score, " y:",Yb.data)
                accuracy = np.mean(label_index == Y)
                print(f"step:{k:03d} all data loss:{total_loss.data:.6f}, accuracy:{accuracy * 100:.1f}%")

    print("train done!")
    # visualize decision boundary
    h = 0.25
    # x 轴与y轴, 只有样本维度为2时才可以
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Xmesh = np.c_[xx.ravel(), yy.ravel()]

    inputs = Tensor(Xmesh)
    scores = F.Sigmoid.forward(model(inputs).data)  # 输出预测的分数
    Z = scores.argmax(axis=1)
    Z = Z.reshape(xx.shape)

    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

def test_iris_data_with_mlp_softmax_with_cross_entropy_loss():
    batch_size = 20
    iter = 2000

    class_num = 3
    input_dim = 4

    np.random.seed(1337)
    random.seed(1337)

    iris = load_iris()
    X = iris.data
    y = iris.target

    # 标准化输入
    scaler = StandardScaler()  # 对每个特征，减去均值，除以标准差
    X = scaler.fit_transform(X)

    use_pca = True
    if use_pca: # 使用PCA降维
        input_dim=2
        pca = PCA(n_components=input_dim)
        X = pca.fit_transform(X)
        # PCA降维
        print("降维后形状:", X.shape)
        print("方差解释比例:", pca.explained_variance_ratio_)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        print("累计方差解释比例:", cumulative_variance)

    # 划分训练集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    N = X_train.shape[0]
    print("X:", X.shape, X[:5,:])
    print("y:", y.shape, y[:5])
    # Y in {0,1}

    # visualize in 2D
    #plt.figure(figsize=(5, 5))
    #plt.scatter(X[:, 0], X[:, 1], c=Y, s=20, cmap='jet')
    #plt.show(); return

    # initialize a model
    model = MLP(n_input=input_dim,
                out_size_of_each_layer=[16, 16, 16, 8, class_num],
                activate_func=F.Relu(),
                bias=True)  # 2-layer neural network
    loss_layer = SoftmaxCrossEntropyWithLogitLossLayer(reduction='mean')

    print(model)
    print("number of parameters:", len(model.parameters()))
    print("total parameters:", model.calculate_total_param_num())

    alpha = 1e-4
    # optimization
    for k in range(iter):
        ri = np.random.permutation(N)[:batch_size]  # 随便取一个排列，取前batch_size个元素
        Xb, Yb = Tensor(X_train[ri], name='x'), Tensor(y_train[ri], name='y')

        def get_loss_pred(x_tensor, y_tensor):
            # data loss
            model_out = model(x_tensor)
            data_loss, pred = loss_layer(model_out, y_tensor)

            # L2 - reg loss
            reg_loss = 1e-4
            for p in model.parameters():
                reg_loss += (p * p).sum()

            total_loss = data_loss.mean() + reg_loss * alpha
            return total_loss, pred

        total_loss, pred = get_loss_pred(Xb, Yb)
        # backward
        model.zero_grad()
        total_loss.backward()

        if k==0:
            # 打印tensor依赖图
            print("draw tensor依赖图")
            L_dot = draw_dot_tensor(total_loss, show_data=True)
            #L_dot.view()

        # update (sgd)
        #learning_rate = 1.0 - 0.9 * k / iter  # k 越大，lr越小
        learning_rate = 1E-4
        for p in model.parameters():
            p.data -= learning_rate * p.grad

        hint = True
        if hint:
            if k % 100 == 0:
                total_loss, test_pred = get_loss_pred(Tensor(X_test), Tensor(y_test))
                # also get all accuracy
                label_index = np.argmax(test_pred.numpy(), axis=1).flatten()
                # print("score:", score, " y:",Yb.data)
                accuracy = np.mean(label_index == y_test)
                print(f"step:{k:03d} all data loss:{total_loss.data:.6f}, test accuracy:{accuracy * 100:.1f}%")

    print("train done!")

    if use_pca:
        # visualize decision boundary
        h = 0.25
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h),
                             )
        Xmesh = np.c_[xx.ravel(), yy.ravel()]

        inputs = Tensor(Xmesh)
        scores = F.Sigmoid.forward(model(inputs).data)  # 输出预测的分数
        Z = scores.argmax(axis=1)
        Z = Z.reshape(xx.shape)

        fig = plt.figure()
        # 更好的方式是利用pca/t-snets等降维方法，将数据降维到2维，然后画图
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.CMRmap)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.show()

def test_mlp_with_mse_loss():
    batch_size = 1
    N = 100
    iter = 20000
    alpha = 1e-4

    class_num = 2
    input_dim = 2

    np.random.seed(1337)
    random.seed(1337)
    #X, Y = make_moons(n_samples=100, noise=0.1)
    X, Y = make_blobs(n_samples=N, centers=class_num, n_features=input_dim, random_state=0)
    # X, Y = make_multilabel_classification(n_samples=N, n_classes=class_num, n_labels=1, n_features=input_dim, random_state = 0)
    print("X:", X.shape, X[:5,:])
    print("y:", Y.shape, Y[:5])

    # Y in {0,1}
    # visualize in 2D
    #plt.figure(figsize=(5, 5))
    #plt.scatter(X[:, 0], X[:, 1], c=Y, s=20, cmap='jet')
    #plt.show()

    # initialize a model
    model = MLP(n_input=input_dim,
                out_size_of_each_layer=[16, 16, 1],
                activate_func=F.Tanh(),
                bias=True)  # 2-layer neural network
    loss_layer = MeanSquareErrorLossLayer()

    print(model)
    print("number of parameters:", len(model.parameters()))
    print("total parameters:", model.calculate_total_param_num())

    # optimization
    for k in range(iter):
        ri = np.random.permutation(N)[:batch_size]  # 随便取一个排列，取前batch_size个元素
        Xb, Yb = Tensor(X[ri], name='x'), Tensor(Y[ri].reshape(-1,1), name='y')

        # data loss
        def get_loss_pred(x_tensor, y_tensor):
            # data loss
            pred = model(x_tensor).sigmoid()
            data_loss = loss_layer(pred, y_tensor)

            # reg loss
            reg_loss = 0
            for p in model.parameters():
                w_sum = (p * p).sum()
                reg_loss += w_sum/np.prod(p.shape)

            total_loss = data_loss.mean() + reg_loss * alpha
            return total_loss, pred

        total_loss, pred = get_loss_pred(Xb, Yb)

        # backward
        model.zero_grad()
        total_loss.backward()

        if k==0:
            # 打印tensor依赖图
            print("draw tensor依赖图")
            L_dot = draw_dot_tensor(total_loss, show_data=True)
            L_dot.view()

        # update (sgd)
        #learning_rate = 1.0 - 0.9 * k / iter  # k 越大，lr越小
        learning_rate = 5E-4
        for p in model.parameters():
            p.data -= learning_rate * p.grad

        hint = True
        if hint:
            if k % 100 == 0:
                #data_loss, pred = loss_layer(model(Tensor(X)), Tensor(Y))
                total_loss, pred = get_loss_pred(Tensor(X), Tensor(Y.reshape(-1,1)))
                # also get accuracy
                score = pred.numpy()
                # print("score:", score, " y:",Yb.data)
                is_pos = score >= 0.5
                score[is_pos] = 1
                score[~is_pos] = 0
                accuracy = np.mean(score.flatten() == Y)
                print(f"step:{k:03d} loss:{total_loss.data:.6f}, accuracy:{accuracy * 100:.1f}%")

    print("train done!")
    # visualize decision boundary
    h = 0.25
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Xmesh = np.c_[xx.ravel(), yy.ravel()]

    inputs = Tensor(Xmesh)
    scores = model(inputs).sigmoid().data  # 输出预测的分数
    Z = scores > 0.5  # >0为正样本，<0为负样本
    Z = Z.reshape(xx.shape)

    fig = plt.figure()
    # 画出等高线
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()


if __name__ == "__main__":
    test_iris_data_with_mlp_softmax_with_cross_entropy_loss()

    if False:
        test_mlp_softmax_with_cross_entropy_loss()
        test_demo_data_mlp()
        test_mlp_sigmoid_with_cross_entropy_loss()
        test_mlp_with_mse_loss()
        test_simple_mlp()
