import math
import numpy as np

class Functional:
    class Func:
        @staticmethod
        def forward(x: np.ndarray, **kwargs) -> np.ndarray:
            pass

        @staticmethod
        def backward(x: np.ndarray, y: np.ndarray, y_grad: np.ndarray, **kwargs) -> np.ndarray:
            pass

    class Relu(Func):
        @staticmethod
        def forward(x: np.ndarray, **kwargs) -> np.ndarray:
            # y = x if x>0 else 0
            out_data = x.copy()
            out_data[out_data < 0] = 0
            return out_data

        @staticmethod
        def backward(x: np.ndarray, y: np.ndarray, y_grad: np.ndarray, **kwargs) -> np.ndarray:
            # y = relu(x)
            grad = (x > 0) * y_grad  # <0的地方直接为0
            return grad

    # y = sigmoid(x) = 1/(1+exp(-x))
    # dy/dx = y(1-y)
    class Sigmoid(Func):
        @staticmethod
        def forward(x: np.ndarray, **kwargs) -> np.ndarray:
            """
            y = 1/(1+exp(-x))

            if x>0
                exp(-x)接近于0, 因此计算是稳定的，不会产生上溢
                等价于：1/(1+exp(-abs(x)))
            else if x<0
                1/(1+exp(-x)) = 1- 1/(1+exp(x)) = 1- 1/(1+exp(-abs(x)))
                其中exp(x)接近于0，因此计算也是稳定的
            因此需要计算: 1/(1+exp(-abs(x)))

            :param x:
            :param kwargs:
            :return:
            """
            # origin: 1/(1+exp(-x))
            # numberic stable version
            is_pos = x>0
            a = 1 / (1 + np.exp(-np.abs(x)))
            return is_pos*a + (~is_pos)*(1 - a)

        @staticmethod
        def backward(x: np.ndarray, y: np.ndarray, y_grad: np.ndarray, **kwargs) -> np.ndarray:
            # x_out=Sigmoid(x)
            # x.grad += (x_out.data*(1-x_out.data)) * x_out.grad
            return y * (1 - y) * y_grad

    # y = softmax(x)
    # yi = exp(xi)/sum(exp(xj))
    # A = max(xj)
    # yi = exp(xi-A)/sum(exp(xj-A))
    #
    # if i=j:
    #   dy_i/dx_i = y_i*(1-y_j) = y*i*(1-y_i) = yi-yi*yi
    # else:
    #   dy_i/dx_j = -y_i*y_j
    class Softmax(Func):
        @staticmethod
        def forward(x: np.ndarray, axis=1, **kwargs) -> np.ndarray:
            # axis: which axis to calculate softmax
            # inds = x.data.argmax(axis=axis)
            keepdims = True
            x_max = x.max(axis=axis, keepdims=keepdims)
            exp_x = np.exp(x - x_max)
            prob = exp_x / np.sum(exp_x, axis=axis, keepdims=keepdims)
            return prob

        # x:[N, C]
        # grad:[N,C]
        @staticmethod
        def backward(x: np.ndarray, y: np.ndarray, y_grad: np.ndarray, batch_axis=0, **kwargs) -> np.ndarray:
            """
             对于一个样本，假设x维度为3,则预测的类别数也为3, 则x=[x1,x2,x3], y=[y1,y2, y3]
             [y1,y2,y3] = softmax([x1,x2,x3])

            即
             x1-> exp(x1) -> y1=exp(x1)/sum(exp(xj))
             x2-> exp(x2) -> y2=exp(x2)/sum(exp(xj))
             x3-> exp(x3) -> y3=exp(x3)/sum(exp(xj))

            根据公式有：
            y = softmax(x)
            yi = exp(xi)/sum(exp(xk))
            if i=j:
              dy_i/dx_j = y_i*(1-y_j) = y*i*(1-y_i) = yi-yi*yi
            else:
              dy_i/dx_j = -y_i*y_j

            =>
            而下一个节点的梯度已知，为:
            dL/dy = [dL/dy1, dL/dy2, dL/dy3], 即为代码中的y_grad

            同时由前向传播可知
            x1-> exp(x1) -> y1=exp(x1)/sum(exp(xj)) -> loss
            x2-> exp(x2) -> y2=exp(x2)/sum(exp(xj)) -> loss
            x3-> exp(x3) -> y3=exp(x3)/sum(exp(xj)) -> loss

            即x1可通过y1,y2,y3影响最终loss L:
            x1->y1 -> L
            x1->y2 -> L
            x1->y3 -> L

            因此:
            dL/dx = [dL/dx1, dL/dx2, dL/dx3]

            其中L对dxi的梯度为:
            dL/dxi = sum{k}{ dL/dyk*(dyk/dxi) }
            = dL/dy1*dy1/dxi + dL/dy2*dy2/dxi + dL/dy3*dy3/dxi
            = [dy1/dxi dy2/dxi dy3/dxi]* [dL/dy1 dL/dy2 dL/dy3].T
            = dy/dx * (dL/dy).T
            = softmax_grad * y_grad.T

            dL/dy = [dL/dy1 dL/dy2 dL/dy3]

            dy/dx= softmax_grad梯度矩阵为:
            [dyi/dxj] = [
                [dyi/dx1],
                [dyi/dx2],
                [dyi/dx3]
            ]
            =>
             [
                dy1/dx1 dy1/dx2 dy1/dx3
                dy2/dx1 dy2/dx2 dy2/dx3
                dy3/dx1 dy3/dx2 dy3/dx3
             ]

             =>
             [
                y1-y1*y1 -y1*y2   -y1*y3
                -y2*y1   y2-y2*y2 -y2*y3
                -y3*y1   -y3*y2   y3-y2*y3
             ]

            可拆分为两个矩阵相减 =>
            [
                y1
                    y2
                        y3
            ] - [
                y1*y1 y1*y2 y1*y3
                y2*y1 y2*y2 y2*y3
                y3*y1 y3*y2 y3*y3
            ]

            注意：
            softmax矩阵为对称矩阵,每行与每列之和为0，对于其中的某一行或者某一列的物理意义是:
            当dL/dy为全1向量时，以xi=x1为例,可计算dL/dxi的梯度：
            dL/dx1 =dL/dy*dy/dx1
                   = y1-y1*y1 -y1*y2 -y1*y3
                   = y1-y1(y1+y2+y3) = 0,
            即dL/dy为全1向量时，dL/dxi=0
            而在实际训练数据中，由于y一般为one-hot向量，其中只有一个元素为1，其它均为0，因此，不会出现dL/dxi为0的情况,所有的梯度均有值

            =>
            对应到python代码中，
            x, shape[N,C]
            y, shape:[N, C]
            i为batch中第i个样本

            diag[yi] = np.diag(y[i]) =
            [
                y1
                    y2
                        y3
            ]

            [ yi*yj ] = np.outer(y[i], y[i])  =
            [
                y1*y1 y1*y2 y1*y3
                y2*y1 y2*y2 y2*y3
                y3*y1 y3*y2 y3*y3
            ]
            因此:
            dy/dx = softmax_grad = np.diag(y[i]) - np.outer(y[i], y[i])
            = [
                y1-y1*y1 -y1*y2   -y1*y3
                -y2*y1   y2-y2*y2 -y2*y3
                -y3*y1   -y3*y2   y3-y2*y3
             ]
            """
            batch_size = y.shape[batch_axis]
            # 对于batch中的每个样本进行处理
            batch_grads = []
            for i in range(batch_size): # 对于第i个样本
                yi = np.diag(y[i])  # yi
                yi_x_yj = np.outer(y[i], y[i])  # yi*yj
                # 对于batch中的每个样本，softmax_grad均为[C,C]
                softmax_grad = yi - yi_x_yj
                # dy/dx => softmax_grad:[C, C]
                # dL/dy => y_grad:[N,C]
                # y_grad[i]:[1,C]
                # x_grad:[C,1]
                x_grad = softmax_grad @ y_grad[i].T
                batch_grads.append(x_grad)
            # batch_grads:[N,C]
            batch_grads = np.array(batch_grads, dtype=x.dtype)
            return batch_grads

    # 可以提高数值稳定性
    # p = log_softmax(x)
    # pi = log(exp(xi)/sum(exp(xj)))
    #
    # 数值稳定性优化
    # A = max(xj)
    # pi = exp(xi-A)/sum(exp(xj-A))
    #
    # yi=log(pi) = log(exp(xi-A)) - log(sum(exp(xj-A)))
    #  = (xi-A) - log(sum(exp(xj-A)))
    #
    class LogSoftmax(Func):
        @staticmethod
        def forward(x: np.ndarray, axis=1, **kwargs) -> np.ndarray:
            # axis: which axis to calculate softmax
            # inds = x.data.argmax(axis=axis)
            keepdims = True
            x_max = x.max(axis=axis, keepdims=keepdims)
            x_minus = x - x_max
            prob = x_minus - np.log(np.sum(np.exp(x_minus), axis=axis, keepdims=keepdims))
            return prob

        # y = log(softmax(x))
        # dy/dx=1/softmax(x)* dsoftmax/dx
        # 或者：
        # A = max(xj)
        # yi=log(pi) = log(exp(xi-A)) - log(sum(exp(xj-A)))
        #  = (xi-A) - log(sum(exp(xj-A)))
        @staticmethod
        def backward(x: np.ndarray, y: np.ndarray, y_grad: np.ndarray, batch_axis=0, **kwargs) -> np.ndarray:
            # FIXME:梯度计算不正确
            return Functional.Softmax.backward(x,y,y_grad, batch_axis)/Functional.Softmax.forward(x)

    # y = tanh(x)
    # dy/dx = 1-y*y
    class Tanh(Func):
        @staticmethod
        def forward(x: np.ndarray, axis=None, **kwargs) -> np.ndarray:
            # out = Tensor(np.tanh(x.data), _prev_nodes=(x,), _op='Tanh')
            return np.tanh(x.data)

        @staticmethod
        def backward(x: np.ndarray, y: np.ndarray, y_grad: np.ndarray, **kwargs) -> np.ndarray:
            # x_out=Tanh(x)
            return (1 - y * y) * y_grad

    class Identity(Func):
        # @classmethod
        @staticmethod
        def forward(x: np.ndarray, axis=None, **kwargs) -> np.ndarray:
            # out = Tensor(out_data, _prev_nodes=(x,), _op='Identity')
            return x.copy()

        @staticmethod
        def backward(x: np.ndarray, y: np.ndarray, y_grad: np.ndarray, batch_axis=0, **kwargs) -> np.ndarray:
            return y_grad.copy()

    class SigmoidCrossEntropyWithLogitLoss:
        """
         tensorflow 中的计算方式
         def sigmoid_cross_entropy_with_logits(_sentinel=None, # pylint: disable=invalid-name labels=None, logits=None, name=None):

         Computes sigmoid cross entropy given logits.

         Measures the probability error in discrete classification tasks in which each
         class is independent and not mutually exclusive. For instance, one could
         perform multilabel classification where a picture can contain both an elephant
         and a dog at the same time.

         For brevity, let x = logits, z = labels,z is 0 or 1. The logistic loss is
        for x>0,
        loss= z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
         = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
         = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
         = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
         = (1 - z) * x + log(1 + exp(-x))
         = x - x * z + log(1 + exp(-x))
              = x - x*z + log(1+exp(-abs(x)))
              =  max(x, 0) - x * z + log(1 + exp(-abs(x)))

         即当x很大时,exp(x)会产生上溢，不会太安全，而exp(-x)，只会下溢为0，而不会上溢

         For x < 0, to avoid overflow in exp(-x), we reformulate the above

         x - x * z + log(1 + exp(-x))
         = log(exp(x)) - x * z + log(1 + exp(-x))
         = - x * z + log(1 + exp(x))
              = 0 - x * z + log(1 + exp(-abs(x)))
             =  max(x, 0) - x * z + log(1 + exp(-abs(x)))

         Hence, to ensure stability and avoid overflow, the implementation uses this
         equivalent formulation

         max(x, 0) - x * z + log(1 + exp(-abs(x)))

        即它只会使得exp(x)产生下溢为0，而非上溢，可以使得计算具有稳定性
         logits and labels must have the same type and shape.
        """

        # -[y*log(p) + (1-y)*log(1-p)]
        @staticmethod
        def forward(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            # x:[N]
            # y:[N]
            # y in {0, 1}
            # original implement
            #
            # prob = Functional.Sigmoid.forward(x)
            # logprob = np.log(prob)
            # one_minus_logprob = np.log(1 - prob)
            # is_pos = y > 0
            # loss = -(logprob * is_pos + (~is_pos) * one_minus_logprob)

            # numberic compute stability
            assert x.shape == y.shape
            loss = Functional.Relu().forward(x) - x * y + np.log(1 + np.exp(-np.abs(x)))
            return loss

        @staticmethod
        def backward(y_pred: np.ndarray, y: np.ndarray) -> np.ndarray:
            # y in {0, 1}
            # x:[N]
            # y:[N], 为label
            # y_pred:[N]
            # dL/dx = y_pred - y
            assert y_pred.shape == y.shape
            grad = y_pred - y
            return grad

    class SoftmaxCrossEntropyWithLogitLoss:
        @staticmethod
        def forward(x: np.ndarray, y: np.ndarray, axis=1) -> np.ndarray:
            """
            log(sum_i(exp(x_i))) = A + log(sum_i(exp(x_i - A)))
            where A = max(x_1,...,x_n)

            loss = -sum_i{ yi*log(pi) }
            """
            # x:[N,C]
            # y:[N]

            assert len(x.shape)==2
            assert len(y.shape)==1
            batch_size= x.shape[0]
            pred = Functional.Softmax().forward(x)
            log_prob = np.log(pred)
            loss = - log_prob[np.arange(batch_size), y.astype(np.int64)]
            return loss

        @staticmethod
        def backward(y_pred: np.ndarray, y_label_index: np.ndarray) -> np.ndarray:
            # y_pred:[N,C]
            # label_index:[N],其中label_index值在[0,C-1]之间
            # grad = pred - y
            assert len(y_pred.shape)==2
            assert len(y_label_index.shape)==1

            batch_size = y_pred.shape[0]
            y = np.zeros_like(y_pred)
            y[np.arange(batch_size), y_label_index.astype(np.int64)] = 1
            grad = y_pred - y
            return grad

