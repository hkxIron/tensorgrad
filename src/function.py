import math
import numpy as np

class Functional:
    class Func:

        """
        x为输入，y为输出, y=f(x)
        @param x 输入
        @return y
        """
        @staticmethod
        def forward(x: np.ndarray, **kwargs) -> np.ndarray:
            pass

        """
        x为输入，y为输出
        y=f(x)
        y_grad: 反向传播时，loss对输出y的导数dL/dy
        返回值为L对x的导数:dL/dx 
        dL/dx = dL/dy * dy/dx
        
        @param x 输入
        @return dL/dx
        """
        @staticmethod
        def backward(x: np.ndarray, y: np.ndarray, y_grad: np.ndarray, **kwargs) -> np.ndarray:
            pass

    class LossFunc:

        """
        x为输入，y为label
        """
        @staticmethod
        def forward(x: np.ndarray, y:np.ndarray) -> np.ndarray:
            pass

        @staticmethod
        def backward(x: np.ndarray, y: np.ndarray) -> np.ndarray:
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
            # NOTE:relu(x)导数为：
            #   x>=0: dl/dx=1
            #   x<0: dL/dx=0
            #   dL/dx = dL/dy * dy/dx
            grad = (x > 0) * y_grad  # <0的地方直接为0
            return grad

    # y = sigmoid(x) = 1/(1+exp(-x))
    # dy/dx = y(1-y)
    class Sigmoid(Func):
        @staticmethod
        def forward(x: np.ndarray, **kwargs) -> np.ndarray:
            """
            y = 1/(1+exp(-x))
            caffe中的数值稳定性优优化方法
            if x>0
                exp(-x)接近于0, 因此计算是稳定的，不会产生上溢
                等价于：1/(1+exp(-abs(x)))
            else if x<0
                1/(1+exp(-x)) = 1- 1/(1+exp(x)) = 1- 1/(1+exp(-abs(x)))
                其中 exp(-abs(x)) 接近于0，因此计算也是稳定的

            因此需要计算: abs_sigmoid = 1/(1+exp(-abs(x)))
            """
            # origin: 1/(1+exp(-x))
            # numberic stable version
            is_pos = x>0
            abs_sigmoid = 1 / (1 + np.exp(-np.abs(x)))
            return is_pos*abs_sigmoid + (~is_pos)*(1 - abs_sigmoid)

        @staticmethod
        def backward(x: np.ndarray, y: np.ndarray, y_grad: np.ndarray, **kwargs) -> np.ndarray:
            # x_out=Sigmoid(x)
            # NOTE: sigmoid(x)的导数为：dy/dx= sigmoid(x)(1-sigmoid(x)) = y(1-y)
            #  其中 y = p = sigmoid(x)
            # x.grad += (x_out.data*(1-x_out.data)) * x_out.grad
            return y * (1 - y) * y_grad

    class LogSigmoid(Func):
        @staticmethod
        def forward(x: np.ndarray, **kwargs) -> np.ndarray:
            """
            x: [N, D]
            y = log(1/(1+exp(-x))) = - log(1+exp(-x))

            caffe中的数值稳定性优优化方法
            if x>0
                exp(-x)接近于0, 因此计算是稳定的，不会产生上溢
                等价于：-log(1+exp(-abs(x))), 其中x越大，exp(-abs(x)) 接近于0，因此计算也是稳定的

            else if x<0, -x>0, 南要尽量计算exp(x)而不是exp(-x)
                -log(1+exp(-x)) = log(1/(1+exp(-x))
                分子分母同时乘以exp(x),
                = log(exp(x)/(exp(x)+1))
                = log(exp(-abs(x))/(exp(-abs(x)) +1)
                = -abs(x) - log(1+exp(-abs(x)))

                其中 x越小，exp(-abs(x)) 接近于0，因此计算也是稳定的
            """
            # origin: 1/(1+exp(-x))
            # numberic stable version
            is_pos = x>0
            abs_log_exp = -np.log(1 + np.exp(-np.abs(x)))
            return is_pos*abs_log_exp + (~is_pos)*(-abs(x) + abs_log_exp)

        """
        x:[C]
        y = log(1/(1+exp(-x))) = - log(1+exp(-x))
        p = sigmoid(x)
        
        dy_i/dx_i 
        = -1/(1+exp(-x_i))*exp(-x_i)*(-1)
        = exp(-x_i)/(1+exp(-x_i)) = (1+exp(-x_i) -1)/(1+exp(-x_i)) 
        = 1 - 1/(1+exp(-x_i)) 
        = 1 - p
        """
        @staticmethod
        def backward(x: np.ndarray, y: np.ndarray, y_grad: np.ndarray, **kwargs) -> np.ndarray:
            # x_out=Sigmoid(x)
            # NOTE: log_sigmoid(x)的导数为：dy/dx= log_sigmoid(x)(1-sigmoid(x)) = 1-y
            #  其中 y = p = sigmoid(x)
            p = Functional.Sigmoid(x)
            return (1 - p) * y_grad

    # NOTE:softmax推导：
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
        def forward(x: np.ndarray, axis=-1, **kwargs) -> np.ndarray:
            # axis: which axis to calculate softmax
            keepdims = True
            x_max = x.max(axis=axis, keepdims=keepdims)
            exp_x = np.exp(x - x_max)
            prob = exp_x / np.sum(exp_x, axis=axis, keepdims=keepdims)
            return prob

        # x:[N, C]
        # y: [N,C]
        # y_grad: [N,C]
        # grad:[N,C]
        @staticmethod
        def backward(x: np.ndarray, y: np.ndarray, y_grad: np.ndarray, batch_axis=0, **kwargs) -> np.ndarray:
            """
             NOTE: 结论
                dy/dx = diag(y) - y @ y.T
                = diag(p) - p@p.T = , shape:[C, C]

            对于一个样本，假设x维度为3,则预测的类别数也为3, 则x=[x1,x2,x3], y=[y1,y2, y3]
             [y1,y2,y3] = softmax([x1,x2,x3])

            即
             x1-> exp(x1) -> y1=exp(x1)/sum(exp(xj))
             x2-> exp(x2) -> y2=exp(x2)/sum(exp(xj))
             x3-> exp(x3) -> y3=exp(x3)/sum(exp(xj))
             =>
             xi-> exp(xi) -> yi=exp(xi)/sum(exp(xj))

            下面分2种情况讨论
            1. i=j时:
             dyi/dxi = exp(xi)/sum(exp(xk)) + exp(xi)*(-1)*sum(exp(xk))^(-2)*exp(xi)
             化间后有
             = yi - yi*yi

            2. i!=j时：
            dyi/dxj = exp(xi) * (-1)* sum(exp(xk))^(-2)*exp(xj)
            = -yi*yj

            因此可以归纳出以下求导公式：
            y = p = softmax(x)
            yi = exp(xi)/sum(exp(xk))
            if i=j:
              dyi/dxj = y_i*(1-yj) = y*i*(1-yi) = yi-yi*yi
            else:
              dyi/dxj = -yi*yj

            =>
            而下一个节点的梯度已知，为:
            dL/dy = [dL/dy1, dL/dy2, dL/dy3], 即为代码中的y_grad

            举例

            同时由前向传播可知
            x1-> exp(x1) -> y1=exp(x1)/sum(exp(xj)) -> loss
            x2-> exp(x2) -> y2=exp(x2)/sum(exp(xj)) -> loss
            x3-> exp(x3) -> y3=exp(x3)/sum(exp(xj)) -> loss

            即x1可通过y1,y2,y3影响最终loss L:
            x1->y1 -> Loss
            x1->y2 -> Loss
            x1->y3 -> Loss
            即x1对L的影响路径为, 因此里面必然包含相加：
            x1 -> [y1, y2, y3 ] -> Loss

            因此, 将Loss简写为L:
            dL/dx = [dL/dx1, dL/dx2, dL/dx3]

            其中L对dxi的梯度为:
            dL/dxi = sum{k}{ dL/dyk*(dyk/dxi) }
            = dL/dy1*dy1/dxi + dL/dy2*dy2/dxi + dL/dy3*dy3/dxi
            = [dy1/dxi dy2/dxi dy3/dxi]* [dL/dy1
                                          dL/dy2
                                          dL/dy3]
            = [dy1/dxi dy2/dxi dy3/dxi]* [dL/dy1 dL/dy2 dL/dy3].T
            = dy/dx * (dL/dy).T
            = softmax_grad * y_grad.T

            其中
            dL/dy = [dL/dy1 dL/dy2 dL/dy3]


            推导 :
            dY/dx= softmax_grad梯度矩阵为
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
                y1*y1 y1*y2 y1*y3 # 向量外积
                y2*y1 y2*y2 y2*y3
                y3*y1 y3*y2 y3*y3
            ]
            = diag(Y) - Y @ Y.T

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
            b为batch中第b个样本

            diag[yb] = np.diag(y[b]) =
            [
                y1
                    y2
                        y3
            ]

            [ yb@yb.T ] = np.outer(yb, yb)  =
            [
                y1*y1 y1*y2 y1*y3
                y2*y1 y2*y2 y2*y3
                y3*y1 y3*y2 y3*y3
            ]

            因此:
            dY/dX = softmax_grad = np.diag(Yb) - np.outer(Yb, Yb)
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
            # NOTE: batch中有多个样本，也会计算多个loss, 所有loss最后拼接在一起
            batch_grads = np.array(batch_grads, dtype=x.dtype)
            return batch_grads

    # log_softmax可以提高数值稳定性, 因为exp(x)可能会上溢
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
        # x:[N, C]
        # y:[N, C]
        @staticmethod
        def forward(x: np.ndarray, axis=1, **kwargs) -> np.ndarray:
            # axis: which axis to calculate softmax
            keepdims = True
            x_max = x.max(axis=axis, keepdims=keepdims)
            x_minus = x - x_max
            prob = x_minus - np.log(np.sum(np.exp(x_minus), axis=axis, keepdims=keepdims))
            return prob

        """
         p = softmax(x)
         y = log(p) = log(softmax(x))
         yi = log(exp(xi)) - log(sum(exp(xk))) = xi - log(sum(exp(xk)))
         =>
         dy/dx=1/p* dp/dx
         
         一般地，为了数值计算稳定性，先减去最大值A, 有如下变换：
         A = max(xj)
         yi = log(pi) 
          = log(exp(xi-A)) - log(sum(exp(xj-A)))
          = (xi-A) - log(sum(exp(xj-A)))
          
         下面进行dlog_softmax/dx的求导
         yi = xi - log(sum(exp(xk))) 
        
         1. 第一部分:
         dxi/dxj = 
             if i=j:
                1 
             else:
                0 
         2. 上述公式后半部分的求导
         d[log(sum(exp(xk)))]/dxj 
         = (1/(sum(exp(xk))))*exp(xj) 
         = exp(xj)/(sum(exp(xk))) = pj
         
         将两部分结合，则dyi/dxj的导数为：
         if i=j:
             dyi/dxj = 1 - pj
         else:
             dyi/dxj = - pj
             
         即dyi/dxj = [1-p1,  -p1, -p1
                       -p2, 1-p2, -p2
                       -p3,  -p3, 1-p3]    
                  = I - p.repeat((1,C)) 
                  = I - softmax(x).repeat((1,C))
                  = Eye(C) - softmax(x).repeat((1, C))
        NOTE: 对于Y=log_softmax(x)函数，导数为：
            dy/dX = I - p.repeat((1,C)) 
        """
        # x: [N,C]
        # y: [N,C]
        # y_grad: [N,C]
        # x_grad: [N,C]
        @staticmethod
        def backward(x: np.ndarray, y_grad: np.ndarray, batch_axis=0, softmax_axis=-1, **kwargs) -> np.ndarray:
            """
            NOTE: 对于y=log_softmax(x)函数，导数为：
                dy/dx
                = I - softmax(x).repeat((1, C))
                = I - p.repeat((1,C))
                其中：p = softmax(x)
            """
            batch_size = x.shape[batch_axis]
            C = x.shape[-1]
            # 对于batch中的每个样本进行处理
            batch_grads = []
            # eye:[C, C]
            eye = np.eye(C) # 对角线为1的矩阵
            # p:[N, C]
            p = Functional.Softmax.forward(x, axis=softmax_axis)
            for i in range(batch_size): # 对于第i个样本
                # 对于batch中的每个样本，softmax_grad均为[C,C]
                # dy/dx => delta:[C, C]
                delta = eye - np.repeat(p[i].reshape(C, 1), repeats=C, axis=softmax_axis)
                # dL/dy => y_grad:[N,C]
                # y_grad[i]:[1,C]
                # x_grad:[C,1]
                # dL/dx = dL/dy*dy/dx
                x_grad = delta @ y_grad[i].T
                batch_grads.append(x_grad)
            # batch_grads:[N,C]
            batch_grads = np.array(batch_grads, dtype=x.dtype)
            return batch_grads

    # NOTE: dy/dx = 1-y*y
    #   y = tanh(x)
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

    class SigmoidCrossEntropyWithLogitLoss(LossFunc):
        """
         NOTE:亦称为BCE loss, 即 binary cross entropy loss
         tensorflow 中的计算方式
         def sigmoid_cross_entropy_with_logits(_sentinel=None, # pylint: disable=invalid-name labels=None, logits=None, name=None):

         Computes sigmoid cross entropy given logits.

         Measures the probability error in discrete classification tasks in which each
         class is independent and not mutually exclusive. For instance, one could
         perform multilabel classification where a picture can contain both an elephant
         and a dog at the same time.

        NOTE: origin loss= z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))

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

         即当x很大时,exp(x)会产生上溢，不安全，而exp(-x)，只会下溢为0，而不会上溢

         For x < 0, to avoid overflow in exp(-x), we reformulate the above

         x - x * z + log(1 + exp(-x))
         = log(exp(x)) - x * z + log(1 + exp(-x))
         = - x * z + log(1 + exp(x))
              = 0 - x * z + log(1 + exp(-abs(x)))
             =  max(x, 0) - x * z + log(1 + exp(-abs(x)))

         Hence, to ensure stability and avoid overflow, the implementation uses this
         equivalent formulation

         NOTE: binary_cross_entropy_loss  = max(x, 0) - x * z + log(1 + exp(-abs(x)))

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
        def backward(y_pred: np.ndarray, label: np.ndarray) -> np.ndarray:
            # y in {0, 1}
            # x:[N]
            # label:[N], 为label
            # y_pred:[N]
            # dL/dx = y_pred - y
            assert y_pred.shape == label.shape
            # NOTE: SigmoidCrossEntropyWithLogitLoss 的导数为：
            #   dL/dx = p - z

            """
            L = loss= z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))   
            记 p = sigmoid(x) 
            则： 
                L = -[z * log(p) + (1-z)*log(1-p)] 
            求导：
            dL/dx = -[z * 1/p* p(1-p) + (1-z)*1/(1-p)*(-1)*p(1-p) ]
             = - [z*(1-p) - (1-z)*p]
             = (1-z)p-z(1-p)
             = p-zp-z+zp
             = p-z
             
            """
            grad = y_pred - label
            return grad

    class SoftmaxCrossEntropyWithLogitLoss(LossFunc):
        @staticmethod
        def forward(x: np.ndarray, label: np.ndarray) -> np.ndarray:
            """
            log(sum_i(exp(x_i))) = A + log(sum_i(exp(x_i - A)))
            where A = max(x_1,...,x_n)

            loss = -sum_i{ zi*log(pi) }, z为label
            """
            # x:[N,C]
            # label:[N]

            assert len(x.shape)==2
            assert len(label.shape) == 1
            batch_size= x.shape[0]
            pred = Functional.Softmax().forward(x)
            log_prob = np.log(pred)
            loss = - log_prob[np.arange(batch_size), label.astype(np.int64)]
            return loss

        """
        给定样本标签为 one-hot 向量 `z`，特征为 `x` 向量，`p = softmax(x)`，SoftmaxCrossEntropyWithLogitLoss = `L`，推导 `dL/dx`。

        ## 定义

        令：
        - `x = [x₁, x₂, ..., xₙ]` 为 logits 向量
        - `z = [z₁, z₂, ..., zₙ]` 为 one-hot 编码的标签向量（只有一个元素为1，其余为0），有N个类
        - `p = softmax(x) = [p₁, p₂, ..., pₙ]`，其中 `pᵢ = eˣⁱ / ∑ⱼ eˣʲ`
        - 交叉熵损失：`L = -∑ᵢ zᵢ log(pᵢ)`

        ## 推导过程

        ### 1. 先求 ∂L/∂pⱼ
        由于 `L = -∑ᵢ zᵢ log(pᵢ)`，所以：
        ```
        ∂L/∂pⱼ = -zⱼ/pⱼ, 注意：∂L/∂p是一个向量
        ```

        ### 2. 再求 ∂pⱼ/∂xₖ
        对于 softmax 函数 `pⱼ = eˣʲ / ∑ₘ eˣᵐ`，需要分两种情况：

        **情况1：当 j = k 时**
        ```
        ∂pⱼ/∂xₖ = ∂/∂xₖ (eˣʲ / S) 
        = (eˣʲ·S - eˣʲ·eˣᵏ) / S² 
        = pⱼ(1 - pₖ) 
        = pⱼ -  pⱼpₖ
        = pⱼ -  pⱼpⱼ
        ```
        其中 `S = ∑ₘ eˣᵐ`， ∂S/∂xₖ = eˣᵏ


        **情况2：当 j ≠ k 时**
        ```
        ∂pⱼ/∂xₖ = ∂/∂xₖ (eˣʲ / S)
        = (0 - eˣʲ·eˣᵏ) / S² 
        = - pⱼpₖ = 0 - pⱼpₖ
        ```
        因此,对于3维向量，我们有
        ∂p/∂x = [
           [p0-p0*p0, -p0*p1, -p0*p2],
           [-p1*p0, p1-p1*p1, -p1*p0],
           [-p2*p0, -p2*p1, p2-p2*p2],
        ]
        =
        [
           [p0,0,0],
           [0,p1,0],
           [0,0,p2]
        ]-[
           [-p0*p0, -p0*p1, -p0*p2],
           [-p1*p0, -p1*p1, -p1*p0],
           [-p2*p0, -p2*p1, -p2*p2],
        ]
        = diag(p) - p@p.T

        综上，对于∂p/∂x = diag(p) - p@p.T


        ### 3. 最后求 ∂L/∂xₖ（使用链式法则）
        ```
        ∂L/∂xₖ = ∑ⱼ (∂L/∂pⱼ) · (∂pⱼ/∂xₖ)
                = ∑ⱼ (-zⱼ/pⱼ) · (∂pⱼ/∂xₖ)
        ```

        将两种情况代入：
        ```
        ∂L/∂xₖ = (-zₖ/pₖ) · pₖ(1 - pₖ) 
                 + ∑ⱼ≠ₖ (-zⱼ/pⱼ) · (-pⱼpₖ)
                = -zₖ(1 - pₖ) + ∑ⱼ≠ₖ zⱼpₖ
                = -zₖ + zₖpₖ + pₖ∑ⱼ≠ₖ zⱼ
        ```

        由于 `z` 是 one-hot 向量，`∑ⱼ zⱼ = 1`，所以 `∑ⱼ≠ₖ zⱼ = 1 - zₖ`：
        ```
        ∂L/∂xₖ = -zₖ + zₖpₖ + pₖ(1 - zₖ)
                = -zₖ + zₖpₖ + pₖ - zₖpₖ
                = pₖ - zₖ
        ```

        ## 最终结果

        因此，对于向量形式：
        ```
        dL/dx = p - z
        ```

        或者对于每个元素：
        ```
        ∂L/∂xₖ = pₖ - zₖ
        ```

        ## 总结
        对于 SoftmaxCrossEntropyWithLogitLoss：
        ```
        dL/dx = softmax(x) - z = p-z
        ```
        这个简洁的结果表明：
        - 梯度等于预测概率 `p` 减去真实标签 `z`
        - 当预测正确时（`p ≈ z`），梯度接近于0
        - 当预测错误时，梯度会推动参数向正确方向更新
        """

        # NOTE: dL/dx = softmax(x) - z = p-z, 其中p=softmax(x), 与BCE loss的梯度相同
        @staticmethod
        def backward(y_pred: np.ndarray, label: np.ndarray) -> np.ndarray:
            # y_pred:[N,C]
            # label_index:[N],其中label_index值在[0,C-1]之间
            # grad = pred - y
            assert len(y_pred.shape)==2
            assert len(label.shape) == 1

            batch_size = y_pred.shape[0]
            y = np.zeros_like(y_pred)
            y[np.arange(batch_size), label.astype(np.int64)] = 1
            grad = y_pred - y
            return grad

