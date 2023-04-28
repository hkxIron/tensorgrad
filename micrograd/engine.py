import math

class Value:
    """ stores a single scalar value and its gradient """
    data: float
    grad: float

    # 注意：每个值都带有操作符,以及他的子结点（即c=a+b,那么c的子结点就是a,b）
    def __init__(self, data, _children=(), _op='', name=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward_grad_fn = lambda: None
        self._prev = set(_children) # 需要要用set去重，因为c=a+a，他的子结点相同，均为a
        self._op = _op # the op that produced this node, for graphviz / debugging / etc
        self.name = name

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        # 计算梯度
        def _backward_grad():
            self.grad += out.grad # 注意：之所以这里用+=而不是=，是因为b=a+a里会引用两次a，梯度需要累加
            other.grad += out.grad
        out._backward_grad_fn = _backward_grad

        return out

    # 矩阵相乘
    def __matmul__(self, other):
        pass

    # 逐元素相乘
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward_grad():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward_grad_fn = _backward_grad

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        """
        y = x**n
        dy/dx = n*(x**(n-1))
        """
        def _backward_grad():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward_grad_fn = _backward_grad

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward_grad():
            self.grad += (out.data > 0) * out.grad
        out._backward_grad_fn = _backward_grad

        return out

    def sigmoid(self):
        # y = sigmoid(x) = 1/(1+exp(-x))
        # dy/dx = y(1-y)
        out = Value(1/(1+ math.exp(-self.data)), (self,), 'sigmoid')

        def _backward_grad():
            self.grad += (out*(1-out)) * out.grad
        out._backward_grad_fn = _backward_grad

        return out

    def tanh(self):
        out = Value(math.tanh(self.data), (self,), 'tanh')

        # y = tanh(x)
        # dy/dx = 1-y*y
        def _backward_grad():
            self.grad += (1-out*out) * out.grad
        out._backward_grad_fn = _backward_grad

        return out

    # 选遍历获取所有子节点，然后再计算梯度
    def backward(self):
        # topological order all of the children in the graph
        # 后序遍历获取所有子节点（只有最后一个节点需要这样遍历全部一次，其它的节点均不用遍历）
        topo_nodes = []
        visited = set()
        def build_nodes(v):
            # 当前节点没被访问(由于在图中，可能会有相同的子结点，因此需要添加访问标记)
            if v not in visited:
                visited.add(v)
                # 后序遍历(左右根)它的所有孩子
                for child in v._prev:
                    build_nodes(child)
                topo_nodes.append(v)

        build_nodes(self)

        #print("\n".join([str(t) for t in topo_nodes]))

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        # 从后向前传播梯度
        for v in reversed(topo_nodes):
            v._backward_grad_fn()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1 # 因为已经实现过__pow__

    def __repr__(self):
        return f"Value(name:{self.name},data:{self.data},grad:{self.grad})"
