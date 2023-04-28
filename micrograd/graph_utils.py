# pip install graphviz
import numpy as np
from graphviz import Digraph
from micrograd.engine import Value
from tensorgrad.tensor import Tensor

def trace(root:Value):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root:Value, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})  # , node_attr={'rankdir': 'TB'})

    for n in nodes:
        dot.node(name=str(id(n)), label="{ %s|data:%.4f|grad:%.4f}" % (n.name,n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

def draw_dot_tensor(root:Tensor, format='svg', rankdir='LR', show_data=True):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})  # , node_attr={'rankdir': 'TB'})
    # 设置numpy float打印格式
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

    for n in nodes:
        if show_data:
            dot.node(name=str(id(n)),
                     label="{%s|shape:%s|data %s|grad:%s }" % (n.name, str(n.shape), str(n.data), str(n.grad)),
                     shape='record')
        else:
            dot.node(name=str(id(n)),
                     label="{%s|shape:%s}" % (n.name, str(n.shape)),
                     shape='record')

        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot
