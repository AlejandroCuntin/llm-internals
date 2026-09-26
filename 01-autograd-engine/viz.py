# pyrefly: ignore [missing-import]
from graphviz import Digraph


def trace(root):
    """
    Walk the graph backward from `root` and return:
      - nodes: a set of every Value reachable from root
      - edges: a list of (parent, child) tuples
    """
    nodes, edges = set(), []

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.append((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    """
    Return a graphviz.Digraph of the graph rooted at `root`.
    Each node shows its data and grad.
    Each operation is drawn as a small box between its inputs and output.
    """
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))

        # The Value node itself: a record-shaped box with data and grad.
        dot.node(
            name=uid,
            label=f"{{ data {n.data:.4f} | grad {n.grad:.4f} }}",
            shape='record',
        )

        # If this Value was created by an operation, draw a small op node
        # and an arrow from the op to the Value.
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    # Edges between parents and their operation.
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot