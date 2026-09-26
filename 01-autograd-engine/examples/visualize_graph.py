import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import Value
# pyrefly: ignore [missing-import]
from viz import draw_dot


# A small expression that uses several operations and a shared node.
# f(a, b) = (a * b + a.tanh()) * 2
a = Value(2.0)
b = Value(-3.0)

c = a * b          # a used here
d = a.tanh()       # a used here too (shared node!)
e = c + d
f = e * 2.0


assets = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
os.makedirs(assets, exist_ok=True)

dot_before = draw_dot(f)
path_before = os.path.join(assets, "graph_before_backward.svg")
dot_before.render(path_before.replace(".svg", ""), cleanup=True)
print(f"Wrote {path_before}")

f.backward()


dot_after = draw_dot(f)
path_after = os.path.join(assets, "graph_after_backward.svg")
dot_after.render(path_after.replace(".svg", ""), cleanup=True)
print(f"Wrote {path_after}")

print()
print("Expression: f = (a * b + a.tanh()) * 2")
print(f"a = {a.data}, b = {b.data}")
print()
print(f"f       = {f.data:.6f}")
print(f"a.grad  = {a.grad:.6f}")
print(f"b.grad  = {b.grad:.6f}")