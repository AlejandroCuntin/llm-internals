import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn import MLP
from nn_viz import draw_network


ASSETS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets",
)
os.makedirs(ASSETS, exist_ok=True)


# Small network for a clean diagram
model = MLP(2, [4, 4, 1])

# Draw without activations (structure only)
draw_network(
    model,
    save_path=os.path.join(ASSETS, "network_structure.svg"),
    show=False,
    title="MLP 2 → 4 → 4 → 1",
)

# Draw with an input, showing activations
x = [1.0, 2.0]
draw_network(
    model,
    x=x,
    save_path=os.path.join(ASSETS, "network_forward.svg"),
    show=False,
    title="Forward pass with input [1.0, 2.0]",
)

print("Done. Open the SVGs in assets/.")