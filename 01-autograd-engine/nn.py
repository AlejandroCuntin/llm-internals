import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch


# Color palette
POS_WEIGHT = "#2c7fb8"     # blue for positive weights
NEG_WEIGHT = "#d95f02"     # orange for negative weights
NEURON_EDGE = "#333333"
NEURON_ACTIVE = "#fdd49e"
NEURON_INACTIVE = "#ffffff"
TEXT_COLOR = "#333333"


def draw_network(model, x=None, save_path=None, show=True, title=None):
    """
    Draw an MLP as a layered neural network diagram.

    Args:
        model:      an MLP instance from nn.py
        x:          optional input vector. If given, forward pass runs
                    and activations are shown as neuron colors.
        save_path:  file to save (png, svg, or pdf)
        show:       call plt.show() at the end
        title:      optional title above the diagram
    """
    layers = model.layers
    sizes = [len(layers[0].neurons[0].w)]     # input size
    for layer in layers:
        sizes.append(len(layer.neurons))

    n_layers = len(sizes)
    max_neurons = max(sizes)

    # Compute activations if an input was provided.
    activations = None
    if x is not None:
        activations = [list(x)]
        h = x
        for layer in layers:
            h = layer(h)
            if not isinstance(h, list):
                h = [h]
            activations.append([v.data for v in h])

    fig, ax = plt.subplots(figsize=(n_layers * 2.2, max(max_neurons * 0.9, 3)))
    fig.patch.set_facecolor("white")

    # Compute layer positions: each layer on its own x, neurons stacked on y.
    positions = []
    for i, size in enumerate(sizes):
        y_start = -(size - 1) / 2
        positions.append([(i, y_start + j) for j in range(size)])

    # Connections (weights)
    for i, layer in enumerate(layers):
        for j, neuron in enumerate(layer.neurons):
            x1, y1 = positions[i + 1][j]
            for k, w in enumerate(neuron.w):
                x0, y0 = positions[i][k]
                _draw_weight(ax, (x0, y0), (x1, y1), w.data)

    # Neurons
    for i, layer_pos in enumerate(positions):
        for j, (x0, y0) in enumerate(layer_pos):
            if i == 0 or activations is None:
                color = NEURON_INACTIVE
                label = ""
            else:
                val = activations[i][j]
                color = _activation_color(val)
                label = f"{val:.2f}"
            _draw_neuron(ax, (x0, y0), color, label)

    # Bias markers: small squares below each non-input neuron

    for i, layer in enumerate(layers):
        for j, neuron in enumerate(layer.neurons):
            x0, y0 = positions[i + 1][j]
            b = neuron.b.data
            _draw_bias(ax, (x0 + 0.45, y0), b)

    #Layer Labels
    labels = ["input"] + [f"hidden {i}" for i in range(1, n_layers - 1)] + ["output"]
    for i, label in enumerate(labels):
        ax.text(i, max_neurons / 2 + 0.9, label, ha="center", va="bottom",
                fontsize=11, color=TEXT_COLOR, weight="bold")

    #Layout
    ax.set_xlim(-0.7, n_layers - 0.3)
    ax.set_ylim(-max_neurons / 2 - 1.2, max_neurons / 2 + 1.5)
    ax.set_aspect("equal")
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=14, pad=20, color=TEXT_COLOR)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved {save_path}")

    if show:
        plt.show()

    plt.close(fig)


#Drawing primitives

def _draw_neuron(ax, pos, color, label):
    """A neuron: a circle with an optional label."""
    circle = Circle(
        pos, radius=0.32,
        facecolor=color,
        edgecolor=NEURON_EDGE,
        linewidth=1.5,
        zorder=3,
    )
    ax.add_patch(circle)
    if label:
        ax.text(pos[0], pos[1], label, ha="center", va="center",
                fontsize=7, color=TEXT_COLOR, zorder=4)


def _draw_weight(ax, start, end, weight):
    """
    A weight: a curved line from start to end.
    Color by sign, thickness by magnitude.
    """
    # Small offset so lines curve slightly, giving a hand-drawn feel.
    color = POS_WEIGHT if weight >= 0 else NEG_WEIGHT
    width = min(abs(weight) * 2.5, 3.5) + 0.4
    alpha = min(abs(weight) * 0.9 + 0.25, 1.0)

    arrow = FancyArrowPatch(
        start, end,
        connectionstyle="arc3,rad=0.08",
        arrowstyle="-",
        color=color,
        linewidth=width,
        alpha=alpha,
        zorder=1,
    )
    ax.add_patch(arrow)


def _draw_bias(ax, pos, bias):
    """Bias marker: small square below-right of each neuron."""
    color = POS_WEIGHT if bias >= 0 else NEG_WEIGHT
    size = min(abs(bias) * 0.15, 0.12) + 0.06
    ax.add_patch(plt.Rectangle(
        (pos[0] - size / 2, pos[1] - size / 2),
        size, size,
        facecolor=color,
        edgecolor=NEURON_EDGE,
        linewidth=0.8,
        zorder=3,
        alpha=0.8,
    ))


def _activation_color(value):
    """Map an activation value to a soft color."""
    # Clamp to [-1, 1] assuming tanh output.
    v = max(-1.0, min(1.0, value))
    if v >= 0:
        # White -> soft orange
        return (1.0, 1.0 - 0.35 * v, 1.0 - 0.6 * v)
    else:
        # White -> soft blue
        return (1.0 + 0.6 * v, 1.0 + 0.35 * v, 1.0)