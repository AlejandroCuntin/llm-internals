import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

def draw_network(model, x=None, save_path=None, show=True, title=None,
                 show_values=False):
    """
    Draw an MLP as a layered neural network diagram.

    Args:
        model: an MLP instance from nn.py
        x: optional input vector. If given, forward pass is run and
           activations are shown as neuron colors.
        save_path: if given, save the figure there (png or svg).
        show: if True, call plt.show().
        title: optional title.
        show_values: if True, print activation values to the right of
                     each neuron. Off by default for a clean diagram.
    """
    layers = model.layers
    sizes = [len(layers[0].neurons[0].w)]  # input size
    for layer in layers:
        sizes.append(len(layer.neurons))

    n_layers = len(sizes)
    max_neurons = max(sizes)

    # Layer spacing (horizontal). Wider than 1.0 to leave room for
    # labels to the right of each neuron without overlapping the next layer.
    layer_spacing = 1.8

    # Compute activations by running the forward pass.
    activations = [list(x)] if x is not None else None
    if x is not None:
        h = x
        for layer in layers:
            h = layer(h)
            if not isinstance(h, list):
                h = [h]
            activations.append([v.data for v in h])

    fig, ax = plt.subplots(figsize=(n_layers * 2.6, max(max_neurons * 0.9, 3)))
    fig.patch.set_facecolor("white")

    # Layered positions: each layer has an x-coordinate, neurons spread on y.
    positions = []
    for i, size in enumerate(sizes):
        x_pos = i * layer_spacing
        y_positions = [j - (size - 1) / 2 for j in range(size)]
        positions.append([(x_pos, y) for y in y_positions])

    # Draw connections (weights) first, so circles go on top.
    for i, layer in enumerate(layers):
        for j, neuron in enumerate(layer.neurons):
            x1, y1 = positions[i + 1][j]
            for k, w in enumerate(neuron.w):
                x0, y0 = positions[i][k]
                _draw_connection(ax, (x0, y0), (x1, y1), w.data)

    # Draw neurons.
    for i, layer_positions in enumerate(positions):
        for j, (x0, y0) in enumerate(layer_positions):
            if i == 0:
                color = "white"
                label = ""
            else:
                color_val = activations[i][j] if activations else 0.0
                color = _activation_color(color_val)
                label = f"{color_val:.2f}" if (activations and show_values) else ""
            _draw_neuron(ax, (x0, y0), color, label)

    # Draw bias markers: small squares to the right of each non-input neuron.
    for i, layer in enumerate(layers):
        for j, neuron in enumerate(layer.neurons):
            x0, y0 = positions[i + 1][j]
            _draw_bias(ax, (x0, y0), neuron.b.data)

    # Layer labels above each column.
    labels = ["input"] + [f"hidden {i}" for i in range(1, n_layers - 1)] + ["output"]
    for i, label in enumerate(labels):
        ax.text(i * layer_spacing, max_neurons / 2 + 1.3, label,
                ha="center", va="bottom", fontsize=11, color="#555")

    # Layout.
    ax.set_xlim(-0.8, (n_layers - 1) * layer_spacing + 0.8)
    ax.set_ylim(-max_neurons / 2 - 1.2, max_neurons / 2 + 2.0)
    ax.set_aspect("equal")
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=14, pad=20)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved {save_path}")

    if show:
        plt.show()

    plt.close(fig)



#Drawing primitives

def _draw_neuron(ax, pos, color, label):
    """
    Draw a single neuron as a circle.
    If label is given, it appears to the right of the circle,
    not inside it, to avoid text overlapping.
    """
    circle = Circle(pos, radius=0.30, facecolor=color, edgecolor="#333",
                    linewidth=1.5, zorder=3)
    ax.add_patch(circle)
    if label:
        ax.text(pos[0] + 0.40, pos[1], label,
                ha="left", va="center",
                fontsize=8, color="#222", zorder=4)


def _draw_connection(ax, start, end, weight):
    """
    Draw a weighted connection. Curved line, color by sign,
    width by magnitude.
    """
    color = "#d62728" if weight < 0 else "#2ca02c"
    width = min(abs(weight) * 2, 4.0) + 0.3
    alpha = min(abs(weight) + 0.2, 1.0)
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle="-",
        connectionstyle="arc3,rad=0.08",
        color=color,
        linewidth=width,
        alpha=alpha,
        zorder=1,
    )
    ax.add_patch(arrow)


def _draw_bias(ax, pos, bias):
    """
    Bias marker: a small square to the right of the neuron.
    Color by sign, size by magnitude (clamped).
    """
    color = "#d62728" if bias < 0 else "#2ca02c"
    size = min(abs(bias) * 0.08, 0.10) + 0.05
    ax.add_patch(plt.Rectangle(
        (pos[0] + 0.34, pos[1] - size / 2),
        size, size,
        facecolor=color,
        edgecolor="#333",
        linewidth=0.8,
        zorder=3,
        alpha=0.85,
    ))


def _activation_color(value):
    """Map activation value to a color from blue (negative) to red (positive)."""
    v = max(-1.0, min(1.0, value))
    if v >= 0:
        r, g, b = 1.0, 1.0 - v, 1.0 - v
    else:
        r, g, b = 1.0 + v, 1.0 + v, 1.0
    return (r, g, b)