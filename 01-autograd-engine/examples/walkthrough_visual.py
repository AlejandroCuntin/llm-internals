import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import Value
from viz import draw_dot

ASSETS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets",
    )

os.makedirs(ASSETS, exist_ok=True)

steps = [] # title,description and svg_filename

def snapshot(name, title, description, node):
    """
    Save an SVG of the graph rooted at 'node', and record it.
    """
    path = os.path.join(ASSETS, f"Walkthrough_ {name}")
    draw_dot(node).render(path, cleanup=True)
    svg = f"walkthrough_ {name}.svg"
    steps.append((title,description,svg))
    print(f"[{name} {title}]")
    print(f"  {description}")
    print()

def main():
    # step 1: a single Value, no graph yet

    snapshot(
        "01_Value",
        "Step 1 - a single value",
        "Just a member with grad = 0. No graph yet",
        Value(2.0),
    )

    # step 2: two indpendent input Values

    a = Value(2.0)
    b = Value(3.0)
    snapshot(
        "02_inputs",
        "Step 2 - two input values",
        "a = Value(2.0). A single node with no parents"
        "b = Value(3.0) also exists, but it is not connected to a, "
        "so draw_dot does not show it. Only the graph reachable from"
        "Two independent nodes. No edges yet",
        a,
    )

    # step 3: multiply them. The graph grows by one node.

    c = a * b
    snapshot(
        "03_mul",
        "Step 3 - a * b",
        "A new node c. It has two parents (a and b) and an op label '*'",
        c,
    )

    # step 4: add 'a' again. Now 'a' feeds two different operations.

    d = c + a

    snapshot(
        "04_add",
        "Step 4 - c + a",
        "Node d has parents c and a. Note a feedes into two different ops",
        d,
    )

    # step 5: we run backward. Every node    gets its gradient filled

    d.backward()
    snapshot(
        "05_backward",
        "Step 5 - backward()",
        "Every node now has a grad, a shows the sum of its two paths",
        d,
    )

# we write an html page that stiches all SVGs together

html = [
        "<html><head><meta charset='utf-8'>",
        "<title>Autograd walkthrough</title>",
        "<style>",
        "  body { font-family: sans-serif; max-width: 900px; margin: 40px auto; }",
        "  h1 { border-bottom: 2px solid #333; padding-bottom: 8px; }",
        "  .step { margin: 40px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }",
        "  .step h2 { margin-top: 0; }",
        "  .step p { color: #555; }",
        "  .step img { max-width: 100%; border: 1px solid #eee; }",
        "</style>",
        "</head><body>",
        "<h1>Autograd engine — visual walkthrough</h1>",
        "<p>Every panel is a snapshot of the graph at that step. "
        "Scroll down to see the graph grow and gradients fill in.</p>",
    ]
    
for title, description, svg in steps:
    html.append("<div class='step'>")
    html.append(f"  <h2>{title}</h2>")
    html.append(f"  <p>{description}</p>")
    html.append(f"  <img src='{svg}' alt='{title}'>")
    html.append("</div>")

html.append("</body></html>")

html_path = os.path.join(ASSETS, "walkthrough.html")
with open(html_path, "w", encoding="utf-8") as f:
    f.write("\n".join(html))

print(f"Wrote {html_path}")
print("Open it in your browser.")


if __name__ == "__main__":
    main()




