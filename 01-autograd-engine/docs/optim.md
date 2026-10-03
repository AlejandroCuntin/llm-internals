# `optim.py` — Deep Dive

`engine.py` computes gradients. `nn.py` organizes them into parameters. `optim.py` is the piece that actually **uses** those gradients to change the parameters so the network learns.

It is the smallest file in the project. Everything it does could be written in three lines inside the training loop. The reason it exists as a class is not that the math is complicated — it is that *where* and *how* the update happens has consequences that only appear when you have more than one optimizer, more than one parameter, or more than one training step.

---

## Why an optimizer class at all

The simplest possible training loop updates weights inline:

```python
for step in range(N):
    preds = [model(x) for x in xs]
    loss = compute_loss(preds, ys)

    model.zero_grad()
    loss.backward()

    for p in model.parameters():
        p.data -= 0.01 * p.grad
```

That works. For one optimizer, on one experiment, it is fine. It stops being fine the moment you want any of these:

- **Try Momentum or Adam.** Each has a different update rule. Without a class, your training loop grows `if optimizer == "sgd": ... elif optimizer == "adam": ...`, and every new optimizer touches the loop.
- **Decay the learning rate.** `p.data -= 0.01 * p.grad` hardcodes `0.01`. When you want to schedule it, you edit the loop again.
- **Reuse the code.** Every experiment repeats the same update block. When you find a bug in it (and you will), you fix it in every script.

An optimizer class solves all three. The training loop becomes:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Three lines, identical for every optimizer, every experiment, every task. Swap `SGD` for `Adam` and the loop does not change. This is the standard pattern in PyTorch, Keras, JAX, and every framework you will ever use. Adopting it now means the code you write next year already matches the ecosystem.

---

## Why SGD first

SGD is the simplest possible update rule:

```
p.data -= lr * p.grad
```

Every parameter is nudged in the direction opposite to its gradient, scaled by the learning rate. That is the entire algorithm.

Starting with SGD matters for three reasons:

1. **It is the baseline.** Every other optimizer is judged against it. If Adam does not beat SGD on your problem, Adam is not earning its complexity.
2. **It has one hyperparameter.** `lr` is the only knob. When something goes wrong, the search space is small.
3. **It exposes the raw behavior of the gradient.** Momentum and Adam smooth the trajectory; SGD shows you exactly what the gradients are doing. When you add them, you will know what they are changing, because you have seen the unmodified version.

The same logic that made us build `engine.py` before `nn.py` applies here: understand the base before adding complexity on top.

---

## Why `step()` does not call `zero_grad()`

You might expect `step()` to reset gradients automatically, since that is almost always what you want next. It does not. This is deliberate, and it is the same choice PyTorch made.

The reason is that gradients **accumulate** by design. If you split a large batch into two mini-batches and call `backward()` twice, the accumulated gradients are exactly the gradients of the full batch. That is a feature: it lets you train on datasets that do not fit in memory by doing multiple backward passes before one update.

If `step()` reset gradients, that pattern would break. You would have to manually save and restore them, or update after every mini-batch, which defeats the purpose.

So the responsibility is explicit: `zero_grad()` clears, `step()` updates. The training loop calls them in the order it wants, and the meaning of each call is unambiguous.

The cost of this design is a footgun: forget to call `zero_grad()` and gradients grow monotonically across steps, the effective learning rate explodes, and the loss goes to `nan`. Every beginner hits this once. The remedy is the same as the one we documented in `engine.md`: **treat `zero_grad()` as non-negotiable at the start of every training step.**

---

## Why `lr` is a plain attribute

`self.lr` is not hidden behind a property or a method. It is just a number you can read and reassign at any time:

```python
optimizer = SGD(model.parameters(), lr=0.01)

if step == 500:
    optimizer.lr = 0.001
```

That one-liner is the basis of learning rate schedules. You do not need a scheduler class, a callback, or a configuration object. You need the attribute to be writable, and it is.

When you add Momentum and Adam, they will have their own attributes (`beta`, `betas`, `eps`). All of them will be public for the same reason: flexibility without ceremony.

---

## Why `list(params)` in the constructor

```python
self.params = list(params)
```

The argument could be any iterable: a list, a generator, a `parameters()` call that returns one. If you store the iterable as-is and it is a generator, the first loop over it consumes it, and the second loop (in `zero_grad`, or in a future `step` that iterates twice) sees nothing.

Converting to a list once at construction time makes the optimizer independent of what was passed in. It is defensive, it is one line, and it removes an entire class of bugs that only manifest when you switch to a new calling pattern.

---

## The order of operations

There is one correct order in the training loop:

```python
optimizer.zero_grad()   # 1. forget the previous step
loss.backward()         # 2. compute the current gradients
optimizer.step()        # 3. update the parameters
```

Every other order has a specific failure mode:

| Wrong order | What happens |
|---|---|
| `backward` → `zero_grad` → `step` | `zero_grad` wipes the gradients before they are used. No learning. |
| `zero_grad` → `step` → `backward` | `step` uses zero gradients. No learning. |
| `step` → `backward` → `zero_grad` | One-step lag. The update at step `N` uses gradients from step `N-1`. |
| Never calling `zero_grad` | Gradients accumulate forever. Loss explodes. |

The correct order is not an accident of this project. It is the same in PyTorch, TensorFlow, and every framework. Learn it once, never question it again.

---

## The whole picture in one paragraph

An optimizer is a small object that holds a list of parameters and a rule for updating them. `step()` applies the rule, `zero_grad()` clears the parameters' gradients, and `lr` is a public attribute you can change at any time. SGD is the first rule because it is the simplest and the baseline against which every other optimizer is measured. The separation between `zero_grad` and `step` is deliberate: it preserves gradient accumulation for mini-batch training and matches the semantics of every major framework. The whole file is 20 lines. Its value is not in the math — it is in the interface, which will not change when Momentum, Adam, or anything else is added on top.