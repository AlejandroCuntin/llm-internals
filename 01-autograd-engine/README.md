# Phase 1: Autograde Engine (`engine.py`)

This folder contains "engine.py", the matematical heart of our neural network. Instead of just calculating numbers, 
it creates "numbers with memory" that remember how they were created.

## What is `engine.py`

In standar Python , if you do `c = a + b`, the computer forgets `a` and `b` immediately. Our "engine.py" implements a custom "Value" class.
When you add or multiply "Value" objects, they keep a receipt of the operation. This chain of receipts is called a **Computational Graph**,
and it is essential for the AI to learn from its mistakes

## Core Functions 

### `__init__` (The memory)
Initializes the value and its gradient (set to 0.0). It also creates an empty set `prev´ to store the "parents" of this number

### `__add__` & `__mul__` (The Operations)

They perform standard addition and multiplication, but with a twist: they record the operation and link new result to its original inputs.
This builds the graph dynamically.

### `_backward` (The Time Machine)
This function travels backward through the recorded receipts. It calculates the **gradient** (using the Chain Rule), which is simply a compass
that tells the neural network: *"Should this parameter go up or down to reduce the error?*
