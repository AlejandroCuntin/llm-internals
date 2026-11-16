# LLM Internals

Proyecto de aprendizaje para entender cómo funcionan las redes neuronales construyendo sus componentes desde cero.

## Contenido

El repositorio incluye `01-autograd-engine`, un motor de diferenciación automática implementado en Python puro. Calcula gradientes mediante retropropagación y permite entrenar redes neuronales pequeñas sin usar PyTorch ni TensorFlow.

La carpeta incluye:

- `engine.py`: la clase `Value`, que representa un número, registra las operaciones que lo producen y calcula sus gradientes.
- `nn.py`: componentes para crear neuronas, capas y redes MLP.
- `optim.py`: los optimizadores SGD, Momentum y Adam.
- `viz.py`: visualización de grafos computacionales con Graphviz.
- `nn_viz.py`: diagramas de la arquitectura de una red con Matplotlib.
- `tests/`: pruebas del motor, las redes y los optimizadores.
- `examples/`: ejemplos de entrenamiento, comparación de optimizadores y visualización del flujo de gradientes.

## Ejemplos

Ejecuta estos comandos desde `01-autograd-engine`:

```bash
python examples/train_moons.py
python examples/compare_optimizers.py
python examples/hyperparameter_sweep.py
python examples/mnist_scalar.py
python examples/walkthrough_visual.py
python examples/gradient_evolution.py
