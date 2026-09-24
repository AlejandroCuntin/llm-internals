02-mini-gpt/
├── README.md
├── docs/
│   ├── tokenizer.md
│   ├── attention.md
│   ├── transformer.md
│   ├── model.md
│   └── training.md
├── src/
│   ├── __init__.py
│   ├── tokenizer.py       ← BPE tokenizer desde cero
│   ├── attention.py       ← self-attention y multi-head attention
│   ├── transformer.py     ← bloque transformer (attention + MLP + layer norm + residuals)
│   ├── model.py           ← GPT completo (embeddings + stack de bloques + head)
│   ├── train.py           ← loop de entrenamiento
│   └── generate.py        ← generación (greedy, top-k, top-p)
├── data/
│   └── input.txt          ← corpus pequeño (tiny shakespeare, por ejemplo)
├── examples/
│   ├── train_tiny_shakespeare.py
│   └── generate_sample.py
├── assets/
│   └── loss_curve.png
└── tests/
    ├── test_tokenizer.py
    ├── test_attention.py       ← comparar contra torch.nn.MultiheadAttention
    └── test_model_shapes.py    ← sanity-check de shapes en el forward pass

03-efficient-inference/

03-efficient-inference/
├── README.md
├── docs/
│   ├── lora.md
│   ├── quantization.md
│   └── kv_cache.md
├── src/
│   ├── __init__.py
│   ├── lora.py           ← capas LoRA e inyección en linear layers
│   ├── quantize.py       ← cuantización int8/int4
│   ├── kv_cache.py       ← KV-cache propio para generación autoregresiva
│   └── benchmark.py      ← medición de latencia/memoria antes vs después
├── examples/
│   ├── finetune_lora.py       ← fine-tuning sobre un modelo pequeño real (GPT-2, Llama 3.2 1B)
│   ├── quantize_and_compare.py
│   └── benchmark_kv_cache.py
├── assets/
│   └── benchmark_results.png
└── tests/
    ├── test_lora.py
    ├── test_quantize.py
    └── test_kv_cache.py