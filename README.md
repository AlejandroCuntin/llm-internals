# LLM Internals

I'm building a language model from scratch — tokenizer, attention, training loop, the works — because I got tired of treating transformers like a black box I just import.

This repo is basically my learning log turned into a project: every phase gets built, tested, and explained in my own words before I move on to the next one. No copy-pasting from tutorials without understanding what's underneath.

## Why

I'm a Computer Engineering student, and most of what I'd done in ML so far was "call the library, get the result." This project is my attempt to actually understand what's happening between the input and the output — the math, the design decisions, the trade-offs — and to have something real to show for it, not just a course certificate.

## Phases

- [ ] **01 — Autograd engine**: backpropagation and computational graphs, built from scratch (no PyTorch yet)
- [ ] **02 — Mini-GPT**: tokenizer, self-attention, transformer blocks, and a full training loop for a small language model
- [ ] **03 — Efficient inference**: LoRA fine-tuning, quantization, and a hand-built KV-cache

Each folder has its own README with details, results, and what I learned along the way — including what didn't work.

## Status

Just getting started. Check the commit history for progress.
