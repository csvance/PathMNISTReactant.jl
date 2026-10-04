"""Shared by the PyTorch benchmarks: the neural ODE field and its RK4 rule, the
precision switch, and the JSON line each run emits.

Output: each JSON line is printed and, when BENCH_OUT names a file, appended to it.
"""

import json
import os

import torch
from torch import nn


class ODEFunc(nn.Module):
    """du/dt = dec(tanh(enc(u))): a pure MLP vector field."""

    def __init__(self, dims, hidden):
        super().__init__()
        self.enc = nn.Linear(dims, hidden)
        self.dec = nn.Linear(hidden, dims)

    def forward(self, u):
        return self.dec(torch.tanh(self.enc(u)))


def rk4(f, u, steps):
    """Classical four-stage RK4 with `steps` uniform steps over t in [0, 1]; the
    field is autonomous. The same rule as benchmarks/julia/common.jl."""
    dt = 1.0 / steps
    for _ in range(steps):
        k1 = f(u)
        k2 = f(u + 0.5 * dt * k1)
        k3 = f(u + 0.5 * dt * k2)
        k4 = f(u + dt * k3)
        u = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return u


def set_precision(precision):
    """fp32: strict FP32. tf32: TensorFloat-32 for matmuls and convolutions (the match
    for Reactant's default precision on Ampere). bf16-mixed is Lightning's own setting."""
    tf32 = precision != "fp32"
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.set_float32_matmul_precision("high" if tf32 else "highest")


def persist_grads(module):
    """CUDA-graph modes replay into fixed buffers: a .grad created by the first
    backward would alias a graph output and the next replay would overwrite it.
    Persistent .grad buffers, zeroed in place, make backward accumulate into them."""
    for p in module.parameters():
        p.grad = torch.zeros_like(p)


def uses_cudagraphs(compile_mode):
    return compile_mode in ("reduce-overhead", "max-autotune")


def emit(fields):
    line = json.dumps(fields)
    print(line, flush=True)
    out = os.environ.get("BENCH_OUT")
    if out:
        with open(out, "a") as f:
            f.write(line + "\n")
