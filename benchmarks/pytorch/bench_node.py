"""Training-step throughput for a neural ODE, the PyTorch side.

A pure MLP vector field, du/dt = dec(tanh(enc(u))), integrated over t in [0, 1]
with classical fixed-step RK4 (100 steps by default, common.rk4, the same rule as
the Julia side), then a mean squared error on the final state. One training step
is the forward rollout, the backward through every RK4 stage, and an RAdam update
(lr 1e-3, foreach). A plain training loop, not Lightning, so no framework overhead
is charged to this side; inputs are random and already on the GPU.

    uv run python bench_node.py --compile reduce-overhead

Prints one JSON line per run (see common.emit).
"""

import argparse
import time

import torch
import torch.nn.functional as F
from torch import nn

from common import ODEFunc, emit, persist_grads, rk4, set_precision, uses_cudagraphs


class NODE(nn.Module):
    def __init__(self, dims, hidden, steps):
        super().__init__()
        self.f = ODEFunc(dims, hidden)
        self.steps = steps

    def forward(self, u):
        return rk4(self.f, u, self.steps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--compile", default="none",
                   choices=["none", "default", "reduce-overhead", "max-autotune",
                            "max-autotune-no-cudagraphs"])
    p.add_argument("--precision", default="tf32", choices=["fp32", "tf32"])
    p.add_argument("--dims", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--steps", type=int, default=100, help="RK4 steps over t in [0, 1]")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--warmup", type=int, default=30, help="optimizer steps before timing")
    p.add_argument("--measure", type=int, default=100, help="optimizer steps timed")
    a = p.parse_args()

    set_precision(a.precision)
    torch.manual_seed(0)
    model = NODE(a.dims, a.hidden, a.steps).cuda()
    opt = torch.optim.RAdam(model.parameters(), lr=1e-3, foreach=True)
    run = model if a.compile == "none" else torch.compile(
        model, mode=None if a.compile == "default" else a.compile)
    cudagraphs = uses_cudagraphs(a.compile)
    if cudagraphs:
        persist_grads(model)

    pool = [(torch.randn(a.batch_size, a.dims, device="cuda"),
             torch.randn(a.batch_size, a.dims, device="cuda")) for _ in range(8)]

    def step(i):
        u0, y = pool[i % len(pool)]
        loss = F.mse_loss(run(u0), y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=not cudagraphs)
        return loss

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    step(0)
    torch.cuda.synchronize()
    t_first = time.perf_counter()
    for i in range(1, a.warmup):
        step(i)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(a.warmup, a.warmup + a.measure):
        loss = step(i)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss {loss.item()}")

    dt = t1 - t0
    emit({
        "stack": "pytorch",
        "workload": "node-rk4",
        "compile": a.compile,
        "precision": a.precision,
        "dims": a.dims, "hidden": a.hidden, "steps": a.steps, "batch_size": a.batch_size,
        "params": sum(q.numel() for q in model.parameters()),
        "first_step_s": round(t_first - t_start, 2),
        "step_ms": round(1000 * dt / a.measure, 3),
        "steps_per_s": round(a.measure / dt, 1),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
    })


if __name__ == "__main__":
    main()
