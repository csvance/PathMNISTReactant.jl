"""Training-step throughput for the PyTorch side of the comparison.

The same model as the Julia experiment (ConvNeXt V2 Tiny by default, or BiT
ResNetV2-50 with --model; a 9-class head),
the same work per optimizer step (batch 16, 2 accumulation steps, UInt8 input
normalized inside the model, RAdam at lr 1e-3, no decay, no clipping), trained
with PyTorch Lightning. Batches are random and already on the GPU, so the data
pipeline costs nothing: the most favorable condition for this side.

    uv run python bench_lightning.py --compile max-autotune --channels-last

Prints one JSON line per run.
"""

import argparse
import time

import lightning as L
import timm
import torch
import torch.nn.functional as F
from torch import nn

from common import ODEFunc, emit, persist_grads, rk4, set_precision, uses_cudagraphs

NUM_CLASSES = 9
# timm model name -> the per-channel (mean, std) its pretrained weights expect.
MODELS = {
    "convnextv2_tiny.fcmae_ft_in22k_in1k": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "resnetv2_50x1_bit.goog_in21k": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
}


class ODEHead(nn.Module):
    """Pooled features -> linear -> neural ODE (tanh MLP field, fixed-step RK4 over
    t in [0, 1]) -> linear classifier. The same head as benchmarks/julia/bench_cnn_node.jl."""

    def __init__(self, features, dims, hidden, steps):
        super().__init__()
        self.proj = nn.Linear(features, dims)
        self.f = ODEFunc(dims, hidden)
        self.cls = nn.Linear(dims, NUM_CLASSES)
        self.steps = steps

    def forward(self, feats):
        return self.cls(rk4(self.f, self.proj(feats), self.steps))


class Net(nn.Module):
    """UInt8 pixels in, logits out: conversion and normalization run on device. With
    node_steps > 0 the classifier head is replaced by an ODEHead on the pooled features."""

    def __init__(self, model, node_steps=0, node_dims=64, node_hidden=256):
        super().__init__()
        mean, std = MODELS[model]
        if node_steps > 0:
            # num_classes=0: timm's pooled (and, for ConvNeXt, head-normed) features.
            self.backbone = timm.create_model(model, pretrained=False, num_classes=0)
            self.head = ODEHead(self.backbone.num_features, node_dims, node_hidden, node_steps)
        else:
            self.backbone = timm.create_model(model, pretrained=False, num_classes=NUM_CLASSES)
            self.head = nn.Identity()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        x = x.float().div(255.0)
        return self.head(self.backbone((x - self.mean) / self.std))


class Lit(L.LightningModule):
    def __init__(self, net, cudagraphs=False):
        super().__init__()
        self.net = net
        self.cudagraphs = cudagraphs

    # CUDA-graph modes replay into fixed output buffers. With gradient
    # accumulation, a .grad created by the first micro-batch would alias one,
    # and the second replay would overwrite it. Persistent .grad buffers make
    # backward accumulate into them in place instead.
    def on_train_start(self):
        if self.cudagraphs:
            persist_grads(self)

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=not self.cudagraphs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return F.cross_entropy(self.net(x), y)

    def configure_optimizers(self):
        return torch.optim.RAdam(self.parameters(), lr=1e-3, foreach=True)


class GPUBatches(torch.utils.data.Dataset):
    """A small pool of random batches built once on the GPU, cycled."""

    def __init__(self, n_items, batch_size, channels_last, pool=32):
        fmt = torch.channels_last if channels_last else torch.contiguous_format
        self.pool = [
            (
                torch.randint(0, 256, (batch_size, 3, 224, 224), dtype=torch.uint8, device="cuda")
                .contiguous(memory_format=fmt),
                torch.randint(0, NUM_CLASSES, (batch_size,), device="cuda"),
            )
            for _ in range(pool)
        ]
        self.n_items = n_items

    def __len__(self):
        return self.n_items

    def __getitem__(self, i):
        return self.pool[i % len(self.pool)]


class StepTimer(L.Callback):
    """Times optimizer steps: the first (compile lands here) and a window after warmup."""

    def __init__(self, accum, warmup, measure):
        self.accum, self.warmup, self.measure = accum, warmup, measure
        self.t_start = self.t_first = self.t0 = self.t1 = None

    def on_train_start(self, trainer, pl_module):
        torch.cuda.synchronize()
        self.t_start = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (batch_idx + 1) % self.accum:
            return
        step = (batch_idx + 1) // self.accum
        if step in (1, self.warmup, self.warmup + self.measure):
            torch.cuda.synchronize()
            now = time.perf_counter()
            if step == 1:
                self.t_first = now
            if step == self.warmup:
                self.t0 = now
            if step == self.warmup + self.measure:
                self.t1 = now


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="convnextv2_tiny.fcmae_ft_in22k_in1k", choices=list(MODELS))
    p.add_argument("--node-steps", type=int, default=0,
                   help="RK4 steps of a neural ODE head on the pooled features (0: linear head)")
    p.add_argument("--node-dims", type=int, default=64)
    p.add_argument("--node-hidden", type=int, default=256)
    p.add_argument("--compile", default="none",
                   choices=["none", "default", "reduce-overhead", "max-autotune",
                            "max-autotune-no-cudagraphs"])
    p.add_argument("--channels-last", action="store_true")
    p.add_argument("--precision", default="tf32", choices=["fp32", "tf32", "bf16-mixed"])
    p.add_argument("--no-cudnn-benchmark", action="store_true")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--warmup", type=int, default=30, help="optimizer steps before timing")
    p.add_argument("--measure", type=int, default=100, help="optimizer steps timed")
    a = p.parse_args()

    set_precision(a.precision)

    net = Net(a.model, a.node_steps, a.node_dims, a.node_hidden)
    if a.channels_last:
        net = net.to(memory_format=torch.channels_last)
    if a.compile != "none":
        net = torch.compile(net, mode=None if a.compile == "default" else a.compile, fullgraph=True)

    steps = a.warmup + a.measure
    data = torch.utils.data.DataLoader(
        GPUBatches(steps * a.accum, a.batch_size, a.channels_last), batch_size=None
    )
    timer = StepTimer(a.accum, a.warmup, a.measure)
    trainer = L.Trainer(
        accelerator="gpu", devices=1, max_steps=steps,
        accumulate_grad_batches=a.accum,
        precision="bf16-mixed" if a.precision == "bf16-mixed" else "32-true",
        benchmark=not a.no_cudnn_benchmark,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, num_sanity_val_steps=0, limit_val_batches=0,
        callbacks=[timer],
    )
    torch.cuda.reset_peak_memory_stats()
    cudagraphs = uses_cudagraphs(a.compile)
    trainer.fit(Lit(net, cudagraphs), train_dataloaders=data)

    images = a.measure * a.accum * a.batch_size
    dt = timer.t1 - timer.t0
    emit({
        "stack": "pytorch-lightning",
        "model": a.model,
        "node_steps": a.node_steps,
        "params": sum(q.numel() for q in net.parameters()),
        "compile": a.compile,
        "channels_last": a.channels_last,
        "precision": a.precision,
        "cudnn_benchmark": not a.no_cudnn_benchmark,
        "batch_size": a.batch_size,
        "accum": a.accum,
        "first_step_s": round(timer.t_first - timer.t_start, 2),
        "step_ms": round(1000 * dt / a.measure, 2),
        "images_per_s": round(images / dt, 1),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "torch": torch.__version__,
        "lightning": L.__version__,
        "timm": timm.__version__,
        "gpu": torch.cuda.get_device_name(0),
    })


if __name__ == "__main__":
    main()
