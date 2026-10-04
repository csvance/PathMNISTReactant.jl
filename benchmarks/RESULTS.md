# Training throughput: PyTorch vs ReactantNitro

GPU: NVIDIA RTX A6000 (GPU 2). Throughput is steady state over 100 optimizer steps after 30 warmup steps; `first step` includes compilation.

PyTorch 2.14.0+cu130, Lightning 2.6.6, timm 1.0.30.
Julia 1.12.7, ReactantNitro 0.1.5, Reactant 0.2.290.

## Summary

Steady-state training step at TF32 (PyTorch `tf32`, Reactant's default precision), RTX A6000. "Julia default" is the original code with XLA's defaults; "Julia best" adds LuxLib's `mean_var` Reactant specialization and, for ConvNeXtV2, Luximm's channels-last block (`conv_mlp = false`), all with XLA's default flags (for the ODE, which has no normalization, the best compiler-effort rung). Differences inside a row's run-to-run spread are not real.

| workload | PyTorch eager | PyTorch best | Julia default | Julia best | Julia best vs eager | Julia best vs PyTorch best |
|---|---:|---:|---:|---:|---:|---:|
| ConvNeXt V2 Tiny | 162.5 ms | 79.7 ms (`max-autotune-no-cudagraphs`) | 89.1 ms | 75.7 ms (channels-last, contiguous GRN, `mean_var`) | 2.15x | 1.05x |
| BiT ResNetV2-50x1 | 74.8 ms | 59.7 ms (`max-autotune`) | 72.4 ms | 71.1 ms (`mean_var`) | 1.05x | 0.84x |
| Neural ODE (RK4, 100 steps) | 128.3 ms | 16.1 ms (`reduce-overhead`) | 11.3 ms | 10.6 ms (`max`) | 12.1x | 1.52x |
| CNN -> neural ODE -> classifier | 391.6 ms | 100.3 ms (`max-autotune`) | 101.7 ms | 88.5 ms (channels-last, contiguous GRN, `mean_var`) | 4.42x | 1.13x |

- **Where Julia wins clearly:** the neural ODE. 100 unrolled RK4 steps forward and backward run as one XLA program; PyTorch needs CUDA graphs to get within 1.5x of it, and eager is about 12x slower.
- **The mixed CNN -> neural ODE program is a clear Julia win** (1.13x PyTorch's best mode, with a 318 s compile against `max-autotune`'s 1,420 s; parity, 0.99x, before).
- **ConvNeXt V2 Tiny is now faster than PyTorch's best** (1.05x, from 0.89x): the channels-last block (8.8%), GRN's spatial reduction made contiguous within it (7.1%), and the `mean_var` specialization (1.3%).
- **BiT trails** (0.84x PyTorch's best): the `mean_var` specialization helps (2.4%), but no layout or compiler change has touched the rest of its gap.
- **Compiler effort barely moves Julia.** XLA's defaults are already close to maximum for GEMMs (autotune level 4, Triton GEMM, cuBLASLt, fusion autotuner; CUDA-graph command buffers for everything except convolutions). Capturing convolutions in graphs, exhaustive GEMM tiling search, cuDNN GEMM fusion and Reactant's optional Enzyme-JAX passes are all within noise on the backbones; the only gains are on the ODE (up to 7%, about one spread) and the CNN -> ODE head (1.7%). Forcing NHWC convolutions is 61 to 64% slower on ConvNeXt and neutral on BiT. The remaining backbone gap is therefore not a flag.
- **Single-pass GroupNorm statistics did not help BiT** (section "BiT ResNetV2-50x1: GroupNorm statistics"): Luximm's `ShiftedGroupNorm` is numerically equivalent to Lux's `GroupNorm` but 16% slower in the full training step (83.9 against 72.5 ms), and 66% slower before its shift was detached from the gradient. In XLA's final program it adds about two Triton fusions per GroupNorm (204 against 109) and the optimized StableHLO two transposes per GroupNorm (dumps in `results/dumps/bit/`).
- **The channels-last ConvNeXtV2 block is the largest single gain** (section "Julia with the normalization and layout changes"): timm's `conv_mlp = False` layout, with the same weights, makes ConvNeXt V2 Tiny 8.8% faster (88.20 to 81.06 ms) and the CNN -> ODE program 6.7% faster (99.68 to 93.44 ms). In XLA's program the 1x1 convolutions become Triton GEMM fusions (102, against 3) and cuDNN convolution calls drop from 173 to 65; the extra transposes all fold into neighboring fusions (86 standalone transpose kernels in both layouts). GELU was not fused into the GEMM epilogue (0 of 102 GEMM fusions): GRN's strided spatial reduction made XLA write the 4C activation twice (once transposed), and making that reduction contiguous gave a further 7.1% (81.06 to 75.66 ms). GELU epilogue fusion itself is out of reach in this XLA build: Triton GEMM fusions take ReLU's `maximum` but not `tanh`, `erf`, or an `erf` written as a rational polynomial, and it is worth at most about 0.26 ms per stage-1 block. Dumps in `results/dumps/convnext/`.
- **Detaching μ inside LuxLib's variance is a real, exact gain** (section "Normalization backward"): reverse-mode AD of `var(x; mean = μ)` spends a full-size reduction per normalization on ∂σ²/∂μ, which is identically zero. Detaching it keeps the forward bit-identical and the gradients equal (4.9e-7 relative at strict FP32) and makes BiT 2.4% faster (72.82 to 71.08 ms over four measurements each, non-overlapping ranges); ConvNeXt's LayerNorms 1.3% faster from one pair (89.33 to 88.16 ms, suggestive). Now a `mean_var` method in LuxLib's Reactant extension (local branch `luxlib/reactant-mean-var` of Lux.jl, used here through `[sources]`), so every run from here on includes it; the A/B rows were measured with an equivalent opt-in override that has since been removed. Luximm routes all its normalization statistics, including StdConv's weight standardization, through `mean_var` (branch `perf/mean-var-norms`).

## Status

- Device memory leak: fixed in ReactantNitro 0.1.5 (PR #35, merged). `grad_program` and `opt_program` returned fresh trees, and Reactant donates only arguments a trace mutates, so every superseded accumulator and parameter buffer waited for Julia's GC: +319 MiB per step at `accum = 2` in `prebuilt` mode until OOM. Both programs now write in place; device memory is flat with no GC. `bench.jl` no longer forces a GC per step
- Compiler options reach Reactant through `Nitro(...; compile_options)` (ReactantNitro branch `compile-options`, used here through `[sources]`): Reactant compile keywords or a `Reactant.CompileOptions`, part of each program's compile-cache key. XLA's GPU flags go through `XLA_FLAGS`, which XLA reads once when Reactant loads
- Done: the PyTorch matrices for all four workloads (tf32; fp32 eager for ConvNeXt, BiT and the ODE), Julia default precision for all four, Julia `highest` for the ODE and CNN -> ODE, and the Julia compiler-effort ladder for all four
- Pending: Julia ConvNeXt `prebuilt highest` and `pipeline default`, BiT `prebuilt highest`; a PyTorch fp32 row for CNN -> ODE (Julia `highest` there, 257.8 images/s, has no partner yet). `./run_all.sh` with no arguments reruns everything with the current methodology
- Julia runs on 1.12.7 (the Kaimon gate's version); `julia` compat relaxed to 1.12 and both manifests re-resolved, with no change to the listed package versions

## Findings and caveats

- Run-to-run noise: two steady-state measurements of one configuration differ by up to about 2 ms on the backbones and 0.75 ms on the ODE (about 3% and 7%), which is why the effort tables report the mean and spread of both
- Reactant's experimental `cudnn_hlo_optimize` fails to compile the neural ODE gradient (`MLIR pass pipeline "cudnn-hlo-opt" failed: redefinition of symbol named '__cudnn_fused_elementwise_dot_403'`, Reactant 0.2.290), so the `max` preset leaves it out
- Neural ODE: launch-bound, unlike the vision models. CUDA graphs take PyTorch from 128.3 ms (eager) to 16.1 ms (7.95x), against 61.1 ms (2.1x) for plain `torch.compile`. At strict FP32 (Julia `highest`, PyTorch eager `fp32`): 11.9 ms against 127.1 ms, 10.7x
- CNN -> neural ODE, PyTorch eager: the ODE head adds about 229 ms per step over the same ConvNeXt with a linear head (about 115 ms per micro-batch of 16, close to the standalone ODE's 128 ms at batch 128): eager pays the RK4 launch cost almost independently of batch size. With CUDA graphs it costs about 27 ms per step
- BiT under PyTorch: CUDA graphs add almost nothing over plain `torch.compile` (529.6 vs 523.2 images/s), so compiled PyTorch is not launch-bound at batch 16 on this card, and `channels_last` makes no difference. The XLA side is not launch-bound either: graph capture for convolutions changes nothing on any CNN workload
- Compile times: the Julia effort ladder and every later run compile cold (fresh Inductor and Triton cache directories per PyTorch run, Reactant's persistent cache disabled for this project in `julia/LocalPreferences.toml`). The PyTorch rows and the earlier Julia rows predate that change and have mixed cache states in their first-step columns (`torch.compile` default on the ODE: 110 s cold, 6.6 s warm); steady-state step times are unaffected. Reactant's shared cache saved little: cold Julia compiles came within 7 s of the earlier warm ones

## Notes

- Precision match: Reactant's default `PrecisionConfig.DEFAULT` lets XLA use TF32 on Ampere, so PyTorch `tf32` is the like-for-like row; `fp32` (strict) pairs with Julia `highest`
- PyTorch data: a pool of random UInt8 batches already on the GPU (no loader cost). Julia `prebuilt` is the same condition; Julia `pipeline` is the real memory-mapped loader with augmentation
- Julia `first` pays compilation; `cached` is a fresh Nitro reusing the compiled programs in the same process
- Step counting: both stacks time exactly 100 optimizer steps (200 micro-batches at `accum = 2`) after 30 warmup steps; the Julia scripts refuse to report unless all 130 steps were timed
- `run_all.sh` reruns any or all groups (vision, bit, node, cnn-node, julia-effort) end to end, one process per configuration; the previous results of those groups move to `results/archive/<timestamp>/`

## ConvNeXt V2 Tiny

27,873,417 parameters; 9 classes, batch 16 x 2 accumulation steps, UInt8 input normalized on device, RAdam lr 1e-3, no decay, no clipping.

### PyTorch (Lightning), random batches already on the GPU

| compile | channels_last | precision | cudnn.benchmark | first step (s) | step (ms) | images/s | vs eager | peak mem (GB) |
|---|---|---|---|---:|---:|---:|---:|---:|
| none | False | tf32 | True | 4.5 | 162.53 | 196.9 | 1.00x | 4.23 |
| none | False | fp32 | True | 3.0 | 189.89 | 168.5 | 0.86x | 4.23 |
| none | True | tf32 | True | 4.1 | 163.24 | 196.0 | 1.00x | 4.23 |
| none | True | fp32 | True | 2.9 | 194.75 | 164.3 | 0.83x | 4.23 |
| none | False | tf32 | False | 1.1 | 162.44 | 197.0 | 1.00x | 3.29 |
| default | False | tf32 | True | 43.9 | 80.93 | 395.4 | 2.01x | 3.19 |
| default | True | tf32 | True | 38.5 | 111.78 | 286.3 | 1.45x | 3.18 |
| max-autotune | True | tf32 | True | 167.9 | 109.15 | 293.2 | 1.49x | 3.28 |
| max-autotune-no-cudagraphs | False | tf32 | True | 87.8 | 79.66 | 401.7 | 2.04x | 3.19 |
| max-autotune-no-cudagraphs | True | tf32 | True | 87.2 | 111.26 | 287.6 | 1.46x | 3.18 |

Failed runs: `--compile reduce-overhead --precision tf32`; `--compile reduce-overhead --precision tf32 --channels-last`; `--compile max-autotune --precision tf32`

### Julia (ReactantNitro)

| data | precision | run | threads | first step (s) | step (ms) | images/s | vs PyTorch eager | vs best PyTorch (tf32) | data wait |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| prebuilt | default | first | 16 | 264.0 | 89.58 | 357.2 | 1.81x | 0.89x | 0.2% |
| prebuilt | default | cached | 16 | 0.1 | 89.33 | 358.2 | 1.82x | 0.89x | 0.1% |

## BiT ResNetV2-50x1

23,518,793 parameters (timm's bare linear head, no extra LayerNorm); 9 classes, batch 16 x 2 accumulation steps, UInt8 input normalized on device, RAdam lr 1e-3, no decay, no clipping. Normalization mean and std 0.5.

### PyTorch (Lightning), random batches already on the GPU

| compile | channels_last | precision | cudnn.benchmark | first step (s) | step (ms) | images/s | vs eager | peak mem (GB) |
|---|---|---|---|---:|---:|---:|---:|---:|
| none | False | tf32 | True | 4.8 | 74.84 | 427.6 | 1.00x | 3.96 |
| none | True | tf32 | True | 2.2 | 75.71 | 422.7 | 0.99x | 3.99 |
| default | False | tf32 | True | 48.3 | 61.16 | 523.2 | 1.22x | 3.93 |
| default | True | tf32 | True | 44.2 | 61.10 | 523.8 | 1.22x | 3.88 |
| max-autotune-no-cudagraphs | False | tf32 | True | 1005.1 | 60.32 | 530.5 | 1.24x | 3.93 |
| max-autotune-no-cudagraphs | True | tf32 | True | 137.6 | 60.12 | 532.3 | 1.24x | 3.89 |
| reduce-overhead | False | tf32 | True | 42.1 | 60.42 | 529.6 | 1.24x | 4.01 |
| reduce-overhead | True | tf32 | True | 41.5 | 60.11 | 532.3 | 1.24x | 3.97 |
| max-autotune | False | tf32 | True | 106.5 | 59.71 | 535.9 | 1.25x | 4.01 |

### Julia (ReactantNitro)

| data | precision | run | threads | first step (s) | step (ms) | images/s | vs PyTorch eager | vs best PyTorch (tf32) | data wait |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| prebuilt | default | first | 16 | 311.8 | 71.75 | 446.0 | 1.04x | 0.83x | 0.3% |
| prebuilt | default | cached | 16 | 0.1 | 71.60 | 446.9 | 1.05x | 0.83x | 0.1% |

## Neural ODE (RK4)

An MLP vector field `du/dt = dec(tanh(enc(u)))`, state 32, hidden 256 (16672 parameters), integrated over t in [0, 1] with classical fixed-step RK4, 100 steps; MSE on the final state; batch 128. One step is the forward rollout, the backward through every RK4 stage, and an RAdam update (lr 1e-3). Random inputs already on the device. PyTorch runs a plain training loop (no Lightning); Julia unrolls the loop at trace time (SpiNODENitro's RK4 rule), so the rollout and its Enzyme adjoint are one XLA program. Speedup is PyTorch step time over this row's step time.

### PyTorch

| compile | precision | first step (s) | step (ms) | steps/s | speedup vs eager | peak mem (GB) |
|---|---|---:|---:|---:|---:|---:|
| none | fp32 | 0.4 | 127.075 | 7.9 | 1.01x | 0.077 |
| none | tf32 | 0.4 | 128.259 | 7.8 | 1.00x | 0.077 |
| default | tf32 | 6.6 | 61.110 | 16.4 | 2.10x | 0.077 |
| reduce-overhead | tf32 | 6.4 | 16.123 | 62.0 | 7.96x | 0.068 |
| max-autotune | tf32 | 505.3 | 21.236 | 47.1 | 6.04x | 0.066 |
| max-autotune-no-cudagraphs | tf32 | 6.9 | 24.200 | 41.3 | 5.30x | 0.060 |

### Julia (ReactantNitro)

| precision | run | first step (s) | step (ms) | steps/s | speedup vs PyTorch eager | vs best PyTorch (tf32) |
|---|---|---:|---:|---:|---:|---:|
| default | first | 99.3 | 10.842 | 92.2 | 11.83x | 1.49x |
| default | cached | 0.0 | 10.788 | 92.7 | 11.89x | 1.49x |
| highest | first | 98.5 | 12.564 | 79.6 | 10.21x | 1.28x |
| highest | cached | 0.0 | 11.883 | 84.2 | 10.79x | 1.36x |

## CNN -> neural ODE -> classifier

ConvNeXt V2 Tiny backbone, pooled features -> linear (768 => 64) -> neural ODE (the MLP field above, hidden 256, RK4 with 100 steps) -> linear (64 => 9) logits, cross-entropy; 27,949,385 parameters. A mixed program: a few large convolution kernels, then 400 small vector-field evaluations, forward and backward. 9 classes, batch 16 x 2 accumulation steps, UInt8 input normalized on device, RAdam lr 1e-3, no decay, no clipping.

### PyTorch (Lightning), random batches already on the GPU

| compile | channels_last | precision | cudnn.benchmark | first step (s) | step (ms) | images/s | vs eager | peak mem (GB) |
|---|---|---|---|---:|---:|---:|---:|---:|
| none | False | tf32 | True | 6.0 | 391.62 | 81.7 | 1.00x | 4.23 |
| default | False | tf32 | True | 217.6 | 178.84 | 178.9 | 2.19x | 3.19 |
| reduce-overhead | False | tf32 | True | 146.6 | 107.21 | 298.5 | 3.65x | 3.29 |
| max-autotune | False | tf32 | True | 1420.3 | 100.28 | 319.1 | 3.91x | 3.29 |
| max-autotune-no-cudagraphs | False | tf32 | True | 345.7 | 114.41 | 279.7 | 3.42x | 3.19 |

### Julia (ReactantNitro)

| data | precision | run | threads | first step (s) | step (ms) | images/s | vs PyTorch eager | vs best PyTorch (tf32) | data wait |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| prebuilt | default | first | 16 | 303.6 | 101.83 | 314.3 | 3.85x | 0.98x |  |
| prebuilt | default | cached | 16 | 0.1 | 101.59 | 315.0 | 3.86x | 0.99x |  |
| prebuilt | highest | first | 16 | 304.0 | 124.21 | 257.6 | 3.15x | 0.81x |  |
| prebuilt | highest | cached | 16 | 0.1 | 124.12 | 257.8 | 3.16x | 0.81x |  |

## Julia compiler effort

The analog of PyTorch's compile modes: each workload at the default precision (TF32) under rising compiler effort, every run a cold compile. XLA flags are set through `XLA_FLAGS`, the Reactant preset through `Nitro(...; compile_options)`; the exact flags are in `run_all.sh` and `julia/common.jl`. Each configuration trains twice in one process, and both runs time 100 steady-state steps after warmup: the step is their mean and the spread their difference, a floor on what counts as a real difference between rows. The first step is the cold compile (the first run). "vs best PyTorch" divides PyTorch's fastest tf32 step for the same workload by this row's step.

### Neural ODE (RK4)

| config | Reactant preset | first step (s) | step (ms) | spread (ms) | vs Julia default | vs best PyTorch (tf32) |
|---|---|---:|---:|---:|---:|---:|
| default | default | 100.0 | 11.282 | 0.743 | 1.00x | 1.43x |
| xla-graphs | default | 101.6 | 11.040 | 0.192 | 1.02x | 1.46x |
| xla-max | default | 133.6 | 10.887 | 0.751 | 1.04x | 1.48x |
| max | max | 134.2 | 10.584 | 0.702 | 1.07x | 1.52x |

### CNN -> neural ODE -> classifier

| config | Reactant preset | first step (s) | step (ms) | spread (ms) | vs Julia default | vs best PyTorch (tf32) |
|---|---|---:|---:|---:|---:|---:|
| default | default | 310.8 | 101.671 | 1.086 | 1.00x | 0.99x |
| xla-graphs | default | 308.6 | 101.382 | 0.019 | 1.00x | 0.99x |
| xla-max | default | 317.4 | 99.930 | 0.592 | 1.02x | 1.00x |
| max | max | 318.9 | 100.858 | 0.877 | 1.01x | 0.99x |
| max-nhwc | max | 319.9 | 158.083 | 0.606 | 0.64x | 0.63x |

### ConvNeXt V2 Tiny

| config | Reactant preset | first step (s) | step (ms) | spread (ms) | vs Julia default | vs best PyTorch (tf32) |
|---|---|---:|---:|---:|---:|---:|
| default | default | 271.8 | 89.121 | 0.065 | 1.00x | 0.89x |
| xla-graphs | default | 272.4 | 88.963 | 1.437 | 1.00x | 0.90x |
| xla-max | default | 271.5 | 88.893 | 0.181 | 1.00x | 0.90x |
| max | max | 271.0 | 88.886 | 0.325 | 1.00x | 0.90x |
| max-nhwc | max | 270.8 | 146.429 | 0.827 | 0.61x | 0.54x |

### BiT ResNetV2-50x1

| config | Reactant preset | first step (s) | step (ms) | spread (ms) | vs Julia default | vs best PyTorch (tf32) |
|---|---|---:|---:|---:|---:|---:|
| default | default | 310.6 | 72.424 | 0.484 | 1.00x | 0.82x |
| xla-graphs | default | 309.3 | 72.960 | 0.337 | 0.99x | 0.82x |
| xla-max | default | 308.9 | 72.737 | 1.490 | 1.00x | 0.82x |
| max | max | 309.4 | 73.771 | 2.028 | 0.98x | 0.81x |
| max-nhwc | max | 307.5 | 71.733 | 0.389 | 1.01x | 0.83x |


Failed runs: `julia effort node max (exit 1 with the first max preset: the experimental cudnn_hlo_optimize pass failed to compile; the max row is the rerun without it)`

## BiT ResNetV2-50x1: GroupNorm statistics

Lux's `GroupNorm` (statistics from `mean`, then `var` around it: two dependent reductions, so XLA reads each activation three times) against Luximm's `ShiftedGroupNorm` (`norm_impl = :shifted`: shifted sums, two independent reductions XLA can merge into one pass). Same parameters and weights; at strict FP32 the logits agree to 3e-6 relative and the gradients to 0.12% relative L2, against 2.9% for TF32 noise alone. Processes interleave the two implementations; each gives two steady-state measurements, all of which are averaged. The shift is detached from the gradient (its gradient is exactly zero; the gradient through the mean and variance is intact). A first version with the shift attached measured 121.4 ms against 73.0 ms (0.60x; results in `results/archive/`): the reverse pass of the slice scattered into a full-size tensor at every GroupNorm. In a conv -> GroupNorm -> ReLU -> conv microbenchmark at 56x56x256x16 the detached version is level with Lux (4.908 against 4.893 ms forward and backward), and 11% faster on the GroupNorm alone, but in the full model it is still slower: the expected saving does not materialize.

| precision | GroupNorm | measurements | first step (s) | step (ms) | spread (ms) | images/s | vs Lux GroupNorm | vs best PyTorch (tf32) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| default | lux | 2 | 308.4 | 72.48 | 0.72 | 441.5 | 1.000x | 0.82x |
| default | shifted | 2 | 313.3 | 83.90 | 0.79 | 381.4 | 0.864x | 0.71x |

## Normalization backward: μ detached inside the variance

LuxLib's `mean_var` computes `var(x; mean = μ)`; Enzyme's reverse pass then spends a full-size reduction per normalization on ∂σ²/∂μ, which is identically zero. `BENCH_MEANVAR=stopgrad` detaches μ inside the variance only (an opt-in override for traced arrays in `julia/common.jl`): the forward is bit-identical and the gradient to x is unchanged (BiT at strict FP32: logits identical, gradients 4.9e-7 relative L2). It reaches every LuxLib normalization, GroupNorm in BiT and LayerNorm in ConvNeXt. Default precision (TF32); processes interleave the two modes and each gives two steady-state measurements.

| backbone | mean_var | measurements | step (ms) | spread (ms) | images/s | vs default |
|---|---|---:|---:|---:|---:|---:|
| resnetv2_50x1_bit_goog_in21k | default | 4 | 72.82 | 0.62 | 439.4 | 1.000x |
| resnetv2_50x1_bit_goog_in21k | stopgrad | 4 | 71.08 | 0.56 | 450.2 | 1.024x |
| convnextv2_tiny_fcmae_ft_in22k_in1k | default | 2 | 89.33 | 0.14 | 358.2 | 1.000x |
| convnextv2_tiny_fcmae_ft_in22k_in1k | stopgrad | 2 | 88.16 | 0.28 | 363.0 | 1.013x |

## Julia with the normalization and layout changes

Every run here includes LuxLib's `mean_var` Reactant specialization (Lux.jl branch `luxlib/reactant-mean-var`) and Luximm's StdConv through `mean_var` (branch `perf/mean-var-norms`). `conv_mlp = false` is Luximm's channels-last ConvNeXtV2 block (branch `perf/convnextv2-channels-last`): LayerNorm over a contiguous channel axis and GEMM pointwise layers, same weights; config `conv-mlp-false-grn-rows` adds its GRN with the spatial sum of squares made a contiguous reduction. Default precision (TF32); step is the mean of the two steady-state measurements of one process. "vs best PyTorch" divides PyTorch's fastest tf32 step for the same workload by this row's step.

| workload | backbone | conv_mlp | config | first step (s) | step (ms) | images/s | vs best PyTorch (tf32) |
|---|---|---|---|---:|---:|---:|---:|
| vision | resnetv2_50x1_bit_goog_in21k |  | luxlib-mean-var | 308.0 | 71.09 | 450.1 | 0.84x |
| vision | convnextv2_tiny_fcmae_ft_in22k_in1k | True | luxlib-mean-var | 270.3 | 88.20 | 362.9 | 0.90x |
| cnn-node-rk4 | convnextv2_tiny_fcmae_ft_in22k_in1k | True | luxlib-mean-var | 308.3 | 99.68 | 321.0 | 1.01x |
| vision | convnextv2_tiny_fcmae_ft_in22k_in1k | False | conv-mlp-false | 274.5 | 81.06 | 394.8 | 0.98x |
| cnn-node-rk4 | convnextv2_tiny_fcmae_ft_in22k_in1k | False | conv-mlp-false | 314.2 | 93.44 | 342.4 | 1.07x |
| vision | convnextv2_tiny_fcmae_ft_in22k_in1k | False | conv-mlp-false-grn-rows | 276.4 | 75.66 | 423.0 | 1.05x |
| cnn-node-rk4 | convnextv2_tiny_fcmae_ft_in22k_in1k | False | conv-mlp-false-grn-rows | 317.7 | 88.53 | 361.4 | 1.13x |
| vision | resnetv2_50x1_bit_goog_in21k |  | nitro-sync-readback | 308.6 | 70.74 | 452.4 | 0.84x |
| vision | resnetv2_50x1_bit_goog_in21k |  | nitro-async-readback | 310.5 | 69.03 | 463.6 | 0.87x |
| vision | convnextv2_tiny_fcmae_ft_in22k_in1k | False | nitro-sync-readback-grn-rows | 276.9 | 75.61 | 423.2 | 1.05x |
| vision | convnextv2_tiny_fcmae_ft_in22k_in1k | False | nitro-async-readback-grn-rows | 276.4 | 74.12 | 431.8 | 1.07x |
| node-rk4 | node |  | nitro-sync-readback | 98.9 | 10.61 | 12086.0 | 1.52x |
| node-rk4 | node |  | nitro-async-readback | 100.8 | 11.20 | 11450.9 | 1.44x |
| vision | resnetv2_50x1_bit_goog_in21k |  | nitro-perleaf-accum | 344.6 | 69.41 | 461.0 | 0.86x |
| vision | convnextv2_tiny_fcmae_ft_in22k_in1k | False | nitro-perleaf-accum-grn-rows | 334.8 | 74.79 | 427.9 | 1.07x |
