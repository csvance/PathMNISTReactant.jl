# PathMNISTReactant.jl

Companion code for [Replacing PyTorch with Julia and Reactant: One Year In](https://csvance.dev/posts/2026-10-22-reactant-end-to-end/): a ConvNeXt V2 backbone with a NeuralODE head, trained on [PathMNIST](https://medmnist.com) with [ReactantNitro.jl](https://github.com/EnzymeAD/ReactantNitro.jl), exported to StableHLO, and served with [ReactantServer.jl](https://github.com/EnzymeAD/ReactantServer.jl).

## Quick start

```bash
juliaup add 1.13.1
git clone https://github.com/csvance/PathMNISTReactant.jl
cd PathMNISTReactant.jl
julia +1.13.1 --project=. -e 'using Pkg; Pkg.instantiate()'
```

```julia
using PathMNISTReactant, ReactantNitro

nitro = Nitro(PathMNIST(); run_dir = "runs/pathmnist", restore_best = true)
train!(nitro)                    # downloads PathMNIST on first use
evaluate(nitro; split = :test)
```

The post walks through each step, including export and serving.

## Contents

| Path | What |
|---|---|
| `src/experiment.jl` | The `@experiment`: every setting and how it reaches the compiled program |
| `src/model.jl`, `src/node.jl` | The backbone (from [Luximm.jl](https://github.com/csvance/Luximm.jl)) and the NeuralODE head, RK4 |
| `src/train.jl` | Forward, loss and metrics hooks |
| `src/optim.jl` | Optimizer, schedule, EMA, gradient clipping and the monitored metric |
| `src/data.jl`, `src/dataset.jl`, `src/augment.jl` | Download, memory-mapped splits and augmentation |
| `src/export.jl` | Export hooks for a ReactantServer bundle |
| `client/` | A KServe V2 client that streams the test set to a running server |
| `scripts/trajectories.jl` | Records p(t) along the ODE for the trajectory figures |
| `blog/` | Everything behind the post's figures and numbers: the seed sweep, data dump and figure scripts ([blog/README.md](blog/README.md)) |
| `benchmarks/` | Training-step benchmarks against PyTorch eager and `torch.compile` |
| `test/` | Tests (`] test`) |

## License

MIT, see [LICENSE](LICENSE).
