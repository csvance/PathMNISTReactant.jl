# Blog post assets

Everything the blog post shows from the trained model is produced here, in two stages: a GPU stage
that dumps predictions from a trained handle, and a CPU stage that turns the dump into every
figure, the data the post's interactive figures read, and a report of every number the post
quotes. The CPU stage needs no GPU and no Reactant, so the figures can be redrawn as often as the
post changes.

## 1. GPU stage: dump from a trained model

In the session that trained (or after loading a checkpoint):

```julia
using PathMNISTReactant, ReactantNitro
include("blog/data.jl")
nitro = Nitro(PathMNIST(; data_dir); run_dir, weights = :latest => :best, load_ema = true)   # or the trained handle
blog_data(nitro, "blog/out")
```

It writes the test-set class probabilities, p(t) along the NeuralODE for 256 test tiles
(`scripts/trajectories.jl` compiles that one extra program), and the text of the handle,
`history(nitro)` and `evaluate(nitro; split = :test)` as shown in the post.

## 2. CPU stage: figures and data files

```bash
julia --project=<env with CairoMakie> blog/figures.jl blog/out <data_dir> <post>/figures
```

Each static figure is drawn twice, for the site's light and dark themes (`name.png`,
`name-dark.png`). Also written: `trajectories.json` (the trajectory player), `confusion.json`
(the interactive confusion matrix), and `hero/tiles.jpg` with `hero/tiles.json` (the banner).
`blog/out/numbers.md` lists the metrics, error counts, per-class recall, largest confusions and
trajectory decision times, so the prose can be checked against the model it describes.

`ffmpeg` encodes the banner's tile sprite.
