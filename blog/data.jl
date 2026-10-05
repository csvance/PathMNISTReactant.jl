# GPU stage of the blog assets: everything the post's figures and numbers are made from, dumped
# from a trained handle into one directory, so the CPU stage (`figures.jl`) can redraw every figure
# without a GPU. Run it in the session that trained, or after loading a checkpoint:
#
#   include("blog/data.jl")
#   nitro = Nitro(PathMNIST(; data_dir); run_dir, weights = :latest => :best)   # or the trained handle
#   blog_data(nitro, "blog/out")
#
# Writes to <out>:
#   test_probs.f32     class probabilities on the test split, (9, N) column-major Float32
#   traj_P.f32         p(t) along the ODE for 256 test tiles, (9, 256, 51) Float32
#   traj_picks.i64     those tiles' 1-based test indices (randperm with Xoshiro(1), as before)
#   handle.txt         `show` of the handle            (the "handle after training" block)
#   history.txt        `history(nitro)`                (the per-epoch table)
#   evaluate.txt       `evaluate(nitro; split = :test)` (the test metrics line)
using PathMNISTReactant, ReactantNitro, Reactant, Lux, Random
using PathMNISTReactant: PathMNISTSplit
using PathMNISTReactant.MLUtils: DataLoader, getobs

include(joinpath(@__DIR__, "..", "scripts", "trajectories.jl"))   # class_trajectory

softmax_cols(z) = (e = exp.(z .- maximum(z; dims = 1)); e ./ sum(e; dims = 1))

# The terminal width the tables were captured at, so the text blocks keep their layout.
capture(f) = sprint(f; context = IOContext(stdout, :displaysize => (60, 104), :color => false)) * "\n"

function blog_data(nitro, out; n_traj = 256)
    mkpath(out)
    e = nitro.e
    write(joinpath(out, "handle.txt"), capture(io -> show(io, MIME"text/plain"(), nitro)))
    write(joinpath(out, "history.txt"), capture(io -> show(io, MIME"text/plain"(), history(nitro))))
    test_metrics = evaluate(nitro; split = :test)
    write(joinpath(out, "evaluate.txt"), capture(io -> show(io, test_metrics)))

    # Test probabilities through the handle's compiled forward: no new program.
    test = PathMNISTSplit(e, "test")
    loader = DataLoader(test; batchsize = e.batch_size, partial = true)
    logits = reduce(hcat, (Array(predict(nitro, b).logits) for b in loader))
    write(joinpath(out, "test_probs.f32"), Float32.(softmax_cols(Float64.(logits))))

    # p(t) along the trajectory: one extra compiled program, the same picks as before.
    idx = randperm(Xoshiro(1), length(test.labels))[1:n_traj]
    batch = getobs(test, idx)
    img = Reactant.to_rarray(batch.img)
    traj = @compile class_trajectory(e, nitro.model, nitro.ps, nitro.st, img)
    P = Array(traj(e, nitro.model, nitro.ps, nitro.st, img))
    write(joinpath(out, "traj_P.f32"), Float32.(P))
    write(joinpath(out, "traj_picks.i64"), Int64.(idx))
    return (; test_metrics, out)
end
