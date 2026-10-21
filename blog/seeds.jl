# The five seeds behind the post's mean ± std: the package's recipe as it stands, trained one
# after another in one process, so seeds 1-4 reuse the programs seed 0 compiled. Each run keeps
# only its best checkpoint. Seed 0 is the post's model, and its blog data is dumped to blog/out.
#
#   julia -t 8,1 --project=<env with PathMNISTReactant, CairoMakie> blog/seeds.jl <data_dir>
using PathMNISTReactant, ReactantNitro
include(joinpath(@__DIR__, "data.jl"))      # blog_data

const DATA = ARGS[1]
const RESULTS = joinpath(@__DIR__, "seed_results.txt")

for seed in 0:4
    t0 = time()
    nitro = Nitro(
        PathMNIST(; data_dir = DATA, seed); run_dir = "runs/pathmnist_s$seed", restore_best = true,
        checkpointer = Checkpointer(; keep = Best(1))
    )
    train!(nitro)
    wall = round(Int, time() - t0)
    test = evaluate(nitro; split = :test)
    open(io -> println(io, "seed=$seed wall=$wall ", test), RESULTS, "a")
    seed == 0 && blog_data(nitro, joinpath(@__DIR__, "out"))
end
