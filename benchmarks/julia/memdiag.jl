# Device-memory diagnostic: bytes in use after each optimizer step, prebuilt
# batches, 12 steps. Usage (GPU 2 only):
#   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
#     julia +1.13.1 -t 16,1 --project=benchmarks/julia benchmarks/julia/memdiag.jl [accum]

using ReactantNitro, Reactant, Random
include(joinpath(@__DIR__, "vision.jl"))
const ACCUM = parse(Int, get(ARGS, 1, "2"))
const STEPS = 12
e = Vision(; accum = ACCUM)
struct MemLog
    rows::Vector{Any}
end
function ReactantNitro.log_metrics!(m::MemLog, metrics; context = "train", step = 0, kwargs...)
    context == "train" && push!(m.rows, (step, Reactant.XLA.allocatorstats().bytes_in_use))
    return nothing
end
for f in (:log_params!, :log_tags!)
    @eval ReactantNitro.$f(::MemLog, x) = nothing
end
ReactantNitro.log_other!(::MemLog, k, v) = nothing
ReactantNitro.log_confusion!(::MemLog, m, l; kw...) = nothing
ReactantNitro.finish!(::MemLog, s) = nothing
ReactantNitro.run_id(::MemLog) = nothing
ReactantNitro.run_url(::MemLog) = nothing
ReactantNitro.reattach!(::MemLog, s) = nothing
rng = Xoshiro(0)
pool = [(; img = rand(rng, UInt8, 224, 224, 3, e.batch_size), label = rand(rng, 1:9, e.batch_size)) for _ in 1:8]
data = (; train = [pool[mod1(i, 8)] for i in 1:(STEPS * ACCUM)])
m = MemLog(Any[])
nitro = Nitro(e; data, max_epochs = 1, logger = m, run_dir = mktempdir())
train!(nitro)
println("accum=$ACCUM  flat param buffer = $(round(27_873_417 * 4 / 2^20; digits = 1)) MiB")
prev = nothing
for (step, b) in m.rows
    println("step $step  in use $(round(b / 2^30; digits = 3)) GiB", prev === nothing ? "" : "  delta $(round((b - prev) / 2^20; digits = 1)) MiB")
    global prev = b
end
