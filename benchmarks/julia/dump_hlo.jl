# Dump the compiled programs of a vision backbone's training gradient: the StableHLO Reactant
# traces, the StableHLO after Reactant's and Enzyme-JAX's passes, and XLA's optimized HLO.
#
#   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
#     julia --project=benchmarks/julia benchmarks/julia/dump_hlo.jl <backbone> <outdir>
#
# <backbone> is a key of BACKBONES in common.jl (e.g. bit, convnextv2). The program is the
# gradient of the mean cross-entropy with respect to every parameter for one micro-batch, the same
# computation as the training step's gradient program, without the framework around it.

include(joinpath(@__DIR__, "common.jl"))
using PathMNISTReactant
const Enzyme = Reactant.Enzyme

const BACKBONE = ARGS[1]
const OUT = mkpath(ARGS[2])

lossf(p, m, s, x, y) = (oh = reshape(1:9, :, 1) .== reshape(y, 1, :);
    -mean(sum(oh .* logsoftmax(first(Lux.apply(m, x, p, s)); dims = 1); dims = 1)))
grad(p, m, s, x, y) = Enzyme.gradient(Enzyme.Reverse, lossf, p, Enzyme.Const(m), Enzyme.Const(s),
    Enzyme.Const(x), Enzyme.Const(y))[1]

e = PathMNIST(; BACKBONES[BACKBONE]..., node_steps = 0)
model, ps, st = ReactantNitro.build_model(e, Xoshiro(0))
st = Lux.trainmode(st)
x = Reactant.to_rarray(rand(Xoshiro(1), UInt8, 224, 224, 3, e.batch_size))
y = Reactant.to_rarray(rand(Xoshiro(2), 1:9, e.batch_size))
psr, str = Reactant.to_rarray(ps), Reactant.to_rarray(st)

write(joinpath(OUT, "stablehlo_raw.mlir"), string(@code_hlo optimize = false grad(psr, model, str, x, y)))
write(joinpath(OUT, "stablehlo_opt.mlir"), string(@code_hlo grad(psr, model, str, x, y)))
write(joinpath(OUT, "xla_opt.hlo.txt"), string(@code_xla grad(psr, model, str, x, y)))
println("wrote ", join(readdir(OUT), ", "), " to ", OUT)
