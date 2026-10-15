# Training-step throughput for a neural ODE, the Julia side of bench_node.py.
#
#   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
#     julia -t 16,1 --project=benchmarks/julia benchmarks/julia/bench_node.jl [default|highest]
#
# A pure MLP vector field, du/dt = dec(tanh(enc(u))), integrated over t in [0, 1]
# with fixed-step RK4 (common.jl), then a mean squared error on the final state.
# The rollout, its Enzyme adjoint and the RAdam update are each one XLA program.
# Batches are random and built up front, as on the PyTorch side.

include(joinpath(@__DIR__, "common.jl"))

const PRECISION = get(ARGS, 1, "default")

@experiment struct NODEBench
    "State dimension."
    dims::GraphConst{Int} = 32
    "Hidden width of the vector field."
    hidden::GraphConst{Int} = 256
    "RK4 steps over t in [0, 1]."
    steps::GraphConst{Int} = 100
    seed::Int = 0
    batch_size::Int = 128
    accum::Int = 1
    max_epochs::Int = 1
end

function ReactantNitro.build_model(e::NODEBench, rng)
    model = @compact(
        enc = Dense(e.dims => e.hidden, tanh),
        dec = Dense(e.hidden => e.dims),
        K = e.steps,
    ) do u0                                  # (dims, N)
        @return rk4(u -> dec(enc(u)), u0, K)
    end
    ps, st = Lux.setup(rng, model)
    return model, ps, st
end
ReactantNitro.forward(::NODEBench, model, ps, st; u0) = Lux.apply(model, u0, ps, st)
ReactantNitro.loss(::NODEBench, uT; y) = mean(abs2, uT .- y)

const E0 = NODEBench()

function batches(; pool = 8, rng = Xoshiro(0))
    b = [
        (;
            u0 = randn(rng, Float32, E0.dims, E0.batch_size),
            y = randn(rng, Float32, E0.dims, E0.batch_size),
        ) for _ in 1:pool
    ]
    return [b[mod1(i, pool)] for i in 1:STEPS]
end

make_nitro(logger) = Nitro(
    E0; data = (; train = batches()), max_epochs = 1, logger,
    run_dir = mktempdir(), compile_options = compile_options()
)

for run in ("first", "cached")      # first pays the compile; cached reuses the programs
    r = timed_train(make_nitro, PRECISION)
    emit(
        (;
            stack = "reactantnitro", workload = "node-rk4", precision = PRECISION, run,
            dims = E0.dims, hidden = E0.hidden, steps = E0.steps, batch_size = E0.batch_size,
            params = Lux.parameterlength(r.nitro.model), first_step_s = r.first_step_s,
            step_ms = r.step_ms, steps_per_s = round(MEASURE / r.dt; digits = 1), env_fields()...,
        )
    )
end
