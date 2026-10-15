# Shared by every Julia benchmark: the step timer, the RK4 rule, the backbone presets,
# the precision switch, one timed `train!`, and the JSON line each run emits.
#
# Timing: 30 warmup optimizer steps, then 100 timed. A script trains twice in one
# process with a fresh Nitro: `first` pays the compile, `cached` reuses the programs.
#
# Output: each JSON line is printed and, when `BENCH_OUT` names a file, appended to it.

using ReactantNitro, Reactant, Lux, Random, Statistics

const WARMUP, MEASURE = 30, 100
const STEPS = WARMUP + MEASURE

# A logger that timestamps every optimizer step (context "train") and keeps the
# per-epoch data-wait report (context "data"). The other verbs are no-ops.
struct StepTimer
    times::Vector{Float64}
    data::Vector{Any}
end
StepTimer() = StepTimer(Float64[], Any[])
function ReactantNitro.log_metrics!(t::StepTimer, metrics; context = "train", kwargs...)
    context == "train" && push!(t.times, time())
    context == "data" && push!(t.data, metrics)
    return nothing
end
ReactantNitro.log_params!(::StepTimer, params) = nothing
ReactantNitro.log_tags!(::StepTimer, tags) = nothing
ReactantNitro.log_other!(::StepTimer, key, value) = nothing
ReactantNitro.log_confusion!(::StepTimer, matrix, labels; kwargs...) = nothing
ReactantNitro.finish!(::StepTimer, status) = nothing
ReactantNitro.run_id(::StepTimer) = nothing
ReactantNitro.run_url(::StepTimer) = nothing
ReactantNitro.reattach!(::StepTimer, state) = nothing

# Classical four-stage RK4 with `K` uniform steps over t in [0, 1] (SpiNODENitro's
# `_rk4_solve` rule, final state only). The fields here are autonomous, so `t` never
# enters. The loop unrolls at trace time into one static program.
function rk4(f, u, K::Integer)
    T = eltype(u)
    dt = one(T) / T(K)
    half = dt / T(2)
    sixth = dt / T(6)
    for _ in 1:K
        k1 = f(u)
        k2 = f(u .+ half .* k1)
        k3 = f(u .+ half .* k2)
        k4 = f(u .+ dt .* k3)
        u = u .+ sixth .* (k1 .+ T(2) .* k2 .+ T(2) .* k3 .+ k4)
    end
    return u
end

# Experiment keywords per backbone. BiT has its own normalization and timm builds its
# head as a bare linear layer, so it takes no extra LayerNorm.
const BACKBONES = Dict(
    "convnextv2" => (; variant = :convnextv2_tiny_fcmae_ft_in22k_in1k),
    "bit" => (;
        variant = :resnetv2_50x1_bit_goog_in21k, pixel_mean = (0.5f0, 0.5f0, 0.5f0),
        pixel_std = (0.5f0, 0.5f0, 0.5f0), head_norm = false,
    ),
)

# default: Reactant's PrecisionConfig.DEFAULT, which lets XLA use TF32 on Ampere (the
# match for PyTorch tf32). highest: strict FP32 (the match for PyTorch fp32).
function with_precision(f, precision::AbstractString)
    precision == "default" && return f()
    precision == "highest" || error("precision is default or highest, not $precision")
    hi = Reactant.PrecisionConfig.HIGHEST
    return Reactant.with_config(f; dot_general_precision = hi, convolution_precision = hi)
end

"""
    timed_train(make_nitro, precision) -> NamedTuple

`make_nitro(logger)` builds a Nitro for one epoch of exactly `STEPS` optimizer steps;
this trains it and returns the timings.
"""
function timed_train(make_nitro, precision)
    timer = StepTimer()
    nitro = make_nitro(timer)
    t_start = time()
    with_precision(() -> train!(nitro), precision)
    t = timer.times
    length(t) == STEPS || error("expected $STEPS timed steps, got $(length(t))")
    dt = t[STEPS] - t[WARMUP]
    wait = isempty(timer.data) ? nothing : get(first(timer.data), :data_wait_frac, nothing)
    return (;
        nitro, dt, first_step_s = round(t[1] - t_start; digits = 2),
        step_ms = round(1000 * dt / MEASURE; digits = 3),
        data_wait_frac = wait === nothing ? nothing : round(wait; digits = 4),
    )
end

# Compiler effort, the analog of PyTorch's compile modes, comes from two places:
#   XLA_FLAGS         XLA's GPU flags (autotuning breadth, command buffers, conv layout). XLA
#                     parses it once, when Reactant loads, so run_all.sh sets it before Julia
#                     starts; an unknown flag aborts the process.
#   BENCH_REACTANT    a named set of Reactant compile keywords below, passed to every program
#                     through `Nitro(...; compile_options)` (Enzyme-JAX passes, cuDNN HLO rewrites).
# BENCH_CONFIG names the combination for the report.
const REACTANT_PRESETS = Dict(
    "default" => (;),
    # The optimization passes Reactant leaves off by default for compile time. The pass switches
    # are not `compile` keywords, so this is a `CompileOptions` value rather than a NamedTuple.
    # `cudnn_hlo_optimize` (experimental) is left out: in Reactant 0.2.290 its "cudnn-hlo-opt"
    # pipeline fails on these models with "redefinition of symbol named
    # '__cudnn_fused_elementwise_dot_*'" (the neural ODE gradient, 2026-10-02).
    "max" => Reactant.CompileOptions(;
        disable_slice_to_batch_passes = false,
        disable_structured_tensors_detection_passes = false
    ),
)
reactant_preset() = get(ENV, "BENCH_REACTANT", "default")
compile_options() = REACTANT_PRESETS[reactant_preset()]

env_fields() = (;
    threads = Threads.nthreads(:default),
    config = get(ENV, "BENCH_CONFIG", "default"),
    xla_flags = get(ENV, "XLA_FLAGS", ""), reactant_preset = reactant_preset(),
    julia = string(VERSION),
    reactantnitro = string(pkgversion(ReactantNitro)),
    reactant = string(pkgversion(Reactant)),
)

_json(x::Nothing) = "null"
_json(x::Bool) = string(x)
_json(x::Real) = isfinite(x) ? string(x) : "null"
_json(x::Union{AbstractString, Symbol}) = "\"" * escape_string(string(x)) * "\""

"""
    emit(fields::NamedTuple)

Print one JSON line, and append it to the file `BENCH_OUT` names, if any.
"""
function emit(fields::NamedTuple)
    line = "{" * join(("\"$k\": $(_json(v))" for (k, v) in pairs(fields)), ", ") * "}"
    println(line)
    flush(stdout)
    out = get(ENV, "BENCH_OUT", "")
    isempty(out) || open(io -> println(io, line), out, "a")
    return nothing
end
