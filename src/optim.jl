# The optimizer: RAdam (Nitro's default rule) with decoupled weight decay and a one-cycle
# learning-rate schedule. All of it is host-side, so changing any value recompiles nothing.

# The NeuralODE's vector field (encoder and decoder) is its own group, so it can go
# undecayed: decay would pull the dynamics back toward zero.
ReactantNitro.param_group(::PathMNIST, ks) = (:enc in ks || :dec in ks) ? :node : :default

ReactantNitro.learning_rate(e::PathMNIST) = Float32(e.max_lr)
# Per-group rates are ratios against the base, so the one-cycle schedule scales both together.
ReactantNitro.learning_rate(e::PathMNIST, ::Val{:node}) = Float32(e.max_lr * e.node_lr_ratio)
ReactantNitro.lambda(e::PathMNIST) = Float32(e.weight_decay)
ReactantNitro.lambda(::PathMNIST, ::Val{:node}) = 0.0f0

# No decay on biases or normalization scales and shifts. Nitro's default before 0.2.0 only
# excluded 1-D leaves, and Lux stores LayerNorm's affine parameters as (1, 1, C, 1) in the
# backbone and (C, 1) in the head, so they were decayed. Exclude a leaf whose name is a bias or
# a norm `scale` (Lux's norm layers all use those names), or that has at most one dimension
# larger than 1.
function ReactantNitro.no_decay(::PathMNIST, ks, x)
    k = last(ks)
    named = k isa Symbol && (occursin("bias", String(k)) || k === :scale)
    return named || count(>(1), size(x)) <= 1
end

# PyTorch OneCycleLR's defaults, mapped argument by argument: start at max_lr / 25,
# end at max_lr / 25 / 1e4, cosine annealing.
ReactantNitro.schedules(e::PathMNIST) = (;
    eta = total -> OneCycle(
        total, Float32(e.max_lr); percent_start = e.pct_start,
        startval = Float32(e.max_lr / 25), endval = Float32(e.max_lr / 25 / 1e4)
    ),
)

# Keep the checkpoints with the best macro accuracy, the main metric.
ReactantNitro.checkpointer(::PathMNIST) = TopKCheckpointer(; metric = :macro_acc, mode = :max)
