#region optimizer
ReactantNitro.param_group(::PathMNIST, ks) = (:enc in ks || :dec in ks) ? :node : :default

ReactantNitro.learning_rate(e::PathMNIST) = Float32(e.max_lr)
ReactantNitro.learning_rate(e::PathMNIST, ::Val{:node}) = Float32(e.max_lr * e.node_lr_ratio)
ReactantNitro.lambda(e::PathMNIST) = Float32(e.weight_decay)
ReactantNitro.lambda(::PathMNIST, ::Val{:node}) = 0.0f0

# No decay on biases or norm scales and shifts: any leaf with at most one
# dimension larger than 1, whatever singleton dimensions Lux gives it
ReactantNitro.no_decay(::PathMNIST, ks, x) = count(>(1), size(x)) <= 1

ReactantNitro.schedules(e::PathMNIST) = (;
    eta = total -> OneCycle(
        total, Float32(e.max_lr); percent_start = e.pct_start,
        startval = Float32(e.max_lr / 25), endval = Float32(e.max_lr / 25 / 1.0e4)
    ),
)

# Clip the global gradient norm loosely: it only catches the spikes that
# occasionally make the warm-up diverge
ReactantNitro.gradient_clip_norm(::PathMNIST) = 10.0f0

# Validation, checkpoint selection and export read this average of the weights
ReactantNitro.ema(::PathMNIST) = EMA(0.999)

# The metric the run is judged by: checkpoint selection and `restore_best` follow it
ReactantNitro.monitor(::PathMNIST) = :macro_acc => :max

# Report the loss on validation and test from its own small device program
ReactantNitro.eval_loss(::PathMNIST) = true
#endregion
