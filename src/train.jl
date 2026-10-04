# `forward` is the exportable program and is what `predict` calls: inputs only.
ReactantNitro.forward(e::PathMNIST, model, ps, st; img) = Lux.apply(model, img, ps, st)

# Mean cross-entropy over the batch. Labels arrive as 1-based integers and are
# one-hot encoded inside the traced program, so batches stay small.
function cross_entropy(e::PathMNIST, logits, label)
    onehot = reshape(1:e.num_classes, :, 1) .== reshape(label, 1, :)
    return -mean(sum(onehot .* logsoftmax(logits; dims = 1); dims = 1))
end

# The NeuralODE head returns its logits with the trajectory's kinetic energy, which is
# penalized (Finlay et al., 2020); a linear head (node_steps = 0) returns bare logits.
# One method per hook: Nitro reads the batch fields a hook needs from its keywords.
function ReactantNitro.loss(e::PathMNIST, out; label)
    out isa NamedTuple || return cross_entropy(e, out, label)
    return cross_entropy(e, out.logits, label) + e.kinetic_weight * mean(out.energy)
end

# Per training step, on the device: what the dynamics are doing, logged for free.
ReactantNitro.train_metrics(e::PathMNIST, out) =
    out isa NamedTuple ? (; kinetic = mean(out.energy), motion = mean(out.motion)) : (;)

"Confidence bins for the expected calibration error."
const ECE_BINS = 15

# Validation metrics, on the host (the default residency). A (sum, count) pair is a mean
# Nitro divides once over the split; a count of `nothing` is a plain sum, used here for
# the confusion matrix and the calibration bins.
function ReactantNitro.metrics(e::PathMNIST, out; label)
    logits = out isa NamedTuple ? out.logits : out
    n = length(label)
    probs = softmax(logits; dims = 1)
    confusion = zeros(Int, e.num_classes, e.num_classes)       # rows true, columns predicted
    calib = zeros(3, ECE_BINS)                                  # per bin: count, confidence, correct
    for (p, l) in zip(eachcol(probs), label)
        conf, pred = findmax(p)
        confusion[l, pred] += 1
        b = clamp(ceil(Int, conf * ECE_BINS), 1, ECE_BINS)
        calib[:, b] .+= (1, conf, pred == l)
    end
    m = (; val_loss = (cross_entropy(e, logits, label) * n, n),
           confusion = (confusion, nothing), calib = (calib, nothing))
    out isa NamedTuple || return m
    # How much the ODE moves the state, and how much energy it spends doing it.
    return (; m..., kinetic = (sum(out.energy), n), motion = (sum(out.motion), n))
end

# Once per split per epoch: macro accuracy (the mean of per-class recall, the main
# metric), plain accuracy, and the top-label expected calibration error.
function ReactantNitro.finalize_metrics(e::PathMNIST, acc, split)
    c = acc.confusion
    # Over the classes present in the split, as balanced accuracy is usually defined.
    recall = [c[k, k] / sum(c[k, :]) for k in axes(c, 1) if sum(c[k, :]) > 0]
    n, conf, correct = eachrow(acc.calib)
    ece = sum(abs.(conf .- correct)) / sum(n)
    scores = (; acc.val_loss, macro_acc = sum(recall) / length(recall),
                acc = sum(c[k, k] for k in axes(c, 1)) / sum(c), ece)
    return haskey(acc, :motion) ? (; scores..., acc.motion, acc.kinetic) : scores
end
