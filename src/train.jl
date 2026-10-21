#region hooks
ReactantNitro.forward(e::PathMNIST, model, ps, st; img) = Lux.apply(model, img, ps, st)

function cross_entropy(e::PathMNIST, logits, label)
    onehot = reshape(1:e.num_classes, :, 1) .== reshape(label, 1, :)
    return -mean(sum(onehot .* logsoftmax(logits; dims = 1); dims = 1))
end

ReactantNitro.loss(e::PathMNIST, out; label) =
    cross_entropy(e, out.logits, label) + e.kinetic_weight * mean(out.energy)

ReactantNitro.train_metrics(e::PathMNIST, out) =
    (; kinetic = mean(out.energy), motion = mean(out.motion))
#endregion

#region metrics
const ECE_BINS = 15    # confidence bins for the expected calibration error

function ReactantNitro.metrics(e::PathMNIST, out; label)
    n = length(label)
    probs = softmax(out.logits; dims = 1)
    confusion = zeros(Int, e.num_classes, e.num_classes)       # rows true, columns predicted
    samples, conf, correct = zeros(ECE_BINS), zeros(ECE_BINS), zeros(ECE_BINS)
    for (p, l) in zip(eachcol(probs), label)
        c, pred = findmax(p)
        confusion[l, pred] += 1
        b = clamp(ceil(Int, c * ECE_BINS), 1, ECE_BINS)
        samples[b] += 1
        conf[b] += c
        correct[b] += pred == l
    end
    return (;
        confusion = confusion => :sum,
        calib = (; samples, conf, correct) => :sum,
        kinetic = sum(out.energy) => n,
        motion = sum(out.motion) => n,
    )
end

function ReactantNitro.finalize_metrics(e::PathMNIST, acc, split)
    c = acc.confusion
    recall = [c[k, k] / sum(c[k, :]) for k in axes(c, 1) if sum(c[k, :]) > 0]
    (; samples, conf, correct) = acc.calib
    return (;
        macro_acc = sum(recall) / length(recall),            # mean per-class recall
        acc = sum(c[k, k] for k in axes(c, 1)) / sum(c),
        ece = sum(abs.(conf .- correct)) / sum(samples),
        acc.motion,
        acc.kinetic,
    )
end
#endregion
