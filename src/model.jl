# UInt8 pixels to Float32 in [0, 1]. Two methods because the model runs both
# eagerly on host arrays and traced by Reactant, which has no
# `Float32(::TracedRNumber{UInt8})`; the traced path converts at the array level.
to_float32(x::AbstractArray) = Float32.(x) ./ 255.0f0
to_float32(x::Reactant.TracedRArray) =
    Reactant.Ops.convert(Reactant.TracedRArray{Float32, ndims(x)}, x) ./ 255.0f0

# One level of the 2D Haar transform of a (W, H, C, N) image: the half-size low-pass image,
# and the detail subbands that differ across the first axis, across the second, and across
# both (diagonal). Each is an octave band-pass at this scale.
function haar_level(x)
    W, H, C, N = size(x)
    y = reshape(x, 2, W ÷ 2, 2, H ÷ 2, C, N)
    a, b = y[1, :, 1, :, :, :], y[2, :, 1, :, :, :]
    c, d = y[1, :, 2, :, :, :], y[2, :, 2, :, :, :]
    return (a .+ b .+ c .+ d) ./ 2, (a .- b .+ c .- d) ./ 2, (a .+ b .- c .- d) ./ 2, (a .- b .- c .+ d) ./ 2
end

# Texture features: per level, the log mean energy of the two axis-aligned subbands pooled
# (90° rotations swap them) and of the diagonal subband, per channel. (2C × levels, N).
function wavelet_features(x, levels)
    size(x, 1) % 2^levels == 0 && size(x, 2) % 2^levels == 0 ||
        throw(ArgumentError("a $(size(x, 1))×$(size(x, 2)) tile does not divide into $levels Haar levels"))
    logenergy(s) = log.(dropdims(mean(s; dims = (1, 2)); dims = (1, 2)) .+ 1.0f-6)   # (C, N)
    out = []
    for _ in 1:levels
        x, d1, d2, dd = haar_level(x)
        push!(out, logenergy((d1 .^ 2 .+ d2 .^ 2) ./ 2), logenergy(dd .^ 2))
    end
    return reduce(vcat, out)
end

function ReactantNitro.build_model(e::PathMNIST, rng)
    # num_classes = 0 drops the ImageNet classifier and returns the last
    # feature map. prefix = (:backbone,) points the weight loader at the
    # @compact slot of the same name.
    backbone, load_backbone = create_pretrained(e.variant;
        num_classes = 0, prefix = (:backbone,))
    width = last(feature_info(e.variant).channels)          # 768 for Tiny
    # Captured by the closure rather than passed to @compact, which would
    # make them trainable parameters.
    μ = reshape(collect(e.pixel_mean), 1, 1, 3, 1)
    σ = reshape(collect(e.pixel_std), 1, 1, 3, 1)

    levels = e.wavelet_levels
    nw = 2 * 3 * levels                    # wavelet features: 2 per level per channel
    model = @compact(
        backbone = backbone,
        norm = e.head_norm ? LayerNorm((width,)) : NoOpLayer(),
        wnorm = levels > 0 ? LayerNorm((nw,)) : NoOpLayer(),
        # The NeuralODE head; node_steps = 0 (the vision benchmarks) is a linear head.
        head = e.node_steps > 0 ? node_head(e, width + nw) : Dense(width + nw => e.num_classes),
    ) do x                                 # (W, H, 3, N) UInt8, as stored on disk
        x01 = to_float32(x)                # convert on device: 4x less to transfer
        feats = backbone((x01 .- μ) ./ σ)  # (W/32, H/32, width, N)
        pooled = norm(dropdims(mean(feats; dims = (1, 2)); dims = (1, 2)))
        if levels > 0                      # texture statistics beside the backbone
            pooled = vcat(pooled, wnorm(wavelet_features(x01, levels)))
        end
        @return head(pooled)               # (num_classes, N) logits
    end

    ps, st = Lux.setup(rng, model)
    # Load here, not later: Nitro snapshots the weights build_model returns
    # as the initial point (w0).
    ps, st = load_backbone(ps, st)
    return model, ps, st
end
