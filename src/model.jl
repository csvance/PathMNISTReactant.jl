# UInt8 pixels to Float32 in [0, 1]. Two methods because the model runs both
# eagerly on host arrays and traced by Reactant, which has no
# `Float32(::TracedRNumber{UInt8})`; the traced path converts at the array level.
to_float32(x::AbstractArray) = Float32.(x) ./ 255.0f0
to_float32(x::Reactant.TracedRArray) =
    Reactant.Ops.convert(Reactant.TracedRArray{Float32, ndims(x)}, x) ./ 255.0f0

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

    model = @compact(
        backbone = backbone,
        norm = e.head_norm ? LayerNorm((width,)) : NoOpLayer(),
        # The NeuralODE head; node_steps = 0 (the vision benchmarks) is a linear head.
        head = e.node_steps > 0 ? node_head(e, width) : Dense(width => e.num_classes),
    ) do x                                 # (W, H, 3, N) UInt8, as stored on disk
        x = to_float32(x)                  # convert on device: 4x less to transfer
        x = (x .- μ) ./ σ
        feats = backbone(x)                # (W/32, H/32, width, N)
        pooled = dropdims(mean(feats; dims = (1, 2)); dims = (1, 2))
        @return head(norm(pooled))         # (num_classes, N) logits
    end

    ps, st = Lux.setup(rng, model)
    # Load here, not later: Nitro snapshots the weights build_model returns
    # as the initial point (w0).
    ps, st = load_backbone(ps, st)
    return model, ps, st
end
