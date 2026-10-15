# The vision benchmarks' model: a Luximm backbone with one linear head, no NeuralODE. BiT
# has its own normalization and timm builds its head as a bare linear layer, so `head_norm`
# turns the pooled-feature LayerNorm off for it. Data comes from the package's PathMNIST
# loader, so the `pipeline` mode runs the same build_data path as training.

using PathMNISTReactant, ReactantNitro, Lux, Luximm, Statistics

@experiment struct Vision
    variant::GraphConst{Symbol} = :convnextv2_tiny_fcmae_ft_in22k_in1k
    num_classes::GraphConst{Int} = 9
    pixel_mean::GraphConst{NTuple{3, Float32}} = (0.485f0, 0.456f0, 0.406f0)
    pixel_std::GraphConst{NTuple{3, Float32}} = (0.229f0, 0.224f0, 0.225f0)
    "LayerNorm on the pooled features: ConvNeXt's timm head has one, BiT's does not."
    head_norm::GraphConst{Bool} = true
    data_dir::String = joinpath("data", "pathmnist_224")
    seed::Int = 0
    batch_size::Int = 16
    accum::Int = 2
    max_epochs::Int = 1
end

function ReactantNitro.build_model(e::Vision, rng)
    backbone, load_backbone = create_pretrained(e.variant; num_classes = 0, prefix = (:backbone,))
    width = last(feature_info(e.variant).channels)
    μ = reshape(collect(e.pixel_mean), 1, 1, 3, 1)
    σ = reshape(collect(e.pixel_std), 1, 1, 3, 1)
    model = @compact(
        backbone = backbone,
        norm = e.head_norm ? LayerNorm((width,)) : NoOpLayer(),
        head = Dense(width => e.num_classes),
    ) do x                                   # (W, H, 3, N) UInt8
        x = PathMNISTReactant.to_float32(x)
        x = (x .- μ) ./ σ
        pooled = norm(dropdims(mean(backbone(x); dims = (1, 2)); dims = (1, 2)))
        @return head(pooled)                 # (num_classes, N) logits
    end
    ps, st = Lux.setup(rng, model)
    ps, st = load_backbone(ps, st)
    return model, ps, st
end

ReactantNitro.forward(::Vision, model, ps, st; img) = Lux.apply(model, img, ps, st)

function ReactantNitro.loss(e::Vision, logits; label)
    onehot = reshape(1:e.num_classes, :, 1) .== reshape(label, 1, :)
    return -mean(sum(onehot .* logsoftmax(logits; dims = 1); dims = 1))
end

ReactantNitro.build_data(e::Vision, dist) = ReactantNitro.build_data(
    PathMNIST(; data_dir = e.data_dir, seed = e.seed, batch_size = e.batch_size), dist
)
