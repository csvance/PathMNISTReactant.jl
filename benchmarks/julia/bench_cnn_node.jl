# Training-step throughput for a mixed program: a pretrained CNN backbone into a
# neural ODE into a classifier. The Julia side of `bench_lightning.py --node-steps 100`.
#
#   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
#     julia -t 16,1 --project=benchmarks/julia benchmarks/julia/bench_cnn_node.jl \
#       [default|highest] [convnextv2|bit]
#
# UInt8 (224, 224, 3, N) -> backbone -> pooled features (LayerNorm for ConvNeXt, as
# timm's head has) -> Dense(width => 64) -> RK4 over t in [0, 1], 100 steps,
# du/dt = dec(tanh(enc(u))) with hidden 256 -> Dense(64 => 9) logits, mean
# cross-entropy. Batch 16 x 2 accumulation steps, random batches built up front: the
# conditions of bench.jl `prebuilt`.

include(joinpath(@__DIR__, "common.jl"))
using PathMNISTReactant, Luximm

const PRECISION = get(ARGS, 1, "default")
const BACKBONE = get(ARGS, 2, "convnextv2")

@experiment struct CNNNODE
    variant::GraphConst{Symbol} = :convnextv2_tiny_fcmae_ft_in22k_in1k
    num_classes::GraphConst{Int} = 9
    pixel_mean::GraphConst{NTuple{3,Float32}} = (0.485f0, 0.456f0, 0.406f0)
    pixel_std::GraphConst{NTuple{3,Float32}} = (0.229f0, 0.224f0, 0.225f0)
    "LayerNorm on the pooled features: ConvNeXt's timm head has one, BiT's does not."
    head_norm::GraphConst{Bool} = true
    "ODE state dimension."
    node_dims::GraphConst{Int} = 64
    "Hidden width of the ODE vector field."
    node_hidden::GraphConst{Int} = 256
    "RK4 steps over t in [0, 1]."
    node_steps::GraphConst{Int} = 100
    seed::Int = 0
    batch_size::Int = 16
    accum::Int = 2
    max_epochs::Int = 1
end

function ReactantNitro.build_model(e::CNNNODE, rng)
    backbone, load_backbone = create_pretrained(e.variant; num_classes = 0, prefix = (:backbone,))
    width = last(feature_info(e.variant).channels)
    μ = reshape(collect(e.pixel_mean), 1, 1, 3, 1)
    σ = reshape(collect(e.pixel_std), 1, 1, 3, 1)
    model = @compact(
        backbone = backbone,
        norm = e.head_norm ? LayerNorm((width,)) : NoOpLayer(),
        proj = Dense(width => e.node_dims),
        enc = Dense(e.node_dims => e.node_hidden, tanh),
        dec = Dense(e.node_hidden => e.node_dims),
        cls = Dense(e.node_dims => e.num_classes),
        K = e.node_steps,
    ) do x                                   # (W, H, 3, N) UInt8
        x = PathMNISTReactant.to_float32(x)
        x = (x .- μ) ./ σ
        pooled = norm(dropdims(mean(backbone(x); dims = (1, 2)); dims = (1, 2)))
        @return cls(rk4(u -> dec(enc(u)), proj(pooled), K))     # (num_classes, N)
    end
    ps, st = Lux.setup(rng, model)
    ps, st = load_backbone(ps, st)
    return model, ps, st
end
ReactantNitro.forward(::CNNNODE, model, ps, st; img) = Lux.apply(model, img, ps, st)
function ReactantNitro.loss(e::CNNNODE, logits; label)
    onehot = reshape(1:e.num_classes, :, 1) .== reshape(label, 1, :)
    return -mean(sum(onehot .* logsoftmax(logits; dims = 1); dims = 1))
end

const E0 = CNNNODE(; BACKBONES[BACKBONE]...)

function prebuilt_batches(; pool = 32, rng = Xoshiro(0))
    batches = [(; img = rand(rng, UInt8, 224, 224, 3, E0.batch_size),
                  label = rand(rng, 1:9, E0.batch_size)) for _ in 1:pool]
    return [batches[mod1(i, pool)] for i in 1:STEPS * E0.accum]
end

make_nitro(logger) =
    Nitro(E0; data = (; train = prebuilt_batches()), max_epochs = 1, logger, run_dir = mktempdir(),
          compile_options = compile_options())

for run in ("first", "cached")      # first pays the compile; cached reuses the programs
    r = timed_train(make_nitro, PRECISION)
    emit((; stack = "reactantnitro", workload = "cnn-node-rk4", backbone = E0.variant,
          node_steps = E0.node_steps, params = Lux.parameterlength(r.nitro.model),
          mode = "prebuilt", precision = PRECISION, run, batch_size = E0.batch_size,
          accum = E0.accum, first_step_s = r.first_step_s, step_ms = r.step_ms,
          images_per_s = round(MEASURE * E0.accum * E0.batch_size / r.dt; digits = 1),
          env_fields()...))
end
