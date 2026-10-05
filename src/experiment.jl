@experiment struct PathMNIST
    "Luximm variant: selects both the architecture and the pretrained checkpoint."
    variant::GraphConst{Symbol} = :convnextv2_tiny_fcmae_ft_in22k_in1k
    "PathMNIST tissue classes."
    num_classes::GraphConst{Int} = 9
    "Per-channel mean the pretrained weights expect (ImageNet)."
    pixel_mean::GraphConst{NTuple{3,Float32}} = (0.485f0, 0.456f0, 0.406f0)
    "Per-channel standard deviation the pretrained weights expect (ImageNet)."
    pixel_std::GraphConst{NTuple{3,Float32}} = (0.229f0, 0.224f0, 0.225f0)
    "LayerNorm on the pooled features before the head. ConvNeXt's timm head has one; BiT
    ResNetV2 already ends in a norm and its head is a bare linear layer, so set false for it."
    head_norm::GraphConst{Bool} = true
    "Augmented ODE state dimensions, predicted from the pooled features; the state also
    carries one logit dimension per class, starting at zero."
    aug_dims::GraphConst{Int} = 32
    "Hidden width of the ODE vector field (encoder, tanh, decoder)."
    node_hidden::GraphConst{Int} = 256
    "RK4 steps over t in [0, 1]. 0 replaces the NeuralODE head with one linear layer."
    node_steps::GraphConst{Int} = 100
    "Weight of the kinetic-energy penalty on the ODE's trajectory (Finlay et al., 2020).
    Off by default; a traced input, so turning it on or scheduling it recompiles nothing."
    kinetic_weight::Device{Float32} = 0.0f0

    "Where the archive comes from on first use (MedMNIST+ on Zenodo)."
    data_url::String = "https://zenodo.org/records/10519652/files/pathmnist_224.npz?download=1"
    "Directory for the converted, memory-mapped .u8 files, relative to the working directory."
    data_dir::String = joinpath("data", "pathmnist_224")
    "Seeds the shuffle; Nitro also reads it as the run seed."
    seed::Int = 0
    "Per-step batch: fits a 12 GB consumer GPU."
    batch_size::Int = 16
    "Gradient accumulation steps (read by Nitro by name): an effective batch of 32."
    accum::Int = 2
    max_epochs::Int = 5

    "Peak learning rate of the one-cycle schedule (RAdam)."
    max_lr::Float64 = 1e-3
    "Learning rate of the NeuralODE vector field relative to the rest of the model."
    node_lr_ratio::Float64 = 1.0
    "Fraction of the run spent warming up to max_lr."
    pct_start::Float64 = 0.3
    "Decoupled weight decay; the NeuralODE vector field is not decayed."
    weight_decay::Float64 = 1e-4

    "Augmentation: random 90° rotations (k = 0:3)."
    aug_rot90::Bool = true
    "Augmentation: random mirror. With aug_rot90, covers all 8 orientations."
    aug_flip::Bool = true
    "Augmentation: random hue shift, up to this fraction of a turn either way (0.05 is ±18°)."
    aug_hue::Float64 = 0.05
    "Augmentation: random saturation scale, by a factor in [1 - aug_sat, 1 + aug_sat]."
    aug_sat::Float64 = 0.3
    "Augmentation: random value (brightness) scale, by a factor in [1 - aug_val, 1 + aug_val]."
    aug_val::Float64 = 0.2
end
