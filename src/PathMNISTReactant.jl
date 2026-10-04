"""
    PathMNISTReactant

Companion code for the blog post: ConvNeXt V2 Tiny into an RK4 NeuralODE head,
fine-tuned on PathMNIST with ReactantNitro, exported and served with
ReactantServer. Pure Julia.
"""
module PathMNISTReactant

using ReactantNitro, Reactant, Downloads, Lux, Luximm, MLUtils, Mmap, Random, Statistics, ZipArchives
using ParameterSchedulers: OneCycle

export PathMNIST, PathMNISTSplit, ensure_dataset_mmapable, convert_pathmnist, dihedral

include("experiment.jl")   # the @experiment: every knob, one flat struct
include("node.jl")         # RK4 and the NeuralODE head
include("model.jl")        # build_model: Luximm backbone + head via @compact
include("train.jl")        # forward, loss, and metrics
include("optim.jl")        # RAdam, decay, one-cycle schedule, checkpoint selection
include("augment.jl")      # the 8 orientations of a square tile
include("data.jl")         # memory-mapped splits and build_data
include("dataset.jl")      # download and one-time .npz -> flat .u8 conversion

end
