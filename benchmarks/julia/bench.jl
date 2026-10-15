# Training-step throughput for the vision side of the comparison: a Luximm backbone with a
# 9-class linear head (vision.jl).
#
#   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
#     julia -t 16,1 --project=benchmarks/julia benchmarks/julia/bench.jl \
#       [prebuilt|pipeline] [default|highest] [convnextv2|bit]
#
# prebuilt: random batches built up front and passed as `data`, skipping
#           build_data; the same favorable condition the PyTorch script gets.
# pipeline: the real build_data path over a memory-mapped synthetic dataset
#           (fan-out over threads, augmentation on).
# Precision and backbones: see common.jl.

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "vision.jl"))

const MODE = get(ARGS, 1, "prebuilt")
const PRECISION = get(ARGS, 2, "default")
const BACKBONE = get(ARGS, 3, "convnextv2")
const E0 = Vision(; BACKBONES[BACKBONE]...)
const MICRO = STEPS * E0.accum                  # micro-batches per epoch
const TILE_BYTES = 3 * 224 * 224

# Random tiles in the converted on-disk format, so build_data runs unchanged.
function synthetic_dataset(dir, n_train; n_eval = 16, rng = Xoshiro(0))
    mkpath(dir)
    for (split, n) in (("train", n_train), ("val", n_eval), ("test", n_eval))
        write(joinpath(dir, "$(split)_images.u8"), rand(rng, UInt8, n * TILE_BYTES))
        write(joinpath(dir, "$(split)_labels.u8"), rand(rng, UInt8(0):UInt8(8), n))
    end
    return dir
end

function prebuilt_batches(; pool = 32, rng = Xoshiro(0))
    batches = [
        (;
            img = rand(rng, UInt8, 224, 224, 3, E0.batch_size),
            label = rand(rng, 1:9, E0.batch_size),
        ) for _ in 1:pool
    ]
    return [batches[mod1(i, pool)] for i in 1:MICRO]
end

function make_nitro(logger)
    run_dir = mktempdir()
    MODE == "prebuilt" &&
        return Nitro(
        E0; data = (; train = prebuilt_batches()), max_epochs = 1, logger, run_dir,
        compile_options = compile_options()
    )
    MODE == "pipeline" || error("mode is prebuilt or pipeline, not $MODE")
    dir = synthetic_dataset(joinpath(run_dir, "data"), MICRO * E0.batch_size)
    return Nitro(
        Vision(; BACKBONES[BACKBONE]..., data_dir = dir); max_epochs = 1, logger, run_dir,
        compile_options = compile_options()
    )
end

for run in ("first", "cached")      # first pays the compile; cached reuses the programs
    r = timed_train(make_nitro, PRECISION)
    emit(
        (;
            stack = "reactantnitro", workload = "vision", backbone = E0.variant,
            params = Lux.parameterlength(r.nitro.model), mode = MODE, precision = PRECISION, run,
            batch_size = E0.batch_size, accum = E0.accum, first_step_s = r.first_step_s,
            step_ms = r.step_ms, images_per_s = round(MEASURE * E0.accum * E0.batch_size / r.dt; digits = 1),
            data_wait_frac = r.data_wait_frac, env_fields()...,
        )
    )
end
