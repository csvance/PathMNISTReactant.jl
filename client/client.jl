# Query the served PathMNIST model over KServe V2 gRPC with nothing but ReactantServerClient
# and the memory-mapped test split: no ReactantNitro, no Reactant, no Lux, no GPU.
#
#   julia --project=client client/client.jl <data_dir> [grpc://host:port]
#
# <data_dir> holds the converted .u8 files the training package writes.
using ReactantServerClient, Mmap, Printf

#region load_split
const TILE = (3, 224, 224)                 # bytes on disk are (C, W, H, N)

function load_split(dir, split)
    images = open(joinpath(dir, "$(split)_images.u8")) do io
        Mmap.mmap(io, Array{UInt8, 4}, (TILE..., filesize(io) ÷ prod(TILE)))
    end
    labels = Int.(read(joinpath(dir, "$(split)_labels.u8"))) .+ 1   # 0-based on disk
    return images, labels
end
#endregion

const CLASSES = ("adipose", "background", "debris", "lymphocytes", "mucus", "smooth muscle",
    "normal mucosa", "stroma", "tumor epithelium")

# The model takes (W, H, C, N), UInt8: the same permutation the training loader applies, and
# no augmentation. Normalization happens inside the served program.
tiles(images, idx) = permutedims(images[:, :, :, idx], (2, 3, 1, 4))

softmax(z) = (e = exp.(z .- maximum(z; dims = 1)); e ./ sum(e; dims = 1))

# The fast path for a whole split: the client hands each chunk a slot in its staging pool,
# we write the tiles straight into it, and the bytes go out raw (or through shared memory
# when the server is on the same host), with several chunks in flight.
#region tile_io
struct TileIO <: AbstractInferenceIO
    images::Array{UInt8, 4}           # memory-mapped, (C, W, H, N)
    logits::Matrix{Float32}           # filled in as responses arrive
end
Base.length(io::TileIO) = size(io.images, 4)
ReactantServerClient.item_input_bytes(::TileIO) = prod(TILE)

function ReactantServerClient.infer_encode_chunk!(io::TileIO, r::UnitRange, slot::PoolSlot)
    input = scratch(slot, "img", (224, 224, 3, length(r)), UInt8)
    buf = pool_view(input)
    for (k, i) in enumerate(r)        # (C, W, H) on disk to the model's (W, H, C)
        permutedims!(view(buf, :, :, :, k), view(io.images, :, :, :, i), (2, 3, 1))
    end
    return input
end

function ReactantServerClient.infer_decode_chunk!(io::TileIO, r::UnitRange, response)
    io.logits[:, r] = InferOutput("logits", response, Float32)
    return nothing
end
#endregion

function main(data_dir, url = "grpc://127.0.0.1:8001"; batch = 16)
    images, labels = load_split(data_dir, "test")
    N = length(labels)
    logits = zeros(Float32, length(CLASSES), N)
    kserve_init()
    try
        model = KServeModel(url, "pathmnist_node"; max_batch_size = batch)
        io = TileIO(images, logits)
        infer_async(model, TileIO(images[:, :, :, 1:batch], zeros(Float32, length(CLASSES), batch)))  # warm up
        t0 = time()
        infer_async(model, io)                                # the whole split, chunks in flight
        dt = time() - t0
        # Single-tile latency, one plain request at a time.
        infer(idx) = InferOutput("logits", infer_sync(model, [InferInput("img", tiles(images, idx))]), Float32)
        lat = [(t = time(); infer(i:i); 1000 * (time() - t)) for i in 1:200]
        sort!(lat)
        @printf("test set: %d tiles in %.1f s (%.0f tiles/s, batch %d)\n", N, dt, N / dt, batch)
        @printf("batch-1 latency: p50 %.1f ms, p99 %.1f ms\n", lat[100], lat[198])
    finally
        kserve_shutdown()
    end
    probs = softmax(logits)
    pred = [argmax(c) for c in eachcol(probs)]
    recall = [count((pred .== k) .& (labels .== k)) / count(labels .== k) for k in eachindex(CLASSES)]
    @printf("accuracy %.2f%%, macro accuracy %.2f%%\n", 100 * count(pred .== labels) / N, 100 * sum(recall) / length(recall))
    return (; logits, probs, pred, labels)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS...)
end
