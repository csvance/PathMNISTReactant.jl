# NumPy stores each split as C-order (N, H, W, C). Read column-major, the same
# bytes are (C, W, H, N), so the file maps with no copy.
const TILE = (3, 224, 224)

"PathMNIST's nine tissue classes, in label order (NCT-CRC-HE-100K)."
const CLASS_NAMES = (
    "adipose", "background", "debris", "lymphocytes", "mucus", "smooth muscle",
    "normal mucosa", "stroma", "tumor epithelium",
)

"""
    PathMNISTSplit

One split, memory-mapped from the converted files. `getobs` builds a batch
in the (W, H, C, N) layout the model takes, augmenting when asked.
"""
struct PathMNISTSplit{E}
    images::Array{UInt8, 4}       # (C, W, H, N), memory-mapped
    labels::Vector{Int}          # 1:9
    e::E
    augment::Bool
end

function PathMNISTSplit(e, split::AbstractString; augment = false)
    images = open(joinpath(e.data_dir, "$(split)_images.u8")) do io
        Mmap.mmap(io, Array{UInt8, 4}, (TILE..., filesize(io) ÷ prod(TILE)))
    end
    labels = Int.(read(joinpath(e.data_dir, "$(split)_labels.u8"))) .+ 1   # 0-based on disk
    return PathMNISTSplit(images, labels, e, augment)
end

MLUtils.numobs(d::PathMNISTSplit) = length(d.labels)

function MLUtils.getobs(d::PathMNISTSplit, idxs::AbstractVector{<:Integer})
    # Batches are built on several threads at once, so each gets its own RNG,
    # seeded from its indices: reproducible for a given shuffle.
    rng = Xoshiro(hash(idxs))
    img = Array{UInt8, 4}(undef, TILE[2], TILE[3], TILE[1], length(idxs))
    for (j, i) in enumerate(idxs)
        tile = permutedims(view(d.images, :, :, :, i), (2, 3, 1))   # (W, H, C)
        img[:, :, :, j] = d.augment ? augment(d.e, tile, rng) : tile
    end
    return (; img, label = d.labels[idxs])
end

#region build_data
function ReactantNitro.build_data(e::PathMNIST, dist)
    ensure_dataset_mmapable(e)      # downloads and converts on first use
    # Training drops its short final batch (fixed program shape); eval keeps
    # it, and Nitro pads and slices it back.
    loader(split, train) = DataLoader(
        PathMNISTSplit(e, split; augment = train);
        batchsize = e.batch_size, shuffle = train, partial = !train,
        rng = Xoshiro(e.seed)
    )
    return (;
        train = loader("train", true), val = loader("val", false),
        test = loader("test", false),
    )
end
#endregion
