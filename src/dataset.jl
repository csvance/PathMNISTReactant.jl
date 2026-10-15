const SPLITS = ("train", "val", "test")

"""
    ensure_dataset_mmapable(e::PathMNIST) -> nothing

Make sure `e.data_dir` holds the converted, memory-mappable split files,
downloading and converting the archive on first use. A no-op once the files
exist and their sizes agree. Every file is written under a `.partial` name
and renamed when complete, so an interrupted run never leaves files that
look finished.
"""
function ensure_dataset_mmapable(e)
    dataset_complete(e.data_dir) && return nothing
    mkpath(e.data_dir)
    npz = joinpath(e.data_dir, "pathmnist_224.npz")
    if !isfile(npz)
        @info "Downloading PathMNIST (12.6 GB) to $npz"
        Downloads.download(e.data_url, npz * ".partial"; progress = download_progress())
        mv(npz * ".partial", npz; force = true)
    end
    @info "Converting $npz to memory-mappable files"
    convert_pathmnist(npz, e.data_dir)
    dataset_complete(e.data_dir) || error("conversion finished but $(e.data_dir) is incomplete")
    rm(npz)
    return nothing
end

# Complete means all six files exist, the image bytes are a whole number of
# tiles, and each split has one label per image.
function dataset_complete(dir)
    for split in SPLITS
        images = joinpath(dir, "$(split)_images.u8")
        labels = joinpath(dir, "$(split)_labels.u8")
        isfile(images) && isfile(labels) || return false
        n, rem = divrem(filesize(images), prod(TILE))
        rem == 0 && n == filesize(labels) || return false
    end
    return true
end

# Log at most once per 5% of the download.
function download_progress()
    last = Ref(-1)
    return function (total, now)
        total > 0 || return
        pct = 5 * fld(100 * now, 5 * total)
        pct > last[] || return
        last[] = pct
        return @info "  downloaded $(round(now / 1.0e9; digits = 1)) / $(round(total / 1.0e9; digits = 1)) GB ($pct%)"
    end
end

"""
    convert_pathmnist(npz, dir)

Convert `pathmnist_224.npz` into flat `.u8` files in `dir`, which training
memory-maps. Each array streams from the archive straight to disk, so this
never holds a split in memory.
"""
function convert_pathmnist(npz, dir)
    mkpath(dir)
    # ZipArchives inflates through zlib at a couple of hundred MB/s; ZipFile.jl managed
    # about 1.4 MB/s here, which made the first run take hours.
    return open(npz) do f
        zip = ZipReader(Mmap.mmap(f))
        buf = Vector{UInt8}(undef, 1 << 24)
        for name in zip_names(zip)                 # train_images.npy, train_labels.npy, ...
            path = joinpath(dir, first(splitext(name)) * ".u8")
            zip_openentry(zip, name) do member
                read_npy_header(member)
                open(path * ".partial", "w") do out
                    while !eof(member)
                        write(out, view(buf, 1:readbytes!(member, buf)))
                    end
                end
            end
            mv(path * ".partial", path; force = true)
        end
    end
end

# The .npy header is a magic string, a version, a length, and a Python dict
# literal. Every PathMNIST array is C-order uint8, which is all we accept.
function read_npy_header(io)
    read(io, 6) == b"\x93NUMPY" || error("not an .npy array")
    major, _ = read(io, UInt8), read(io, UInt8)
    len = major == 1 ? Int(ltoh(read(io, UInt16))) : Int(ltoh(read(io, UInt32)))
    header = String(read(io, len))
    occursin("'descr': '|u1'", header) && occursin("'fortran_order': False", header) ||
        error("expected a C-order uint8 array, got $header")
    return header
end
