"""
    dihedral(x, k, mirror) -> array

One of the 8 symmetries of a square tile: mirror (optional), then rotate by
`k × 90°`. Index permutations only, so there is no interpolation and `UInt8`
stays `UInt8`. `x` is `(W, H, C)` with `W == H`.
"""
function dihedral(x::AbstractArray{<:Any,3}, k::Integer, mirror::Bool)
    size(x, 1) == size(x, 2) || throw(ArgumentError("tile must be square, got $(size(x))"))
    y = mirror ? reverse(x; dims = 1) : x
    for _ in 1:mod(k, 4)
        y = reverse(permutedims(y, (2, 1, 3)); dims = 1)   # one turn, as rotl90
    end
    return y
end

"""
    augment(e::PathMNIST, x, rng) -> array

Training-time augmentation for one `(W, H, C)` tile, controlled by Host fields on the
experiment, so changing them recompiles nothing. With `aug_rot90` and `aug_flip` both on,
all 8 orientations are equally likely.
"""
function augment(e::PathMNIST, x, rng)
    k = e.aug_rot90 ? rand(rng, 0:3) : 0
    mirror = e.aug_flip && rand(rng, Bool)
    return dihedral(x, k, mirror)
end
