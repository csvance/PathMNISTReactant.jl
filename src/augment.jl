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
    hsv_jitter(x, dh, s, v) -> Array{UInt8,3}

Shift hue by `dh` (a fraction of a full turn) and scale saturation by `s` and value
by `v`, per pixel of a `(W, H, C)` RGB tile in `UInt8`. Pixels with no saturation
(the white background, gray debris) have no hue to shift, so they only change in
brightness.
"""
function hsv_jitter(x::AbstractArray{UInt8,3}, dh::Real, s::Real, v::Real)
    y = similar(x, UInt8, size(x))
    dh, s, v = Float32(dh), Float32(s), Float32(v)
    @inbounds for j in axes(x, 2), i in axes(x, 1)
        r, g, b = x[i, j, 1] / 255.0f0, x[i, j, 2] / 255.0f0, x[i, j, 3] / 255.0f0
        hi, lo = max(r, g, b), min(r, g, b)
        c = hi - lo
        # RGB to HSV, hue in [0, 1)
        h = c == 0 ? 0.0f0 :
            hi == r ? mod((g - b) / c, 6.0f0) / 6 :
            hi == g ? ((b - r) / c + 2) / 6 : ((r - g) / c + 4) / 6
        sat = hi == 0 ? 0.0f0 : c / hi
        h, sat, val = mod(h + dh, 1.0f0), min(sat * s, 1.0f0), min(hi * v, 1.0f0)
        # and back
        c = val * sat
        hp = 6h
        xx = c * (1 - abs(mod(hp, 2.0f0) - 1))
        r, g, b = hp < 1 ? (c, xx, 0.0f0) : hp < 2 ? (xx, c, 0.0f0) : hp < 3 ? (0.0f0, c, xx) :
                  hp < 4 ? (0.0f0, xx, c) : hp < 5 ? (xx, 0.0f0, c) : (c, 0.0f0, xx)
        m = val - c
        y[i, j, 1] = round(UInt8, 255 * clamp(r + m, 0.0f0, 1.0f0))
        y[i, j, 2] = round(UInt8, 255 * clamp(g + m, 0.0f0, 1.0f0))
        y[i, j, 3] = round(UInt8, 255 * clamp(b + m, 0.0f0, 1.0f0))
    end
    return y
end

"""
    augment(e::PathMNIST, x, rng) -> array

Training-time augmentation for one `(W, H, C)` tile, controlled entirely by
Host fields on the experiment, so changing any of them recompiles nothing:

- orientation: with `aug_rot90` and `aug_flip` both on, all 8 orientations are equally likely
- colour: an HSV jitter, so the model cannot lean on one lab's stain. Hue shifts by up to
  `±aug_hue` of a turn, saturation and value scale by up to `±aug_sat` and `±aug_val`.
  Setting all three to zero turns it off.
"""
function augment(e::PathMNIST, x, rng)
    if e.aug_hue > 0 || e.aug_sat > 0 || e.aug_val > 0
        u(a) = 2a * rand(rng) - a                          # uniform in [-a, a]
        x = hsv_jitter(x, u(e.aug_hue), 1 + u(e.aug_sat), 1 + u(e.aug_val))
    end
    k = e.aug_rot90 ? rand(rng, 0:3) : 0
    mirror = e.aug_flip && rand(rng, Bool)
    return dihedral(x, k, mirror)
end
