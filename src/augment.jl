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
    hsv_pixel(r, g, b, dh, s, v) -> NTuple{3,UInt8}

One pixel of the HSV jitter: to Float32, to HSV, shift hue by `dh` (a fraction of a
full turn) and scale saturation by `s` and value by `v`, then back to `UInt8`. A
pixel with no saturation (the white background) has no hue to shift, so it only
changes in brightness.
"""
@inline function hsv_pixel(r::UInt8, g::UInt8, b::UInt8, dh::Float32, s::Float32, v::Float32)
    r, g, b = r / 255.0f0, g / 255.0f0, b / 255.0f0
    hi, lo = max(r, g, b), min(r, g, b)
    c = hi - lo
    h = c == 0 ? 0.0f0 :
        hi == r ? mod((g - b) / c, 6.0f0) / 6 :
        hi == g ? ((b - r) / c + 2) / 6 : ((r - g) / c + 4) / 6
    sat = hi == 0 ? 0.0f0 : c / hi
    # Saturation and value are clipped to [0, 1] here too, so a scale factor of any sign stays in range.
    h, sat, val = mod(h + dh, 1.0f0), clamp(sat * s, 0.0f0, 1.0f0), clamp(hi * v, 0.0f0, 1.0f0)
    c = val * sat
    hp = 6h
    x = c * (1 - abs(mod(hp, 2.0f0) - 1))
    r, g, b = hp < 1 ? (c, x, 0.0f0) : hp < 2 ? (x, c, 0.0f0) : hp < 3 ? (0.0f0, c, x) :
              hp < 4 ? (0.0f0, x, c) : hp < 5 ? (x, 0.0f0, c) : (c, 0.0f0, x)
    m = val - c
    # Clipped to [0, 1] before scaling, so unsafe_trunc only sees [0.5, 255.5] and never
    # wraps; the + 0.5 makes the truncation round to nearest.
    q(u) = unsafe_trunc(UInt8, 255 * clamp(u + m, 0.0f0, 1.0f0) + 0.5f0)
    return (q(r), q(g), q(b))
end

"""
    hsv_jitter(x, dh, s, v) -> Array{UInt8,3}

[`hsv_pixel`](@ref) broadcast over a `(W, H, C)` RGB tile in `UInt8`.
"""
function hsv_jitter(x::AbstractArray{UInt8,3}, dh::Real, s::Real, v::Real)
    px = hsv_pixel.(view(x, :, :, 1), view(x, :, :, 2), view(x, :, :, 3), Float32(dh), Float32(s), Float32(v))
    return cat(getindex.(px, 1), getindex.(px, 2), getindex.(px, 3); dims = 3)
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
