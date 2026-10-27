#region rk4
# Fixed-step RK4 over t in [0, 1]. Also returns each sample's kinetic energy,
# ∫|f(u)|² dt, from the four stage evaluations.
#
# `@trace for` compiles the loop as one XLA while loop instead of unrolling all
# 100 steps; unrolled, the gradient program took 29 minutes to compile on a cold
# cache. `@trace for` writes the final value of `u` back into the array the
# caller passed in, so the loop starts from a copy.
function rk4(f, u, steps::Integer)
    T = eltype(u)
    dt = one(T) / T(steps)
    half, sixth = dt / T(2), dt / T(6)
    sq(k) = sum(abs2, k; dims = 1)
    energy = zero(sq(u))
    u = copy(u)
    @trace for _ in 1:steps
        k1 = f(u)
        k2 = f(u .+ half .* k1)
        k3 = f(u .+ half .* k2)
        k4 = f(u .+ dt .* k3)
        u = u .+ sixth .* (k1 .+ T(2) .* k2 .+ T(2) .* k3 .+ k4)
        energy = energy .+ sixth .* (sq(k1) .+ T(2) .* sq(k2) .+ T(2) .* sq(k3) .+ sq(k4))
    end
    return (; u, energy)
end
#endregion

#region node_head
# Augmented NeuralODE head. The state is [a; z]: `a` comes from the pooled
# features, and `z` starts at zero and ends as the logits.
function node_head(e::PathMNIST, width::Integer)
    # captured, not passed to @compact (they would land in st)
    steps, A, K = e.node_steps, e.aug_dims, e.num_classes
    D = A + K
    return @compact(
        proj = Dense(width => A),
        enc = Dense(D => e.node_hidden, tanh),
        dec = Dense(e.node_hidden => D; init_weight = zeros32, init_bias = zeros32),
    ) do pooled                            # (width, N)
        a0 = proj(pooled)
        u0 = vcat(a0, zeros(eltype(a0), K) .* a0[1:1, :])   # z(0) = 0
        (; u, energy) = rk4(u -> dec(enc(u)), u0, steps)
        motion = sqrt.(sum(abs2, u .- u0; dims = 1) ./ sum(abs2, u0; dims = 1))
        @return (; logits = u[(A + 1):D, :], energy, motion)
    end
end
#endregion
