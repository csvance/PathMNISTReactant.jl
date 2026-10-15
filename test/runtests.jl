using PathMNISTReactant, ReactantNitro, Lux, MLUtils, Random, Test, ZipArchives

# A small .npz in the real file's format: C-order uint8 arrays, DEFLATE
# compressed, with every byte computed from its (n, h, w, c) position.
expected(n, h, w, c) = UInt8(((n - 1) * 31 + (h - 1) * 7 + (w - 1) * 3 + (c - 1) * 101) % 256)

function npy_bytes(shape, values)
    dict = "{'descr': '|u1', 'fortran_order': False, 'shape': (" * join(shape, ", ") * (length(shape) == 1 ? ",), }" : "), }")
    pad = 64 - (10 + length(dict) + 1) % 64
    header = dict * " "^pad * "\n"
    return vcat(b"\x93NUMPY\x01\x00", reinterpret(UInt8, [htol(UInt16(length(header)))]), codeunits(header), values)
end

function write_synthetic_npz(path; sizes = (train = 40, val = 20, test = 20))
    return ZipWriter(path) do zip
        for (split, N) in pairs(sizes)
            # C order (N, H, W, C): the last index varies fastest
            pixels = UInt8[expected(n, h, w, c) for c in 1:3, w in 1:224, h in 1:224, n in 1:N]
            for (name, bytes) in (
                    ("images", npy_bytes((N, 224, 224, 3), vec(pixels))),
                    ("labels", npy_bytes((N, 1), UInt8[(n - 1) % 9 for n in 1:N])),
                )
                zip_newfile(zip, "$(split)_$(name).npy"; compress = true)
                write(zip, bytes)
            end
        end
    end
end

const DIR = mktempdir()
write_synthetic_npz(joinpath(DIR, "synthetic.npz"))
const URL = "file://" * joinpath(DIR, "synthetic.npz")
const DATA_DIR = joinpath(DIR, "u8")
ensure_dataset_mmapable(PathMNIST(; data_url = URL, data_dir = DATA_DIR))

@testset "PathMNISTReactant" begin
    @testset "experiment fields" begin
        e = PathMNIST()
        @test ReactantNitro.host_fields(PathMNIST) ==
            (
            :data_url, :data_dir, :seed, :batch_size, :accum, :max_epochs, :max_lr, :node_lr_ratio,
            :pct_start, :weight_decay, :aug_rot90, :aug_flip,
        )
        @test ReactantNitro.device_fields(PathMNIST) == (:kinetic_weight,)
        @test (e.batch_size, e.accum) == (16, 2)
        @test PathMNIST(; max_epochs = 3).max_epochs == 3
    end

    @testset "download and conversion" begin
        # Only the six converted files remain: no archive, no .partial files
        @test sort(readdir(DATA_DIR)) == sort(["$(s)_$(k).u8" for s in ("train", "val", "test") for k in ("images", "labels")])
        @test filesize(joinpath(DATA_DIR, "train_images.u8")) == 40 * 3 * 224 * 224
        @test filesize(joinpath(DATA_DIR, "train_labels.u8")) == 40

        # A second call is a no-op
        e = PathMNIST(; data_url = URL, data_dir = DATA_DIR)
        stamp = mtime(joinpath(DATA_DIR, "train_images.u8"))
        ensure_dataset_mmapable(e)
        @test mtime(joinpath(DATA_DIR, "train_images.u8")) == stamp

        # A truncated file is detected and the set rebuilt
        dir = mktempdir()
        e2 = PathMNIST(; data_url = URL, data_dir = dir)
        ensure_dataset_mmapable(e2)
        open(io -> truncate(io, 1000), joinpath(dir, "val_images.u8"), "r+")
        @test !PathMNISTReactant.dataset_complete(dir)
        ensure_dataset_mmapable(e2)
        @test PathMNISTReactant.dataset_complete(dir)
        @test filesize(joinpath(dir, "val_images.u8")) == 20 * 3 * 224 * 224
    end

    @testset "split layout" begin
        d = PathMNISTSplit(PathMNIST(; data_dir = DATA_DIR), "train")
        @test numobs(d) == 40
        @test d.labels[1:10] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 1]       # 1-based
        b = getobs(d, [3, 17])
        @test size(b.img) == (224, 224, 3, 2) && eltype(b.img) == UInt8
        @test all(
            b.img[w, h, c, j] == expected(n, h, w, c)
                for (j, n) in enumerate([3, 17]), w in 1:224, h in 1:224, c in 1:3
        )
        @test b.label == d.labels[[3, 17]]
    end

    @testset "orientations" begin
        x = rand(Xoshiro(0), UInt8, 224, 224, 3)
        outs = vec([dihedral(x, k, m) for k in 0:3, m in (false, true)])
        @test length(unique(outs)) == 8
        @test eltype(outs[1]) == UInt8
        @test dihedral(x, 1, false) == cat((rotl90(x[:, :, c]) for c in 1:3)...; dims = 3)
        @test dihedral(x, 4, false) == x
        @test all(dihedral(o, 1, true) in outs for o in outs)          # closed under composition
        @test_throws ArgumentError dihedral(rand(UInt8, 4, 5, 3), 1, false)

        e, rng = PathMNIST(), Xoshiro(1)
        counts = zeros(Int, 8)
        for _ in 1:8000
            counts[findfirst(==(PathMNISTReactant.augment(e, x, rng)), outs)] += 1
        end
        @test all(800 .< counts .< 1200)                                # uniform over 8
        rot_only = PathMNIST(; aug_flip = false)
        @test all(PathMNISTReactant.augment(rot_only, x, rng) in outs[1:4] for _ in 1:200)
        off = PathMNIST(; aug_rot90 = false, aug_flip = false)
        @test all(PathMNISTReactant.augment(off, x, rng) == x for _ in 1:50)
    end

    @testset "augmented batches" begin
        e = PathMNIST(; data_dir = DATA_DIR)
        plain = getobs(PathMNISTSplit(e, "train"), [3, 17])
        d = PathMNISTSplit(e, "train"; augment = true)
        b = getobs(d, [3, 17])
        orients(x) = [dihedral(x, k, m) for k in 0:3, m in (false, true)]
        @test all(b.img[:, :, :, j] in orients(plain.img[:, :, :, j]) for j in 1:2)
        @test getobs(d, [3, 17]).img == b.img                           # reproducible per batch
    end

    @testset "build_data through Nitro's loader path" begin
        # build_data on an empty directory downloads and converts first
        fresh = mktempdir()
        data = ReactantNitro.build_data(PathMNIST(; data_url = URL, data_dir = fresh), nothing)
        @test PathMNISTReactant.dataset_complete(fresh)
        widths(dl) = (
            plan = ReactantNitro.begin_epoch!(dl);
            [size(ReactantNitro.batch_at(dl, i, plan).img, 4) for i in 1:length(dl)]
        )
        @test widths(data.train) == [16, 16]     # short batch dropped
        @test widths(data.val) == [16, 4]        # kept; Nitro pads it
        @test widths(data.test) == [16, 4]
        for (name, dl) in pairs(data)
            @test ReactantNitro.check_source_options(dl, name, (; path = :fanout)) === nothing
        end
    end

    @testset "loss" begin
        logits = randn(Xoshiro(3), Float32, 9, 5)
        label = [1, 4, 9, 2, 2]
        onehot = Float32.((1:9) .== label')
        out = (; logits, energy = fill(2.0f0, 1, 5))
        @test ReactantNitro.loss(PathMNIST(), out; label) ≈
            CrossEntropyLoss(; logits = Val(true))(logits, onehot)
        @test ReactantNitro.loss(PathMNIST(; kinetic_weight = 0.5f0), out; label) ≈
            CrossEntropyLoss(; logits = Val(true))(logits, onehot) + 1
    end

    @testset "model" begin
        model, ps, st = ReactantNitro.build_model(PathMNIST(), Xoshiro(0))
        @test Lux.parameterlength(ps) == 27_912_393
        @test keys(ps) == (:backbone, :norm, :head)
        @test keys(ps.head) == (:proj, :enc, :dec)
        fresh, _ = Lux.setup(Xoshiro(0), model)
        @test fresh.head == ps.head                                     # head left at init
        y, _ = model(rand(Xoshiro(2), UInt8, 224, 224, 3, 2), ps, Lux.testmode(st))
        @test size(y.logits) == (9, 2) && all(isfinite, y.logits)
        # The decoder starts at zero: no dynamics, so no energy, no motion, and the logits,
        # which live in the state and start at zero, stay there: a uniform first prediction.
        @test all(iszero, y.energy) && all(iszero, y.motion) && all(iszero, y.logits)

    end

    @testset "rk4" begin
        # du/dt = -u from u0 = 1: the exact answer at t = 1 is exp(-1).
        r = PathMNISTReactant.rk4(u -> -u, [1.0;;], 100)
        @test r.u[1] ≈ exp(-1) atol = 1.0e-10
        # u = exp(-t), so the kinetic energy is the integral of exp(-2t) over [0, 1].
        @test r.energy[1] ≈ (1 - exp(-2)) / 2 atol = 1.0e-8
    end

    @testset "metrics" begin
        e = PathMNIST(; num_classes = 3)
        logits = Float32[2 0 0; 0 2 0; 0 0 2]       # predicts classes 1, 2, 3
        out = (; logits, energy = Float32[1 2 3], motion = Float32[0 1 2])
        m = ReactantNitro.metrics(e, out; label = [1, 2, 1])
        @test m.confusion == ([1 0 1; 0 1 0; 0 0 0] => :sum)
        @test m.val_loss.second == 3
        @test m.calib.second === :sum
        @test sum(m.calib.first.samples) == 3 && sum(m.calib.first.correct) == 2
        # Reduced as Nitro would: the count divided, the sums kept.
        f = ReactantNitro.finalize_metrics(
            e,
            (;
                val_loss = m.val_loss.first / 3, confusion = m.confusion.first, calib = m.calib.first,
                motion = m.motion.first / 3, kinetic = m.kinetic.first / 3,
            ), :val
        )
        @test f.acc ≈ 2 / 3
        @test f.macro_acc ≈ (1 / 2 + 1) / 2                     # class 3 has no samples: left out
        conf = maximum(softmax(logits; dims = 1); dims = 1)[1]  # every prediction has this confidence
        @test f.ece ≈ abs(conf - 2 / 3) rtol = 1.0e-5
        @test f.kinetic ≈ 2 && f.motion ≈ 1
    end

    @testset "optimizer" begin
        e = PathMNIST()
        @test ReactantNitro.param_group(e, (:head, :dec, :weight)) === :node
        @test ReactantNitro.param_group(e, (:head, :cls, :weight)) === :default
        @test ReactantNitro.lambda(e, Val(:node)) == 0
        @test ReactantNitro.lambda(e) ≈ 1.0f-4
        @test ReactantNitro.learning_rate(PathMNIST(; node_lr_ratio = 0.1), Val(:node)) ≈ 1.0f-4
        s = ReactantNitro.schedules(e).eta(1000)
        @test s(1) ≈ 4.0f-5 && maximum(s.(1:1000)) ≈ 1.0f-3 && s(1000) < 1.0f-8
    end
end
