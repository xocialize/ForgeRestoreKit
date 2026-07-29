# ForgeRestoreKit

Pixel quality at constant resolution — the classical, net-clean half.

> ⚠️ **Deliberately partial.** Today this repo ships **`RestoreTier0` only**. The umbrella product, the
> engine-backed `RestoreTier2` half and the `FrameRestoreProvider` seam are held back on purpose: their
> job vocabulary becomes public API the moment it is tagged, and that naming decision is still open.
> Nothing here names a job, so nothing here preempts it.

No MLX, no engine, no weights, no downloads, no vendored binaries. macOS 14.

**"No Metal requirement" is not "no Metal".** The package must build and run without a GPU, and it
does — `GuidedFilterMetal.shared` is `nil` with no device and every caller has a CPU path. The shader
compiles at runtime from a source string, so there is no `.metal` file and no metallib to package. The
CPU implementation stays the oracle and parity is tested, not assumed.

⚠️ **The plane API is the reference layer, not the app surface.** These take and return `[Float]` luma.
An app driving an `MTKView` from `MTLTexture`s wants `CVPixelBuffer`-backed-by-`IOSurface` currency
end to end; accelerating one stage does not close that, and it is tracked separately.

## Noise-aware sharpening

Multi-band guided-filter detail enhancement with an Immerkaer noise estimate, coring, a Polesel
variance gate, Sobel edge roll-off and an asymmetric halo clamp.

```swift
// CPU, or the Metal backend when a device is available — same answers, parity-tested.
let filter = GuidedFilterMetal.shared ?? GuidedFilter()
let (sharpened, report) = NoiseAwareSharpener(filter: filter).sharpen(luma, width: w, height: h)
report.noiseSigma            // measured, or the value you supplied
report.appliedBandDiameters  // bands wider than the picture are skipped
report.clampedFraction       // how hard the halo clamp had to work
```

**Sharpen is DSP; Deblur / Lens Blur / Motion Blur are models.** Ship them in separate sections. The
competitor conflates them under one panel and that conflation is the documented source of their
over-sharpening complaints.

**What it enhances is texture, not edges.** A high-amplitude transition has variance far above the
detail scale, so the guided filter keeps it in the base layer and declines to boost it — that is the
halo prevention, and it is why this beats an unsharp mask. Chroma is never touched: the signature takes
a luma plane, so colour fringing is impossible rather than merely discouraged.

**Two knobs, not one.** `detailScale` says what counts as detail (halo control); σ says what counts as
noise (coring and the variance gate). `ε = max(detailScale, k·σ)²`. Collapsing them — tying ε to the
noise floor alone, as is sometimes suggested — makes the sharpener a **no-op on clean images**, because
σ→0 drives ε→0 and the guided filter becomes an identity.

## Film grain — AFGS1

A deterministic, seedable film-grain synthesizer: a Gaussian draw, a causal auto-regressive filter, a
brightness-dependent scaling curve, and the 2-sample overlap blend that hides the 32-px block grid.

```swift
let parameters = FilmGrainParameters.preset(.silverRich, seed: 0x4242, size: 0.6, amount: 0.4)
let grain = try FilmGrainSynthesizer(parameters: parameters)

// Full frame…
let output = grain.apply(to: luma, width: w, height: h)

// …or a viewport tile, which produces bit-identical pixels for the same region.
let tile = grain.apply(to: tileLuma, width: 512, height: 512, originX: 1024, originY: 768)
```

### Grain must not move when the picture does

Grain is addressed by **absolute** position, so a tile rendered at an arbitrary offset carries
bit-identical grain to the same region of a full-frame render. This is the whole reason the synthesizer
carries a seed and walks its own LFSR rather than using `CIRandomGenerator`, which exposes no seed and
no properties: with it, reproducing a grain field across renders, tiles or undo steps is not merely
untested, it is impossible — and grain that shimmers when the user pans is an immediate quality
complaint.

### Presets

| Preset | Shape |
|---|---|
| `.gaussian` | Lag 0, uncorrelated. The scaling curve is derived from photon-noise physics — amplitude rises with the square root of the signal. |
| `.grey` | Luma-only, moderate correlation, broadly flat curve with a shoulder roll-off. |
| `.silverRich` | Lag 3, and a curve peaked in the midtones and rolled off hard in blacks and whites — emulsion's tonal signature. |

`size` and `amount` are separate axes on purpose. Size is the AR feedback gain: how large a clump of
grain is. Amount is a gain on the scaling curve: how strongly that structure shows. Folding them into
one "strength" slider loses the control that makes grain read as film rather than as noise.

### Conformance status — read before claiming AFGS1 compatibility

This is a **structurally-AFGS1 engine, not a verified-conformant one.** For applying grain as an effect
that distinction does not matter; for round-tripping AFGS1 metadata or matching a decoder bit-for-bit it
does, and this engine is not there yet.

The substantive gap is the Gaussian source. AV1 specifies a fixed 2048-entry table; it is data, it is not
derivable, and it is not reproduced here — a table that is 99% right would look correct and be silently
wrong. So the sequence is **injectable**, and the default is a documented substitute matching the real
table's observable shape (2048 entries, ≈N(0, 512²), quantized to multiples of 4). Reaching conformance
is mechanical: take `gaussian_sequence` from a normative source — `dav1d/src/tables.c` (BSD-2) or
libplacebo's `film_grain_av1.c` (MIT-dual-licensed out of that LGPL project, per its own header; **not**
FFmpeg's `aom_film_grain.c`, which is LGPL) — pass it as `gaussianSequence`, and verify the constants
marked `SPEC` in `AFGS1Spec.swift` against the same source.

## Licence

MIT. See `LICENSE`.
