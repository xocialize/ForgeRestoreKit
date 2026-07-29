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

**Two surfaces, and it matters which one you use.** The `[Float]`-plane APIs are the reference
implementation and the parity oracle. An app driving an `MTKView` from `MTLTexture`s should use
`MetalSharpenPipeline`, which stays GPU-resident from `CVPixelBuffer` in to `CVPixelBuffer` out:

```swift
let pipeline = MetalSharpenPipeline()          // nil with no Metal device
let sharpened = pipeline?.process(frame) ?? frame
```

`process(_:) -> CVPixelBuffer` matches `ImageBridge.FrameProcessor` without importing media-bridge —
a host that wants the conformance writes `extension MetalSharpenPipeline: FrameProcessor {}`.

⚠️ **Do not mix the two.** Routing an `MTLTexture` app through the plane API gives
`texture → CPU plane → GPU kernel → CPU plane → texture`, which is **slower than staying on the CPU
entirely** — the transfers dominate a per-pixel kernel. That is why the pipeline moves the *whole*
chain (ingest, noise estimate, three bands, coring, gate, Sobel, clamp, write-back), touching the
buffer exactly twice.

Formats: packed BGRA, and biplanar 4:2:0 — where plane 1 is never bound, so "chroma is never touched"
is a fact of the layout rather than a discipline.

Grain has the same surface (`MetalFilmGrain`), so the two **chain without a readback** — both take and
return IOSurface-backed buffers, which the texture cache maps zero-copy:

```swift
let sharpened = sharpen.process(frame)
let finished  = grain.process(sharpened)
```

🚨 **A tiled renderer must pass the tile's true origin** — `grain.applied(to:originX:originY:)`.
Rendering every tile at `(0, 0)` is the call-site half of the shimmer-on-pan bug, and the synthesizer
cannot detect it for you. There is a test pinning that the two differ.

⚠️ Grain preserves colour differences by adding an equal delta to R, G and B — **except once a channel
clips**. Heavy grain on near-white or near-black colour desaturates slightly. That is a property of an
additive-delta model, not a defect; the fix, if it ever matters, is to scale the delta by the available
headroom.

## Deinterlacing

```swift
let progressive = try Deinterlace.frame(current: frame, previous: prev, next: next,
                                        width: w, height: h, parity: .top)
CombDetector.detect(frame, width: w, height: h).isInterlaced   // field-order / progressive check
```

🔑 **`lossless` is the point, and it is bit-exact.** Retained source field lines are provably the
original samples, not a reconstruction that scores well. No learned method can offer that, and for
archival restoration it is not a nice-to-have. Asserted by test.

**Why classical wins here is not a cost argument.** Every learned deinterlacer is trained by dropping
alternate lines from a *progressive* frame, which makes its two fields **co-temporal** — while real
50i/60i fields are a field-time apart, and that offset *is* the hard problem. Also absent from every
training set: field-order errors, blended and orphaned fields, 3:2 telecine, dot crawl, TBC error.

⚠️ **No FFmpeg.** The plan of record suggested linking FFmpeg's LGPL deinterlacers, and its licensing
analysis is correct — yadif/bwdif/w3fdif/estdif genuinely are LGPL and not GPL-gated. But the binding
constraint was never the licence: this project's doctrine is *"no FFmpeg, ever, no vendored binaries"*,
and a dynamic link is a vendored binary. So this is clean-room **edge-directed line averaging with
motion-adaptive temporal weighting** — a published technique, deliberately not a transcription of
yadif and not claimed to match it. There is no Apple fallback either: `VTFrameProcessor`'s seven
configs contain no deinterlacer.

## Hot and dead pixel correction

A RAW feature Apple does not expose at all. Operates on the **CFA mosaic, before demosaic** — the only
place it is well defined, since after demosaic a hot sensel's error is smeared across a neighbourhood
and into two other colour channels.

```swift
let (repaired, report) = try BadPixelCorrection.detectAndCorrect(
    cfa: mosaic, width: w, height: h,
    knownBadPixels: darkFrameMap)      // measured sites; corrected unconditionally
report.defectFraction                  // parts per million on a real sensor
report.phaseDetectPatternSuspected     // the sites look like a PDAF lattice
```

Detection is a signed deviation from the same-colour median **normalized by local high-frequency
energy** — which is the whole detector: an absolute threshold flags every fine texture, while
normalizing asks *"is this anomalous relative to how busy the neighbourhood already is"* and backs off
automatically. Repair uses the DNG SDK's `FixIsolatedPixel` rule: four directional estimates, and every
direction whose gradient is within 1.5× of the best is averaged — so a defect on an edge is repaired
*along* the edge rather than smeared across it.

### Two false-positive classes, and only one is solvable here

🔴 **A star is geometrically identical to a hot pixel.** No threshold separates them, because from one
frame there is nothing to separate — a test asserts the detector *does* fire on an isolated bright
point, so nobody later assumes it got smarter. The answers are a measured bad-pixel list (a dark frame,
or DNG opcode 5 — that is what `knownBadPixels` is for) or multi-frame agreement.

🟡 **PDAF sensels sit on a lattice**, which *is* detectable. A sparse pattern is flagged and still
repaired; a dense one throws `phaseDetectPatternDetected` rather than `implausibleDefectDensity`,
because the latter reads as "lower your threshold" and the right fix is a PDAF mask upstream.

🚨 An implausible density **aborts**. Past a few parts per million this stops being repair and becomes
a smoothing filter over every isolated detail in the frame.

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

### Match the original grain

The differentiated half: fit a grain model to a denoise **residual**, so re-grain reproduces the film
the source actually had rather than a preset that resembles it. The input is free — a restore pipeline
already produces the denoised image.

```swift
let estimate = try FilmGrainEstimator.fit(original: noisy, denoised: cleaned,
                                          width: w, height: h)
let matched  = try FilmGrainSynthesizer(parameters: estimate.parameters)
estimate.flatBlockFraction   // how much of the frame was usable — a confidence signal
estimate.residualSigma       // the measured noise level
```

Three stages: flat-block finding from the gradient covariance eigenvalues (fit where there is no
structure, or the model learns the *content*), AR coefficients by least squares over the causal
neighbourhood, and a scaling curve reduced to ≤14 knees.

🔑 **The curve is calibrated by measurement, not derived.** Output amplitude is a product of the
Gaussian source's spread, `grainScaleShift`, the AR gain, the curve and `grainScalingShift` — a closed
form through all of that is brittle and would silently break if the Gaussian source were swapped for
the normative AV1 table. Instead the estimator builds the template, measures its RMS, and solves the
curve against that.

Measured round trip — strength within 8%, coarseness within 10%:

| source | σ in → out | coarseness in → out |
|---|---|---|
| `.gaussian` | 0.0478 → 0.0471 | −0.011 → −0.042 *(both white)* |
| `.grey` | 0.0406 → 0.0374 | 0.183 → 0.169 |
| `.silverRich` | 0.0599 → 0.0548 | 0.426 → 0.467 |

⚠️ It does **not** recover the true coefficients, and is not trying to — fitting an AR model to one
noisy realization is ill-posed. The goal is a perceptual round trip: same strength, same coarseness.
An all-structure frame throws rather than returning a confident model of the picture.

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
