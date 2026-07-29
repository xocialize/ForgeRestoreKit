//
// NoiseAwareSharpener.swift — RestoreTier0
//
// N1 — noise-aware sharpening. See `mlxengine-todo/GAP-PROGRAM.md` §N1.
//
// 🚨 **The product rule this type exists to enforce, and it is a UI rule as much as a code one:**
//
//     Ship "Sharpen" as DSP and "Deblur / Lens Blur / Motion Blur" as models, in separate sections.
//     The competitor conflates them under one panel and that conflation is the documented source of
//     their over-sharpening complaints. Do not copy it.
//
// Sharpening is a *preference*, not a reconstruction — there is no sharpening benchmark, no dataset and
// no NTIRE track, because you cannot write an L1 loss against "more pleasing". That is an argument for
// a well-behaved classical operator, not against one: the single architecturally-relevant paper,
// **HDRNet** (SIGGRAPH 2017, arXiv:1707.02880), was trained to *approximate existing classical
// operators* including local Laplacian detail manipulation.
//
// ═══════════════════════════════════════════════════════════════════════════════════════════════
// ⚠️ CORRECTION TO THE PLAN — measured 2026-07-29, and it changes the parameterization
// ═══════════════════════════════════════════════════════════════════════════════════════════════
//
// GAP-PROGRAM §N1 states: *"the noise-awareness collapses to one line: set the guided filter's
// ε = (k·σ)². Because ε is in intensity² units, any local variation below the noise floor is treated
// as flat and lands in base, never entering detail."*
//
// **Both halves of that are wrong, and the first build did nothing at all as a result** — on a clean
// edge the sharpened slope came back 0.04954 → 0.049555, i.e. unchanged.
//
//   • **Direction.** The guided filter's `a = var/(var + ε)` means high-variance regions are
//     *preserved* (`q ≈ I`) and low-variance regions are *smoothed* (`q ≈ mean`). Since
//     `detail = I − base`, sub-ε variation is exactly what **does** end up in detail. Noise lands in
//     the detail layer, not the base layer — the opposite of the claim.
//   • **Magnitude.** On a clean source σ→0, so ε→0, so `a ≈ 1` everywhere, so `base ≈ I` and
//     `detail ≈ 0`. Tying ε to the noise floor alone makes the sharpener a **no-op on exactly the
//     clean images it is most wanted on.**
//
// 🔑 **What the guided filter is actually for here is halo prevention, and that is worth having.**
// A true edge has variance ≫ ε, so it stays in *base* and is never boosted — which is why this beats
// unsharp masking, and why He/Sun/Tang's freedom from gradient reversal matters. So **ε is a
// detail-scale control**: large enough that texture enters detail, small enough that edges do not.
//
// **Noise-awareness lives in the two stages that follow**, and they were always in the plan — it just
// attributed the work to the wrong stage:
//   • **coring** `d′ = sign(d)·max(0, |d| − c·σ)` removes sub-noise wobble from the detail layer;
//   • the **Polesel variance gate** `g = g_max·a²/(a² + (k·σ)²)` withholds gain where there is nothing
//     but noise to amplify. Note it is keyed to `(k·σ)²` — *not* to ε. Reusing ε here (as the first
//     build did) couples the two knobs back together and reintroduces the bug.
//
// So this type carries **two** parameters where the plan carried one: `detailScale` (what counts as
// detail) and σ (what counts as noise). `ε = max(detailScale, k·σ)²` — the detail scale normally
// governs, and the noise floor becomes a floor under it, because you cannot meaningfully sharpen
// structure finer than the noise you are sitting in.
//
// 🔑 **The filter is injected, and a Metal backend ships alongside** — `GuidedFilterMetal.shared`
// conforms to the same `EdgeAwareFilter`, so
// `NoiseAwareSharpener(filter: GuidedFilterMetal.shared ?? GuidedFilter())` is the whole integration.
// ⚠️ Category A's rule is *"no Metal **requirement**"*, not "no Metal": the package must build and run
// without a GPU, which `init?()` returning nil guarantees. `media-bridge` is the in-tree precedent —
// category A, and it ships `SSIMULACRA2Metal`.
//
// ⚠️ **Chroma is never touched.** Sharpening Cb/Cr buys nothing visible and manufactures colour fringes
// on every edge. This type takes and returns a luma plane, so that is enforced by the signature rather
// than by a comment a caller can ignore.
//

import Foundation

public struct NoiseAwareSharpener: Sendable {

    public struct Options: Sendable {
        /// Overall strength. Scales the per-band gains.
        public var amount: Double
        /// Band diameters in pixels. The plan's working point is 5 / 17 / 65 — fine texture, mid detail,
        /// local contrast.
        public var bandDiameters: [Int]
        /// Per-band gain before `amount`. Coarser bands get less: pushing the 65-px band is "clarity",
        /// and it is the one that looks cheap fastest.
        public var bandGains: [Double]
        /// 🔑 **What counts as *detail*, as an amplitude in [0, 1] units.** Variation below this scale
        /// enters the detail layer and can be boosted; variation above it is an edge and stays in base,
        /// which is what prevents halos. This is the knob the plan folded into the noise floor — see
        /// the correction at the top of this file.
        public var detailScale: Double
        /// `k` in the noise floor `k·σ`. Used as a *floor* under `detailScale`, and as the knee of the
        /// Polesel gate. The plan's range is 1.5–2.5.
        public var noiseFloorK: Double
        /// `c` in the coring knee `d′ = sign(d)·max(0, |d| − c·σ)`.
        public var coringC: Double
        /// Ceiling for the Polesel variance gate.
        public var maxGain: Double
        /// Above this gradient magnitude the gain rolls off — hard edges are where halos are born.
        public var edgeKnee: Double
        /// Envelope clamp: how far below the local minimum the result may go, as a fraction of the
        /// local range.
        public var undershootAllowance: Double
        /// …and above the local maximum. **Must stay below `undershootAllowance`** — see the note on
        /// `clampToEnvelope`.
        public var overshootAllowance: Double
        /// Override the measured σ. `nil` measures it with Immerkaer.
        public var noiseSigma: Float?

        public init(amount: Double = 1.0,
                    bandDiameters: [Int] = [5, 17, 65],
                    bandGains: [Double] = [1.0, 0.6, 0.3],
                    detailScale: Double = 0.05,
                    noiseFloorK: Double = 2.0,
                    coringC: Double = 1.0,
                    maxGain: Double = 1.0,
                    edgeKnee: Double = 0.25,
                    undershootAllowance: Double = 0.35,
                    overshootAllowance: Double = 0.15,
                    noiseSigma: Float? = nil) {
            self.amount = amount
            self.bandDiameters = bandDiameters
            self.bandGains = bandGains
            self.detailScale = detailScale
            self.noiseFloorK = noiseFloorK
            self.coringC = coringC
            self.maxGain = maxGain
            self.edgeKnee = edgeKnee
            self.undershootAllowance = undershootAllowance
            self.overshootAllowance = overshootAllowance
            self.noiseSigma = noiseSigma
        }

        public static let `default` = Options()
    }

    /// What the sharpener measured and applied — the material for a receipt. **`FORGE-UI-REQUIREMENTS`
    /// §1.1 requires the applied parameters to be user-visible**, and a sharpener that silently
    /// measures its own noise floor is exactly the kind of stage that needs to say what it decided.
    public struct Report: Sendable, Equatable {
        /// The σ used, whether measured or supplied.
        public let noiseSigma: Float
        /// True when `Options.noiseSigma` overrode the measurement.
        public let sigmaWasSupplied: Bool
        /// Mean absolute change, as a fraction of full scale — "how much did this actually do".
        public let meanAbsoluteChange: Double
        /// Fraction of pixels the envelope clamp had to pull back. A large number means the gains are
        /// past what the picture supports, and it is the honest signal for an auto-strength control.
        public let clampedFraction: Double
        /// 🔑 **Which band diameters actually ran.** A band wider than the picture is meaningless — see
        /// `sharpen`. This matters beyond diagnostics: a proxy render and a full-resolution export will
        /// apply *different* band sets on a small proxy, so a UI comparing them is not comparing like
        /// with like unless it checks this.
        public let appliedBandDiameters: [Int]
    }

    public let options: Options
    private let filter: any EdgeAwareFilter

    public init(options: Options = .default, filter: any EdgeAwareFilter = GuidedFilter()) {
        self.options = options
        self.filter = filter
    }

    /// Sharpen a **gamma-encoded** luma plane in [0, 1].
    ///
    /// The order is the plan's, and each stage answers a different failure:
    ///   1. estimate σ (Immerkaer) — everything downstream is expressed in it
    ///   2. multi-band guided-filter split with `ε = (k·σ)²` — noise never enters detail
    ///   3. coring — kills the residual sub-noise wobble a hard threshold would leave ringing
    ///   4. Polesel variance gate — no gain in flat areas, where gain only amplifies what is left
    ///   5. Sobel roll-off — no gain on hard edges, where gain becomes a halo
    ///   6. recombine, then the asymmetric envelope clamp — the last line against overshoot
    public func sharpen(_ plane: [Float], width: Int, height: Int) -> (plane: [Float], report: Report) {
        precondition(plane.count == width * height)
        guard width >= 3, height >= 3, !plane.isEmpty else {
            return (plane, Report(noiseSigma: 0, sigmaWasSupplied: options.noiseSigma != nil,
                                  meanAbsoluteChange: 0, clampedFraction: 0,
                                  appliedBandDiameters: []))
        }

        let sigma = options.noiseSigma ?? NoiseEstimate.immerkaer(plane, width: width, height: height)
        let noiseFloor = options.noiseFloorK * Double(sigma)
        // Detail scale governs; the noise floor is a floor under it. See the correction at the top.
        let epsilon = Float(pow(max(options.detailScale, noiseFloor), 2))
        // The Polesel gate is keyed to the NOISE floor, not to epsilon — they answer different
        // questions and coupling them is what broke the first build.
        let gateKnee = pow(noiseFloor, 2)
        let coringThreshold = Float(options.coringC * Double(sigma))

        // Sobel is computed once on the source: the edge map describes the *picture*, not the
        // partially-sharpened intermediate.
        let gradient = sobelMagnitude(plane, width: width, height: height)

        var current = plane
        var accumulated = [Float](repeating: 0, count: plane.count)

        // ⚠️ **A band wider than the picture is not a band, it is a DC offset.** Its box window clamps
        // to most of the frame, so `base` becomes a near-global mean and `detail` becomes "how far this
        // pixel is from the image average" — a huge low-frequency term that the envelope clamp then has
        // to fight everywhere. On a 64-px test frame the 65-px band alone drove the clamp to 94% of
        // pixels. Skipping is also the right behaviour for previews and proxies, which are exactly where
        // small frames show up in production.
        var appliedDiameters: [Int] = []
        for (index, diameter) in options.bandDiameters.enumerated() {
            guard diameter <= min(width, height) else { continue }
            appliedDiameters.append(diameter)
            let radius = max(1, diameter / 2)
            let base = filter.base(of: current, width: width, height: height,
                                   radius: radius, epsilon: epsilon)
            let variance = GuidedFilter.localVariance(current, width: width, height: height,
                                                      radius: radius)
            let gain = options.amount * (index < options.bandGains.count ? options.bandGains[index] : 0)

            for i in 0..<plane.count {
                let detail = current[i] - base[i]

                // 3 · Smooth-knee coring.
                let magnitude = abs(detail)
                let cored: Float = magnitude <= coringThreshold
                    ? 0
                    : (detail < 0 ? -(magnitude - coringThreshold) : magnitude - coringThreshold)

                // 4 · Polesel variance gate (IEEE TIP 2000, DOI 10.1109/83.826787):
                //     g = g_max · a² / (a² + (kσ)²), with `a` the local standard deviation so the two
                //     terms share units. Flat areas get no gain — which is where sharpening does
                //     nothing but amplify whatever survived step 3.
                let localStd = variance[i].squareRoot()
                let a2 = Double(localStd * localStd)
                let variancePart = options.maxGain * a2 / (a2 + gateKnee + 1e-12)

                // 5 · Sobel roll-off. Hard edges are already sharp; adding gain there is how a halo is
                //     made, and a halo is the artifact users name.
                let edge = Double(gradient[i]) / options.edgeKnee
                let edgePart = 1.0 / (1.0 + edge * edge)

                accumulated[i] += Float(gain * variancePart * edgePart) * cored
            }
            current = base
        }

        // 6 · Recombine and clamp.
        //
        // ⚠️ **Add the boost to the ORIGINAL, not to the coarsest base.** The decomposition telescopes
        // exactly — `I = base₃ + d₃ + d₂ + d₁` — so `I + Σ gᵢ·d′ᵢ` is the enhanced image, and
        // `amount = 0` is then the identity by construction. Reconstructing from `base₃` instead makes
        // a zero gain return a heavily *smoothed* picture, which is a silent, catastrophic no-op: the
        // control appears to work (the image changes) while doing the opposite of its label.
        _ = current
        var out = [Float](repeating: 0, count: plane.count)
        for i in 0..<plane.count { out[i] = plane[i] + accumulated[i] }
        let clamped = clampToEnvelope(out, source: plane, width: width, height: height,
                                      noiseFloor: Float(noiseFloor))

        var change = 0.0
        for i in 0..<plane.count { change += abs(Double(clamped.plane[i] - plane[i])) }

        return (clamped.plane,
                Report(noiseSigma: sigma,
                       sigmaWasSupplied: options.noiseSigma != nil,
                       meanAbsoluteChange: change / Double(plane.count),
                       clampedFraction: Double(clamped.clampedCount) / Double(plane.count),
                       appliedBandDiameters: appliedDiameters))
    }

    /// Asymmetric local min/max envelope clamp.
    ///
    /// 🔑 **The asymmetry is the point, and it is a perceptual fact rather than a tuning choice: bright
    /// halos are far more objectionable than dark ones.** So the result is allowed to undershoot the
    /// local minimum considerably further than it may overshoot the local maximum. A symmetric clamp
    /// with the same total budget looks visibly worse on exactly the high-contrast edges — backlit
    /// hair, a roofline against sky — where sharpening artifacts get noticed.
    /// ⚠️ **The allowance never collapses to zero, and the first build's did.** In a flat region the
    /// local range is 0, so a range-proportional allowance pins the output to the source exactly — every
    /// pixel then counts as clamped, and `clampedFraction` reads 94% on a picture that is mostly sky,
    /// which destroys it as a signal for an auto-strength control. Flooring the allowance at the noise
    /// scale is also just correct: an excursion smaller than the noise is not an artifact.
    func clampToEnvelope(_ sharpened: [Float], source: [Float], width: Int, height: Int,
                         noiseFloor: Float) -> (plane: [Float], clampedCount: Int) {
        let radius = 2
        var out = sharpened
        var count = 0

        for y in 0..<height {
            for x in 0..<width {
                var localMin = Float.greatestFiniteMagnitude
                var localMax = -Float.greatestFiniteMagnitude
                for dy in -radius...radius {
                    let sy = min(max(y + dy, 0), height - 1)
                    for dx in -radius...radius {
                        let sx = min(max(x + dx, 0), width - 1)
                        let v = source[sy * width + sx]
                        localMin = min(localMin, v)
                        localMax = max(localMax, v)
                    }
                }
                let range = max(localMax - localMin, noiseFloor)
                let low = localMin - Float(options.undershootAllowance) * range
                let high = localMax + Float(options.overshootAllowance) * range

                let i = y * width + x
                let value = out[i]
                if value < low { out[i] = low; count += 1 }
                else if value > high { out[i] = high; count += 1 }
                out[i] = min(max(out[i], 0), 1)
            }
        }
        return (out, count)
    }

    /// 3×3 Sobel gradient magnitude, edge-clamped.
    func sobelMagnitude(_ plane: [Float], width: Int, height: Int) -> [Float] {
        var out = [Float](repeating: 0, count: plane.count)
        @inline(__always) func at(_ x: Int, _ y: Int) -> Float {
            plane[min(max(y, 0), height - 1) * width + min(max(x, 0), width - 1)]
        }
        for y in 0..<height {
            for x in 0..<width {
                let gx = -at(x - 1, y - 1) - 2 * at(x - 1, y) - at(x - 1, y + 1)
                       + at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1)
                let gy = -at(x - 1, y - 1) - 2 * at(x, y - 1) - at(x + 1, y - 1)
                       + at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1)
                out[y * width + x] = (gx * gx + gy * gy).squareRoot()
            }
        }
        return out
    }
}
